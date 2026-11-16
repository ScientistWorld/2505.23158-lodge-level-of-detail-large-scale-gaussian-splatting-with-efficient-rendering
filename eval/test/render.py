"""Hidden-evaluator copy of the public renderer (data/gsrender.py) -- keep identical below this line.

This is the public reference implementation of how a submitted representation
is rendered and measured.  Method code may use it for optimisation, and the
visible evaluator (``eval/train``) uses it for scoring.  The hidden evaluator
holds its own identical copy, so editing this file never changes test scores.

Renderer
--------
Standard 3DGS rasterisation: EWA projection of anisotropic 3D Gaussians, the
Mip-Splatting 2D screen-space filter (variance ``MIP_KERNEL`` px^2 with opacity
compensation), 16x16 pixel tiles, per-tile front-to-back alpha compositing with
alpha clamped to [1/255, 0.99] and early termination at transmittance 1e-4.
Everything is written in plain PyTorch so it is differentiable.

Representation protocol (``representation.npz``)
-----------------------------------------------
A pool of N Gaussians plus K precomputed resident sets ("chunks"):

* ``means`` (N,3), ``scales`` (N,3) standard deviations in metres, ``quats``
  (N,4) as (w,x,y,z), ``opacities`` (N,), ``colors`` (N,3) RGB in [0,1].
* optional ``band_lo``/``band_hi`` (N,): a resident Gaussian is drawn only when
  its distance to the camera lies in ``[band_lo, band_hi)`` (default
  ``[0, inf)``); this expresses per-frame distance-based level selection.
* ``chunk_centers`` (K,3), ``chunk_offsets`` (K+1,), ``chunk_indices`` (M,):
  CSR lists of pool indices resident when a chunk is in use.
* ``blend_two_nearest`` scalar 0/1.

For a camera at position ``c`` the renderer finds the nearest chunk centre
(ties broken by index).  With ``blend_two_nearest == 0`` (or K == 1) the
resident set is that chunk.  With ``blend_two_nearest == 1`` the resident set is
the union of the two nearest chunks; Gaussians present in only one of them have
their opacity multiplied by ``t = clip((c - m_o).(m_f - m_o) / |m_f - m_o|^2, 0, 1)``
where ``m_f`` is the centre of the chunk holding the Gaussian and ``m_o`` the
other one.

Measured per view: ``loaded`` = size of the resident set (memory), ``visible``
= resident Gaussians with non-zero effective opacity >= 1/255 whose projected
3-sigma footprint lies in front of the camera and overlaps the image (the
Gaussians the rasteriser must sort and process), ``tile_pairs`` = number of
Gaussian/tile intersections.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TILE = 16
NEAR = 0.2
MIP_KERNEL = 0.1
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4
BACKGROUND = (0.70, 0.80, 0.92)

REQUIRED_ARRAYS = (
    "means",
    "scales",
    "quats",
    "opacities",
    "colors",
    "chunk_centers",
    "chunk_offsets",
    "chunk_indices",
    "blend_two_nearest",
)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------


@dataclass
class Camera:
    eye: np.ndarray
    yaw: float
    pitch: float
    f: float
    width: int
    height: int

    def forward(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)], dtype=np.float64
        )

    def rotation(self) -> np.ndarray:
        """World-to-camera rotation; camera axes are x right, y down, z forward."""
        fwd = self.forward()
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd], axis=0)

    def to_dict(self) -> dict:
        return {"eye": [float(x) for x in self.eye], "yaw": float(self.yaw), "pitch": float(self.pitch)}

    @classmethod
    def from_dict(cls, d: dict, f: float, width: int, height: int) -> "Camera":
        return cls(np.asarray(d["eye"], dtype=np.float64), float(d["yaw"]), float(d["pitch"]), float(f), int(width), int(height))

    def scaled(self, factor: int) -> "Camera":
        return Camera(self.eye, self.yaw, self.pitch, self.f * factor, self.width * factor, self.height * factor)


def cameras_from_arrays(eye: np.ndarray, yaw: np.ndarray, pitch: np.ndarray, f: float, width: int, height: int) -> list[Camera]:
    return [Camera(np.asarray(e, dtype=np.float64), float(y), float(p), f, width, height) for e, y, p in zip(eye, yaw, pitch)]


# --------------------------------------------------------------------------
# Gaussian geometry
# --------------------------------------------------------------------------


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-12)
    w, x, y, z = q.unbind(-1)
    r = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return r.view(-1, 3, 3)


def project_gaussians(means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor, cam: Camera) -> dict:
    device = means.device
    rot = torch.as_tensor(cam.rotation(), dtype=torch.float32, device=device)
    eye = torch.as_tensor(cam.eye, dtype=torch.float32, device=device)
    pc = (means - eye) @ rot.T
    x, y, z = pc.unbind(-1)
    valid = z > NEAR
    zs = torch.clamp(z, min=NEAR)
    f = float(cam.f)
    limx = 1.3 * 0.5 * cam.width / f
    limy = 1.3 * 0.5 * cam.height / f
    tx = torch.clamp(x / zs, -limx, limx)
    ty = torch.clamp(y / zs, -limy, limy)
    u = f * x / zs + 0.5 * cam.width
    v = f * y / zs + 0.5 * cam.height

    rg = quat_to_rotmat(quats)
    m = rg * scales[:, None, :]
    cov_w = m @ m.transpose(1, 2)
    cov_c = rot @ cov_w @ rot.T
    zero = torch.zeros_like(zs)
    row0 = torch.stack([f / zs, zero, -f * tx / zs], dim=-1)
    row1 = torch.stack([zero, f / zs, -f * ty / zs], dim=-1)
    jac = torch.stack([row0, row1], dim=1)
    cov2 = jac @ cov_c @ jac.transpose(1, 2)
    a = cov2[:, 0, 0]
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1]
    det0 = torch.clamp(a * c - b * b, min=1e-12)
    a2 = a + MIP_KERNEL
    c2 = c + MIP_KERNEL
    det = torch.clamp(a2 * c2 - b * b, min=1e-12)
    comp = torch.sqrt(torch.clamp(det0 / det, max=1.0))
    conic = torch.stack([c2 / det, -b / det, a2 / det], dim=-1)
    mid = 0.5 * (a2 + c2)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam.detach()))
    return {
        "mean2d": torch.stack([u, v], dim=-1),
        "conic": conic,
        "comp": comp,
        "depth": z,
        "radius": radius,
        "valid": valid,
    }


# --------------------------------------------------------------------------
# Rasterisation
# --------------------------------------------------------------------------


MAX_PAIRS_PER_GROUP = 1 << 17


def _rasterize(mean2d, conic, opac, colors, depth, radius, width, height, background, return_importance):
    """Vectorised tile rasteriser.

    Gaussian/tile pairs are sorted by tile and, within a tile, front to back.
    Tiles are processed in groups of at most ``MAX_PAIRS_PER_GROUP`` pairs; the
    per-tile exclusive transmittance is a segmented cumulative sum of
    log(1 - alpha) in float64 (identical compositing to a per-tile loop).
    """
    device = mean2d.device
    ntx, nty = width // TILE, height // TILE
    ntiles = ntx * nty
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    n = mean2d.shape[0]
    importance = torch.zeros(n, dtype=torch.float32, device=device) if return_importance else None
    if n == 0:
        return bg.expand(height, width, 3).clone(), 0, importance

    u = mean2d[:, 0].detach()
    v = mean2d[:, 1].detach()
    r = radius
    tx0 = torch.clamp(torch.floor((u - r) / TILE), 0, ntx - 1).long()
    tx1 = torch.clamp(torch.floor((u + r) / TILE), 0, ntx - 1).long()
    ty0 = torch.clamp(torch.floor((v - r) / TILE), 0, nty - 1).long()
    ty1 = torch.clamp(torch.floor((v + r) / TILE), 0, nty - 1).long()
    wt = tx1 - tx0 + 1
    cnt = wt * (ty1 - ty0 + 1)

    order = torch.sort(depth.detach(), stable=True).indices
    cnt_o = cnt[order]
    total = int(cnt_o.sum().item())
    gid = torch.repeat_interleave(order, cnt_o)
    first = torch.repeat_interleave(torch.cumsum(cnt_o, 0) - cnt_o, cnt_o)
    k = torch.arange(total, device=device) - first
    wt_r = wt[gid]
    tile = (ty0[gid] + torch.div(k, wt_r, rounding_mode="floor")) * ntx + (tx0[gid] + k % wt_r)
    srt = torch.sort(tile, stable=True)
    gid = gid[srt.indices]
    tl = srt.values
    counts = torch.bincount(tl, minlength=ntiles)
    starts = torch.cumsum(counts, 0) - counts
    counts_l = counts.tolist()
    starts_l = starts.tolist()

    lin = torch.arange(TILE * TILE, device=device)
    base_x = (lin % TILE).float() + 0.5
    base_y = torch.div(lin, TILE, rounding_mode="floor").float() + 0.5

    groups, cur, cur_pairs = [], [], 0
    for t in range(ntiles):
        c = counts_l[t]
        if cur and cur_pairs + c > MAX_PAIRS_PER_GROUP:
            groups.append(cur)
            cur, cur_pairs = [], 0
        cur.append(t)
        cur_pairs += c
    if cur:
        groups.append(cur)

    out_tiles = []
    for grp in groups:
        t0, t1 = grp[0], grp[-1]
        nt = t1 - t0 + 1
        p0 = starts_l[t0]
        p1 = starts_l[t1] + counts_l[t1]
        if p1 == p0:
            out_tiles.append(bg.expand(nt, TILE * TILE, 3))
            continue
        g = gid[p0:p1]
        tlg = tl[p0:p1]
        txg = (tlg % ntx).float()
        tyg = torch.div(tlg, ntx, rounding_mode="floor").float()
        dx = (base_x[None, :] + txg[:, None] * TILE) - mean2d[g, 0:1]
        dy = (base_y[None, :] + tyg[:, None] * TILE) - mean2d[g, 1:2]
        cg = conic[g]
        power = -0.5 * (cg[:, 0:1] * dx * dx + cg[:, 2:3] * dy * dy) - cg[:, 1:2] * dx * dy
        power = torch.clamp(power, max=0.0)
        alpha = torch.clamp(opac[g, None] * torch.exp(power), max=ALPHA_MAX)
        alpha = torch.where(alpha >= ALPHA_MIN, alpha, torch.zeros_like(alpha))
        la = torch.log1p(-alpha.double())
        cs = torch.cumsum(la, dim=0)
        excl = cs - la
        seg_first = starts[tlg] - p0
        t_excl = torch.exp(excl - excl[seg_first]).float()
        alive = (t_excl.detach() > T_MIN).to(alpha.dtype)
        w = alpha * t_excl * alive
        local = tlg - t0
        col = colors[g]
        rgb = torch.stack(
            [torch.zeros(nt, TILE * TILE, dtype=w.dtype, device=device).index_add(0, local, w * col[:, ch:ch + 1]) for ch in range(3)],
            dim=-1,
        )
        cnt_g = counts[t0:t1 + 1]
        has = cnt_g > 0
        first_t = torch.clamp(starts[t0:t1 + 1] - p0, min=0)
        last_t = torch.clamp(starts[t0:t1 + 1] + cnt_g - 1 - p0, min=0)
        log_tf = cs[last_t] - excl[first_t]
        t_final = torch.where(has[:, None], torch.exp(log_tf).float(), torch.ones_like(log_tf, dtype=torch.float32))
        out_tiles.append(rgb + t_final[..., None] * bg)
        if importance is not None:
            importance.scatter_reduce_(0, g, w.detach().amax(dim=1), reduce="amax", include_self=True)
    img = torch.cat(out_tiles, 0).view(nty, ntx, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(height, width, 3)
    return img, total, importance


def render_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cam: Camera,
    weights: torch.Tensor | None = None,
    background=BACKGROUND,
    return_importance: bool = False,
) -> dict:
    """Render Gaussians; returns image (H,W,3), visible count, tile pairs and optional importance.

    ``importance`` (when requested) is each Gaussian's maximum alpha-blending
    weight over the pixels of this view (RadSplat importance for one view).
    """

    if cam.width % TILE or cam.height % TILE:
        raise ValueError("image size must be a multiple of the tile size")
    n = means.shape[0]
    device = means.device
    if n == 0:
        img, _, _ = _rasterize(
            torch.zeros((0, 2), device=device), torch.zeros((0, 3), device=device), torch.zeros(0, device=device),
            torch.zeros((0, 3), device=device), torch.zeros(0, device=device), torch.zeros(0, device=device),
            cam.width, cam.height, background, False,
        )
        imp = torch.zeros(0, device=device) if return_importance else None
        return {"image": img, "visible": 0, "tile_pairs": 0, "importance": imp}
    proj = project_gaussians(means, scales, quats, cam)
    opac = opacities if weights is None else opacities * weights
    u = proj["mean2d"][:, 0].detach()
    v = proj["mean2d"][:, 1].detach()
    r = proj["radius"]
    inside = (u + r > 0) & (u - r < cam.width) & (v + r > 0) & (v - r < cam.height)
    mask = proj["valid"] & inside & (opac.detach() >= ALPHA_MIN)
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    img, pairs, imp_sub = _rasterize(
        proj["mean2d"][idx],
        proj["conic"][idx],
        (opac * proj["comp"])[idx],
        colors[idx],
        proj["depth"][idx],
        r[idx],
        cam.width,
        cam.height,
        background,
        return_importance,
    )
    importance = None
    if return_importance:
        importance = torch.zeros(n, dtype=torch.float32, device=device)
        importance[idx] = imp_sub
    return {"image": img, "visible": int(idx.numel()), "tile_pairs": int(pairs), "importance": importance}


def render_supersampled(gauss: dict, cam: Camera, factor: int) -> torch.Tensor:
    """Render at ``factor``x resolution and box-average down (used for image targets)."""
    out = render_gaussians(gauss["means"], gauss["scales"], gauss["quats"], gauss["opacities"], gauss["colors"], cam.scaled(factor))
    img = out["image"].permute(2, 0, 1)[None]
    return F.avg_pool2d(img, factor)[0].permute(1, 2, 0)


# --------------------------------------------------------------------------
# Representation protocol
# --------------------------------------------------------------------------


class RepresentationError(ValueError):
    pass


def _finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise RepresentationError(f"{name} contains non-finite values")


def load_representation(path: Path, max_gaussians: int, max_chunks: int = 4096) -> dict:
    """Load and validate a representation file; raises RepresentationError if invalid."""
    path = Path(path)
    if not path.is_file():
        raise RepresentationError(f"missing {path}")
    try:
        with np.load(path, allow_pickle=False) as z:
            raw = {k: np.array(z[k]) for k in z.files}
    except Exception as exc:  # noqa: BLE001
        raise RepresentationError(f"cannot read {path}: {exc}") from exc
    for key in REQUIRED_ARRAYS:
        if key not in raw:
            raise RepresentationError(f"missing array {key}")
    numeric = ("means", "scales", "quats", "opacities", "colors", "chunk_centers", "band_lo", "band_hi")
    for key in numeric:
        if key in raw and not np.issubdtype(raw[key].dtype, np.number):
            raise RepresentationError(f"{key} must be numeric")
    means = raw["means"].astype(np.float32)
    n = means.shape[0] if means.ndim == 2 else -1
    if means.ndim != 2 or means.shape[1] != 3 or n < 1:
        raise RepresentationError("means must have shape (N,3) with N >= 1")
    if n > max_gaussians:
        raise RepresentationError(f"pool has {n} Gaussians, above the cap {max_gaussians}")
    shapes = {"scales": (n, 3), "quats": (n, 4), "opacities": (n,), "colors": (n, 3)}
    rep = {"means": means}
    for key, shp in shapes.items():
        arr = raw[key].astype(np.float32)
        if arr.shape != shp:
            raise RepresentationError(f"{key} must have shape {shp}, got {arr.shape}")
        rep[key] = arr
    for key in ("means", "scales", "quats", "opacities", "colors"):
        _finite(key, rep[key])
    if np.any(rep["scales"] <= 1e-7) or np.any(rep["scales"] > 100.0):
        raise RepresentationError("scales must lie in (1e-7, 100] metres")
    if np.any(np.linalg.norm(rep["quats"], axis=1) < 1e-6):
        raise RepresentationError("quaternions must be non-zero")
    if np.any(rep["opacities"] < 0) or np.any(rep["opacities"] > 1):
        raise RepresentationError("opacities must lie in [0,1]")
    if np.any(rep["colors"] < 0) or np.any(rep["colors"] > 1):
        raise RepresentationError("colors must lie in [0,1]")
    lo = raw.get("band_lo", np.zeros(n, dtype=np.float32)).astype(np.float32)
    hi = raw.get("band_hi", np.full(n, np.inf, dtype=np.float32)).astype(np.float32)
    if lo.shape != (n,) or hi.shape != (n,):
        raise RepresentationError("band_lo/band_hi must have shape (N,)")
    _finite("band_lo", lo)
    if np.any(np.isnan(hi)) or np.any(lo < 0) or np.any(hi < lo):
        raise RepresentationError("bands must satisfy 0 <= band_lo <= band_hi")
    rep["band_lo"], rep["band_hi"] = lo, hi

    centers = raw["chunk_centers"].astype(np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or not 1 <= centers.shape[0] <= max_chunks:
        raise RepresentationError(f"chunk_centers must have shape (K,3) with 1 <= K <= {max_chunks}")
    _finite("chunk_centers", centers)
    k = centers.shape[0]
    offsets = raw["chunk_offsets"]
    indices = raw["chunk_indices"]
    if not np.issubdtype(offsets.dtype, np.integer) or not np.issubdtype(indices.dtype, np.integer):
        raise RepresentationError("chunk_offsets/chunk_indices must be integer arrays")
    offsets = offsets.astype(np.int64)
    indices = indices.astype(np.int64)
    if offsets.shape != (k + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != indices.size:
        raise RepresentationError("chunk_offsets must be a CSR offset array of length K+1")
    if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= n)):
        raise RepresentationError("chunk_indices out of range")
    chunks = []
    for j in range(k):
        s = np.sort(indices[offsets[j]:offsets[j + 1]])
        if s.size > 1 and np.any(s[1:] == s[:-1]):
            raise RepresentationError(f"chunk {j} lists a Gaussian twice")
        chunks.append(s)
    blend = np.asarray(raw["blend_two_nearest"])
    if blend.size != 1 or not np.issubdtype(blend.dtype, np.number) or float(blend.reshape(-1)[0]) not in (0.0, 1.0):
        raise RepresentationError("blend_two_nearest must be a scalar 0 or 1")
    rep["chunk_centers"] = centers
    rep["chunks"] = chunks
    rep["blend_two_nearest"] = int(float(blend.reshape(-1)[0]))
    return rep


def save_representation(path: Path, gauss: dict, chunk_centers: np.ndarray, chunk_lists: list, blend_two_nearest: bool,
                        band_lo: np.ndarray | None = None, band_hi: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(gauss[k], dtype=np.float32) for k in ("means", "scales", "quats", "opacities", "colors")}
    lists = [np.asarray(c, dtype=np.int64) for c in chunk_lists]
    offsets = np.zeros(len(lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([c.size for c in lists])
    arrays["chunk_centers"] = np.asarray(chunk_centers, dtype=np.float32).reshape(-1, 3)
    arrays["chunk_offsets"] = offsets
    arrays["chunk_indices"] = np.concatenate(lists) if lists else np.zeros(0, dtype=np.int64)
    arrays["blend_two_nearest"] = np.array(1 if blend_two_nearest else 0, dtype=np.int64)
    if band_lo is not None:
        arrays["band_lo"] = np.asarray(band_lo, dtype=np.float32)
    if band_hi is not None:
        arrays["band_hi"] = np.asarray(band_hi, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def select_resident(rep: dict, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resident pool indices and per-Gaussian opacity weights for a camera position."""
    eye = np.asarray(eye, dtype=np.float64)
    centers = rep["chunk_centers"]
    chunks = rep["chunks"]
    dist = np.linalg.norm(centers - eye[None, :], axis=1)
    order = np.argsort(dist, kind="stable")
    a = int(order[0])
    if len(chunks) == 1 or rep["blend_two_nearest"] == 0:
        resident = chunks[a]
        weights = np.ones(resident.size, dtype=np.float32)
    else:
        b = int(order[1])
        ma, mb = centers[a], centers[b]
        denom = float(np.sum((ma - mb) ** 2))
        if denom <= 1e-12:
            ta = tb = 1.0
        else:
            ta = float(np.clip(np.dot(eye - mb, ma - mb) / denom, 0.0, 1.0))
            tb = float(np.clip(np.dot(eye - ma, mb - ma) / denom, 0.0, 1.0))
        ca, cb = chunks[a], chunks[b]
        resident = np.union1d(ca, cb)
        in_a = np.isin(resident, ca, assume_unique=True)
        in_b = np.isin(resident, cb, assume_unique=True)
        weights = np.where(in_a & in_b, 1.0, np.where(in_a, ta, tb)).astype(np.float32)
    if resident.size:
        d = np.linalg.norm(rep["means"][resident].astype(np.float64) - eye[None, :], axis=1)
        active = (d >= rep["band_lo"][resident]) & (d < rep["band_hi"][resident])
        weights = weights * active.astype(np.float32)
    return resident, weights


def representation_to_torch(rep: dict, device: torch.device) -> dict:
    return {k: torch.as_tensor(rep[k], dtype=torch.float32, device=device) for k in ("means", "scales", "quats", "opacities", "colors")}


@torch.no_grad()
def render_representation(rep: dict, rep_t: dict, cam: Camera) -> dict:
    resident, weights = select_resident(rep, cam.eye)
    idx = torch.as_tensor(resident, dtype=torch.long, device=rep_t["means"].device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=rep_t["means"].device)
    out = render_gaussians(
        rep_t["means"][idx], rep_t["scales"][idx], rep_t["quats"][idx], rep_t["opacities"][idx], rep_t["colors"][idx], cam, weights=w
    )
    out["loaded"] = int(resident.size)
    return out


# --------------------------------------------------------------------------
# Image metrics (uint8 protocol)
# --------------------------------------------------------------------------


def to_uint8(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().float().cpu().numpy()
    img = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def psnr_u8(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) / 255.0 - gt.astype(np.float64) / 255.0) ** 2))
    if mse <= 1e-10:
        return 100.0
    return float(-10.0 * math.log10(mse))


def _gauss_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])


def ssim_u8(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM (11x11 Gaussian window, sigma 1.5, valid region, mean over channels)."""
    x = torch.as_tensor(pred.astype(np.float64) / 255.0).permute(2, 0, 1)[None]
    y = torch.as_tensor(gt.astype(np.float64) / 255.0).permute(2, 0, 1)[None]
    w = _gauss_window().expand(3, 1, 11, 11).contiguous()
    mu_x = F.conv2d(x, w, groups=3)
    mu_y = F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mu_x * mu_x
    syy = F.conv2d(y * y, w, groups=3) - mu_y * mu_y
    sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2))
    return float(s.mean())


def transition_psnr(preds: list, gts: list, worst_k: int = 5) -> float:
    """Temporal PSNR of the ``worst_k`` worst frame-to-frame changes along a path.

    For consecutive frames the rendered change ``R[t+1]-R[t]`` is compared to the
    target change ``G[t+1]-G[t]``; popping or flicker that the true scene does
    not have shows up as a large residual.
    """
    mses = []
    for t in range(len(preds) - 1):
        dp = preds[t + 1].astype(np.float64) / 255.0 - preds[t].astype(np.float64) / 255.0
        dg = gts[t + 1].astype(np.float64) / 255.0 - gts[t].astype(np.float64) / 255.0
        mses.append(float(np.mean((dp - dg) ** 2)))
    if not mses:
        return 100.0
    top = sorted(mses)[-worst_k:]
    m = float(np.mean(top))
    if m <= 1e-10:
        return 100.0
    return float(-10.0 * math.log10(m))


# --------------------------------------------------------------------------
# Public data loading helpers
# --------------------------------------------------------------------------


def load_base_gaussians(data_dir: Path) -> dict:
    with np.load(Path(data_dir) / "base_gaussians.npz", allow_pickle=False) as z:
        return {k: np.array(z[k], dtype=np.float32) for k in ("means", "scales", "quats", "opacities", "colors")}


def load_camera_sets(data_dir: Path) -> dict:
    meta = json.loads((Path(data_dir) / "cameras.json").read_text())
    intr = meta["intrinsics"]
    out = {}
    for key, cams in meta["sets"].items():
        out[key] = [Camera.from_dict(c, intr["f"], intr["width"], intr["height"]) for c in cams]
    return out


def load_images(data_dir: Path, name: str) -> np.ndarray:
    with np.load(Path(data_dir) / "images.npz", allow_pickle=False) as z:
        return np.array(z[name])
