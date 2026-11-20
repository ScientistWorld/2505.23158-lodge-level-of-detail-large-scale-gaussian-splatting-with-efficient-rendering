"""Hardening audit harness (run inside a compute job).

Stages (each resumable; outputs under tmp/audit and notes/hardening):
  world      build the hidden world, public data and hidden targets (if missing)
  tune       short sweep of the paper's free hyperparameters (gamma, s) on the
             public validation views; pick the most aggressive setting whose
             validation PSNR stays within 0.3 dB of the full representation
  reference  the paper-faithful reference (1000 fine-tuning steps/round) with
             seeds 0,1,2; seed 0 also exports the nearest-chunk / dynamic-LOD ablations
  variants   better / weaker variants and simple hand-written rules
  cases      stage checkpoints/ for every case and run the unmodified hidden test
             (bash scripts/evaluate_test.sh) and the visible evaluator on it
  final      restore the reference (seed 0) and write scoring/scores_{train,test}.json

Usage: python notes/hardening/audit.py [--smoke] [--stages world,tune,...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.gsrender import load_base_gaussians, load_camera_sets, save_representation  # noqa: E402

NOTES = ROOT / "notes" / "hardening"


def log(msg: str) -> None:
    print(f"[audit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], env_extra: dict | None = None, check: bool = True) -> int:
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    log("$ " + " ".join(cmd))
    t = time.time()
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    log(f"rc={rc} ({time.time() - t:.0f}s)")
    if check and rc != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return rc


class Ctx:
    def __init__(self, smoke: bool):
        self.smoke = smoke
        self.work = ROOT / "tmp" / ("audit_smoke" if smoke else "audit")
        self.data = (self.work / "data") if smoke else ROOT / "data" / "lodge_city"
        self.hidden = (self.work / "hidden_set.npz") if smoke else ROOT / "eval" / "test" / "hidden_set.npz"
        self.reps = self.work / "reps"
        self.out = (self.work / "notes") if smoke else NOTES
        self.steps = 20 if smoke else 1000
        self.tune_steps = 10 if smoke else 400
        self.cache = self.work / "threshold_cache.json"
        for p in (self.work, self.reps, self.out):
            p.mkdir(parents=True, exist_ok=True)

    def method(self, out: Path, *extra: str, ablation: Path | None = None) -> None:
        if (out / "representation.npz").exists():
            log(f"skip existing {out}")
            return
        cmd = [sys.executable, "-m", "method.lodge", "--data-dir", str(self.data), "--out-dir", str(out),
               "--threshold-cache", str(self.cache)] + list(extra)
        if ablation is not None:
            cmd += ["--ablation-dir", str(ablation)]
        run(cmd)


# --------------------------------------------------------------------------


def check_rasterizer() -> None:
    """Compare the vectorised rasteriser with a straightforward per-tile loop."""
    import torch

    from data import gsrender as R

    def loop_raster(mean2d, conic, opac, colors, depth, radius, width, height, background):
        ntx, nty = width // 16, height // 16
        bg = torch.as_tensor(background, dtype=torch.float32, device=mean2d.device)
        img = bg.expand(height, width, 3).clone()
        order = torch.sort(depth, stable=True).indices
        for ty in range(nty):
            for tx in range(ntx):
                x0, y0 = tx * 16, ty * 16
                u, v, r = mean2d[order, 0], mean2d[order, 1], radius[order]
                m = (torch.clamp(torch.floor((u - r) / 16), 0, ntx - 1) <= tx) & (torch.clamp(torch.floor((u + r) / 16), 0, ntx - 1) >= tx) & \
                    (torch.clamp(torch.floor((v - r) / 16), 0, nty - 1) <= ty) & (torch.clamp(torch.floor((v + r) / 16), 0, nty - 1) >= ty)
                g = order[m]
                yy, xx = torch.meshgrid(torch.arange(16, device=mean2d.device) + y0 + 0.5, torch.arange(16, device=mean2d.device) + x0 + 0.5, indexing="ij")
                T = torch.ones(16, 16, device=mean2d.device)
                acc = torch.zeros(16, 16, 3, device=mean2d.device)
                for j in g.tolist():
                    dx, dy = xx - mean2d[j, 0], yy - mean2d[j, 1]
                    p = torch.clamp(-0.5 * (conic[j, 0] * dx * dx + conic[j, 2] * dy * dy) - conic[j, 1] * dx * dy, max=0)
                    a = torch.clamp(opac[j] * torch.exp(p), max=R.ALPHA_MAX)
                    a = torch.where(a >= R.ALPHA_MIN, a, torch.zeros_like(a))
                    a = torch.where(T > R.T_MIN, a, torch.zeros_like(a)) if False else a
                    w = a * T * (T > R.T_MIN).float()
                    acc += w[..., None] * colors[j]
                    T = T * (1 - a)
                img[y0:y0 + 16, x0:x0 + 16] = acc + T[..., None] * bg
        return img

    g = torch.Generator().manual_seed(0)
    n = 400
    means = torch.rand(n, 3, generator=g) * torch.tensor([8.0, 6.0, 3.0]) + torch.tensor([3.0, -3.0, 0.0])
    scales = torch.rand(n, 3, generator=g) * 0.3 + 0.01
    quats = torch.randn(n, 4, generator=g)
    opac = torch.rand(n, generator=g) * 0.9 + 0.05
    cols = torch.rand(n, 3, generator=g)
    err = 0.0
    dev = R.default_device()
    for yaw, pitch, mp in ((0.1, 0.05, 1 << 17), (-0.75, 0.3, 1 << 17), (-0.75, 0.3, 300), (0.4, -0.2, 50)):
        R.MAX_PAIRS_PER_GROUP = mp
        cam = R.Camera(np.array([0.0, 0.0, 1.5]), yaw, pitch, 40.0, 64, 48)
        proj = R.project_gaussians(means.to(dev), scales.to(dev), quats.to(dev), cam)
        m = proj["valid"]
        args_ = (proj["mean2d"][m], proj["conic"][m], (opac.to(dev) * proj["comp"])[m], cols.to(dev)[m], proj["depth"][m], proj["radius"][m])
        vec, _, _ = R._rasterize(*args_, 64, 48, R.BACKGROUND, False)
        ref = loop_raster(*[a.cpu() for a in args_], 64, 48, R.BACKGROUND)
        err = max(err, float((vec.cpu() - ref).abs().max()))
    R.MAX_PAIRS_PER_GROUP = 1 << 17
    log(f"rasteriser self-check: max abs difference vs per-tile loop = {err:.2e}")
    if err > 1e-4:
        raise RuntimeError("vectorised rasteriser disagrees with the reference loop")


def stage_world(ctx: Ctx) -> None:
    if (ctx.data / "base_gaussians.npz").exists() and ctx.hidden.exists():
        log("world exists")
        return
    cmd = [sys.executable, "-m", "eval.test.world", "--build", "--data-dir", str(ctx.data), "--hidden", str(ctx.hidden)]
    if ctx.smoke:
        cmd.append("--smoke")
    run(cmd)


def eval_train_on(ctx: Ctx, rep_dir: Path, tag: str) -> dict:
    stage = ctx.work / "stage_train" / tag
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "lodge").mkdir(parents=True)
    shutil.copy(rep_dir / "representation.npz", stage / "lodge" / "representation.npz")
    out = ctx.work / "train_scores" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "eval.train.evaluate", "--data-dir", str(ctx.data), "--checkpoints-dir", str(stage), "--scores", str(out)])
    return json.loads(out.read_text())


def stage_tune(ctx: Ctx) -> dict:
    path = ctx.out / "tuning.json"
    if path.exists():
        return json.loads(path.read_text())
    grid = [(g, s) for s in (0.2, 1.0) for g in (0.01, 0.03, 0.1)]
    if ctx.smoke:
        grid = [(0.03, 0.2), (0.03, 1.0)]
    rows = []
    for g, s in grid:
        tag = f"tune_g{g}_s{s}"
        out = ctx.reps / tag
        ctx.method(out, "--gamma", str(g), "--smooth-s", str(s), "--finetune-steps", str(ctx.tune_steps), "--seed", "0")
        sc = eval_train_on(ctx, out, tag)
        st = sc["_meta"]["status"]["lodge"]
        m = st.get("measured", {})
        base = {e: sc["experiments"][e]["results"]["full_representation"] for e in sc["experiments"]}
        rows.append({
            "gamma": g, "smooth_s": s, "passed": st["passed"], "reason": st["reason"],
            "val_psnr": {e: m.get(e, {}).get("psnr") for e in base},
            "val_full_psnr": {e: base[e]["psnr"] for e in base},
            "val_loaded": m.get("trajectory_views", {}).get("loaded_gaussians"),
            "val_visible": m.get("trajectory_views", {}).get("visible_gaussians"),
            "method_info": json.loads((out / "method_info.json").read_text()),
        })
        log(f"tune {tag}: {rows[-1]['val_psnr']} loaded={rows[-1]['val_loaded']} visible={rows[-1]['val_visible']}")

    valid = [r for r in rows if r["val_psnr"]["trajectory_views"] is not None]
    top = max(r["val_psnr"]["trajectory_views"] for r in valid)
    cands = [r for r in valid if r["val_psnr"]["trajectory_views"] >= top - 0.5]
    best = min(cands, key=lambda r: r["val_loaded"] + r["val_visible"])
    rule = "lowest val loaded+visible among settings within 0.5 dB of the best val trajectory PSNR of the grid"
    res = {"grid": rows, "chosen": {"gamma": best["gamma"], "smooth_s": best["smooth_s"]}, "rule": rule, "tune_steps": ctx.tune_steps}
    path.write_text(json.dumps(res, indent=1))
    log(f"chosen {res['chosen']}")
    return res


def chosen_args(ctx: Ctx) -> list[str]:
    ch = json.loads((ctx.out / "tuning.json").read_text())["chosen"]
    return ["--gamma", str(ch["gamma"]), "--smooth-s", str(ch["smooth_s"])]


def stage_reference(ctx: Ctx) -> None:
    for seed in (0, 1, 2):
        abl = ctx.reps / "ablation_seed0" if seed == 0 else None
        ctx.method(ctx.reps / f"reference_seed{seed}", *chosen_args(ctx), "--finetune-steps", str(ctx.steps), "--seed", str(seed), ablation=abl)


def stage_variants(ctx: Ctx) -> None:
    base = chosen_args(ctx)
    steps = ["--finetune-steps", str(ctx.steps)]
    seeds = (0,) if ctx.smoke else (0, 1, 2)
    for seed in seeds:
        ctx.method(ctx.reps / f"better_levels3_seed{seed}", *base, *steps, "--levels", "3", "--seed", str(seed))
        ctx.method(ctx.reps / f"better_pseudo_seed{seed}", *base, *steps, "--pseudo-frac", "0.5", "--seed", str(seed))
    ctx.method(ctx.reps / "weaker_no_finetune", *base, "--finetune-steps", "0", "--seed", "0")
    info = json.loads((ctx.reps / "reference_seed0" / "method_info.json").read_text())
    t = info["thresholds"]
    manual = [str(round(t[0] * 1.5, 1)), str(round(t[-1] * 2.5, 1))]
    ctx.method(ctx.reps / "weaker_manual_thresholds", *base, *steps, "--thresholds", *manual, "--seed", "0")
    build_rules(ctx, info)


def build_rules(ctx: Ctx, info: dict) -> None:
    """Short hand-written LOD rules (no optimisation) used as shortcut probes."""
    base = load_base_gaussians(ctx.data)
    train = load_camera_sets(ctx.data)["train"]
    pos = np.stack([c.eye for c in train])
    n0 = base["means"].shape[0]
    k = int(info["n_chunks"])
    order = np.argsort(pos[:, 0])
    centers = np.stack([pos[order[int(i)]] for i in np.linspace(0, len(pos) - 1, k)])
    rng = np.random.default_rng(0)
    keep_far = rng.random(n0) < 0.25
    coarse = {kk: v[keep_far].copy() for kk, v in base.items()}
    coarse["scales"][:, :2] *= 2.0
    pool = {kk: np.concatenate([base[kk], coarse[kk]]) for kk in base}
    far_idx = np.arange(n0, n0 + int(keep_far.sum()))
    lists = []
    for c in centers:
        dn = np.linalg.norm(base["means"] - c[None], axis=1)
        dc = np.linalg.norm(coarse["means"] - c[None], axis=1)
        lists.append(np.concatenate([np.where(dn < 12.0)[0], far_idx[dc >= 12.0]]))
    out = ctx.reps / "rule_distance_decimation" / "representation.npz"
    if not out.exists():
        save_representation(out, pool, centers, lists, blend_two_nearest=True)
    out = ctx.reps / "rule_distance_cull" / "representation.npz"
    if not out.exists():
        lists = [np.where(np.linalg.norm(base["means"] - c[None], axis=1) < 25.0)[0] for c in centers]
        save_representation(out, base, centers, lists, blend_two_nearest=True)
    out = ctx.reps / "full_as_method" / "representation.npz"
    if not out.exists():
        save_representation(out, base, pos.mean(0)[None], [np.arange(n0)], blend_two_nearest=False)


def write_rep(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in z.files}


def stage_cases(ctx: Ctx) -> None:
    ck = ROOT / "checkpoints" if not ctx.smoke else ctx.work / "checkpoints"
    ref = ctx.reps / "reference_seed0" / "representation.npz"
    ref_arr = load_npz(ref)
    base = load_base_gaussians(ctx.data)
    n0 = base["means"].shape[0]
    pos = np.stack([c.eye for c in load_camera_sets(ctx.data)["train"]])

    def single(rep_dir: str):
        return {"lodge": ctx.reps / rep_dir / "representation.npz"}

    cases: dict[str, dict] = {}
    for seed in (0, 1, 2):
        if (ctx.reps / f"reference_seed{seed}").exists():
            cases[f"reference_seed{seed}"] = single(f"reference_seed{seed}")
        for v in ("better_levels3", "better_pseudo"):
            if (ctx.reps / f"{v}_seed{seed}").exists():
                cases[f"{v}_seed{seed}"] = single(f"{v}_seed{seed}")
    for v in ("weaker_no_finetune", "weaker_manual_thresholds", "rule_distance_decimation", "rule_distance_cull", "full_as_method"):
        if (ctx.reps / v).exists():
            cases[v] = single(v)
    cases["ablation_nearest_chunk"] = {"lodge": ctx.reps / "ablation_seed0" / "nearest_chunk" / "representation.npz"}
    cases["ablation_lod_dynamic"] = {"lodge": ctx.reps / "ablation_seed0" / "lod_dynamic" / "representation.npz"}

    # Shortcut probes.
    sc = ctx.work / "shortcuts"
    one = {"means": np.zeros((1, 3), np.float32), "scales": np.full((1, 3), 0.01, np.float32), "quats": np.array([[1, 0, 0, 0]], np.float32),
           "opacities": np.zeros(1, np.float32), "colors": np.zeros((1, 3), np.float32), "chunk_centers": np.zeros((1, 3), np.float32),
           "chunk_offsets": np.array([0, 1]), "chunk_indices": np.array([0]), "blend_two_nearest": np.array(0)}
    write_rep(sc / "empty" / "representation.npz", one)
    nan = dict(ref_arr)
    nan["means"] = nan["means"].copy()
    nan["means"][::50] = np.nan
    write_rep(sc / "nan" / "representation.npz", nan)
    bad = dict(ref_arr)
    bad["chunk_indices"] = bad["chunk_indices"].copy()
    bad["chunk_indices"][0] = ref_arr["means"].shape[0] + 5
    write_rep(sc / "bad_indices" / "representation.npz", bad)
    dup = {kk: np.concatenate([base[kk], base[kk]]) for kk in base}
    save_representation(sc / "over_memory" / "representation.npz", dup, pos.mean(0)[None], [np.arange(2 * n0)], blend_two_nearest=False)
    huge = {kk: np.concatenate([base[kk]] * 5) for kk in base}
    save_representation(sc / "huge_pool" / "representation.npz", huge, pos.mean(0)[None], [np.arange(n0)], blend_two_nearest=False)

    cases["sc_missing_lodge"] = {}
    cases["sc_empty_representation"] = {"lodge": sc / "empty" / "representation.npz"}
    cases["sc_nan_means"] = {"lodge": sc / "nan" / "representation.npz"}
    cases["sc_bad_indices"] = {"lodge": sc / "bad_indices" / "representation.npz"}
    cases["sc_over_memory"] = {"lodge": sc / "over_memory" / "representation.npz"}
    cases["sc_huge_pool"] = {"lodge": sc / "huge_pool" / "representation.npz"}
    cases["sc_forged_files"] = {"lodge": ref, "_forge": True}
    cases["sc_baseline_dir"] = {"lodge": ref, "full_representation": sc / "empty" / "representation.npz"}
    cases["sc_extra_method"] = {"lodge": ref, "black_extra": sc / "empty" / "representation.npz"}

    results_dir = ctx.out / "cases"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in cases.items():
        dest = results_dir / name
        if (dest / "scores_test.json").exists():
            summary[name] = json.loads((dest / "summary.json").read_text())
            continue
        if any(not Path(p).exists() for k, p in spec.items() if not k.startswith("_")):
            log(f"case {name}: missing inputs, skipped")
            continue
        if ck.exists():
            shutil.rmtree(ck)
        ck.mkdir(parents=True)
        for method_name, p in spec.items():
            if method_name.startswith("_"):
                continue
            (ck / method_name).mkdir(parents=True, exist_ok=True)
            shutil.copy(p, ck / method_name / "representation.npz")
        if spec.get("_forge"):
            (ck / "lodge" / "stats.json").write_text(json.dumps({"per_view": [{"loaded": 0, "visible": 0}] * 100}))
            (ck / "lodge" / "method_info.json").write_text(json.dumps({"psnr": 99, "loaded": 0, "passed": True}))
        dest.mkdir(parents=True, exist_ok=True)
        if ctx.smoke:
            test_scores = dest / "scores_test.json"
            run([sys.executable, "-m", "eval.test.evaluate", "--checkpoints-dir", str(ck), "--hidden", str(ctx.hidden), "--scores", str(test_scores)])
            run([sys.executable, "-m", "eval.train.evaluate", "--data-dir", str(ctx.data), "--checkpoints-dir", str(ck), "--scores", str(dest / "scores_train.json")])
        else:
            if spec.get("_forge"):
                (ROOT / "scoring" / "scores_test.json").write_text(json.dumps({"experiments": {}, "forged": True}))
                (ROOT / "scoring" / "scores_train.json").write_text(json.dumps({"experiments": {}}))
            run(["bash", "scripts/evaluate_test.sh"])
            shutil.copy(ROOT / "scoring" / "scores_test.json", dest / "scores_test.json")
            run(["bash", "scripts/evaluate_train.sh"])
            shutil.copy(ROOT / "scoring" / "scores_train.json", dest / "scores_train.json")
        test = json.loads((dest / "scores_test.json").read_text())
        summary[name] = {
            "methods": {m: v for m, v in test["_meta"]["status"].items()},
            "results": {e: test["experiments"][e]["results"] for e in test["experiments"]},
        }
        info_path = Path(spec["lodge"]).parent / "method_info.json" if "lodge" in spec else None
        if info_path is not None and info_path.exists():
            summary[name]["method_info"] = json.loads(info_path.read_text())
        (dest / "summary.json").write_text(json.dumps(summary[name], indent=1))
        log(f"case {name}: " + json.dumps({m: s["reason"] for m, s in summary[name]["methods"].items()}))
    (ctx.out / "cases_summary.json").write_text(json.dumps(summary, indent=1))


def stage_final(ctx: Ctx) -> None:
    if ctx.smoke:
        return
    ck = ROOT / "checkpoints"
    if ck.exists():
        shutil.rmtree(ck)
    (ck / "lodge").mkdir(parents=True)
    shutil.copy(ctx.reps / "reference_seed0" / "representation.npz", ck / "lodge" / "representation.npz")
    shutil.copy(ctx.reps / "reference_seed0" / "method_info.json", ck / "lodge" / "method_info.json")
    abl = ck / "_ablation"
    for v in ("nearest_chunk", "lod_dynamic"):
        (abl / v).mkdir(parents=True, exist_ok=True)
        shutil.copy(ctx.reps / "ablation_seed0" / v / "representation.npz", abl / v / "representation.npz")
    run(["bash", "scripts/evaluate_train.sh"])
    run(["bash", "scripts/evaluate_test.sh"])
    run(["bash", "scripts/evaluate.sh"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stages", default="world,tune,reference,variants,cases,final")
    args = parser.parse_args()
    ctx = Ctx(args.smoke)
    import torch

    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    stages = args.stages.split(",")
    check_rasterizer()
    if "world" in stages:
        stage_world(ctx)
    if "tune" in stages:
        stage_tune(ctx)
    if "reference" in stages:
        stage_reference(ctx)
    if "variants" in stages:
        stage_variants(ctx)
    if "cases" in stages:
        stage_cases(ctx)
    if "final" in stages:
        stage_final(ctx)
    log("done")


if __name__ == "__main__":
    main()
