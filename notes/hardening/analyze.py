"""Rank analysis of the hardening audit (pure Python; reads notes/hardening/cases/*/summary.json).

The leaderboard normalises each metric by its spread over later attempts, which
are unknown here.  Two stated panels are used instead:
  H  population std over the honest runs that pass every gate
  A  population std over every case (failed runs enter with their worst values)
Rank = sum_e weight_e * sum_metric |coef| * dir * (value - reference) / std, over
the scored entries of scoring/reference.json (the fixed baseline entry is
identical in every case and contributes zero).  Reference = reference_seed0.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = json.loads((ROOT / "scoring" / "reference.json").read_text())
CASES = {p.parent.name: json.loads(p.read_text()) for p in sorted((ROOT / "notes" / "hardening" / "cases").glob("*/summary.json"))}
HONEST = None  # set after gating


def scored():
    for exp, spec in REF["experiments"].items():
        for metric, m in spec["metrics"].items():
            if m["coefficient"]:
                yield exp, metric, spec["weight"], abs(m["coefficient"]), 1.0 if m["higher_is_better"] else -1.0


N0 = 247095
TOL = {"trajectory_views": 1.0, "flythrough": 1.5, "off_path_views": 3.5}
TRANS_TOL = 0.5
MEM_FRAC = 0.65
VIS_FRAC = {"trajectory_views": 0.65, "flythrough": 0.53, "off_path_views": 0.57}
HIB = {"psnr", "ssim", "worst_view_psnr", "transition_psnr"}


def gate(case: str):
    """Re-apply the final evaluator gates to the stored raw measurements (exact: gates are a
    deterministic function of the measured metrics and the evaluator's own baseline)."""
    st = CASES[case]["methods"].get("lodge", {})
    me = st.get("measured")
    base = {e: CASES[case]["results"][e]["full_representation"] for e in CASES[case]["results"]}
    if me is None or st["reason"].startswith(("invalid", "evaluation")):
        return False, st.get("reason", "missing"), None
    if max(me[e]["peak_loaded_gaussians"] for e in me) > MEM_FRAC * N0:
        return False, "memory budget", me
    for e in me:
        if me[e]["visible_gaussians"] > VIS_FRAC[e] * base[e]["visible_gaussians"]:
            return False, f"work budget ({e})", me
    for e in me:
        if me[e]["psnr"] < base[e]["psnr"] - TOL[e]:
            return False, f"quality ({e})", me
    if me["flythrough"]["transition_psnr"] < base["flythrough"]["transition_psnr"] - TRANS_TOL:
        return False, "transitions", me
    return True, "ok", me


GATED = {c: gate(c) for c in CASES}


def value(case: str, exp: str, metric: str):
    ok, _, me = GATED[case]
    if not ok:
        return 0.0 if metric in HIB else 4.0 * N0
    return me[exp][metric]


def normalisers(panel: list[str]) -> dict:
    out = {}
    for exp, metric, *_ in scored():
        vals = [value(c, exp, metric) for c in panel if value(c, exp, metric) is not None]
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out[exp, metric] = sd or 1.0
    return out


def rank(case: str, norm: dict, ref: str = "reference_seed0") -> float:
    total = 0.0
    for exp, metric, w, c, d in scored():
        v, r = value(case, exp, metric), value(ref, exp, metric)
        if v is None or r is None:
            continue
        total += w * c * d * (v - r) / norm[exp, metric]
    return total


def main() -> None:
    honest = [c for c in CASES if not c.startswith("sc_") and GATED[c][0]]
    panels = {"H": honest, "A": list(CASES)}
    out_gates = {c: GATED[c][1] for c in CASES}
    out = {"honest_panel": honest, "gates": out_gates, "panels": {}}
    for label, panel in panels.items():
        norm = normalisers(panel)
        ranks = {c: rank(c, norm) for c in CASES}
        seeds = [ranks[f"reference_seed{i}"] for i in range(3) if f"reference_seed{i}" in ranks]
        out["panels"][label] = {
            "normalisers": {f"{e}/{m}": v for (e, m), v in norm.items()},
            "rank_vs_reference_seed0": ranks,
            "reference_seed_rank_range": max(seeds) - min(seeds),
        }
        print(f"\n== panel {label} (reference seed range {max(seeds) - min(seeds):.3f})")
        for c, r in sorted(ranks.items(), key=lambda kv: -kv[1]):
            print(f"  {c:28s} {r:+9.3f}  {GATED[c][1]}")
    (ROOT / "notes" / "hardening" / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
