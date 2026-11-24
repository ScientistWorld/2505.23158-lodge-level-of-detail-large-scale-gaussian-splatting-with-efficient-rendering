"""Compare a CPU run of the hidden test with the stored (GPU) scores_test.json."""
import json
import sys
from pathlib import Path

a = json.loads(Path("scoring/scores_test.json").read_text())
b = json.loads(Path(sys.argv[1]).read_text())
out = {"status_cpu": {m: s["reason"] for m, s in b["_meta"]["status"].items()}, "max_abs_diff": {}}
for e, v in a["experiments"].items():
    for m, r in v["results"].items():
        for k, x in r.items():
            y = b["experiments"][e]["results"][m][k]
            key = f"{e}/{m}/{k}"
            out["max_abs_diff"][key] = abs(x - y)
print(json.dumps(out, indent=1))
Path("notes/hardening/cpu_check.json").write_text(json.dumps(out, indent=1))
