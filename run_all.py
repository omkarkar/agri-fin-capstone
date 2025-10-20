# run_all.py — end-to-end: features -> train -> evaluate -> importance -> report
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

steps = [
    ["python", str(DATA / "step5_fe.py")],
    ["python", str(DATA / "step6_train_model.py")],
    ["python", str(DATA / "step7_evaluate.py")],
    ["python", str(DATA / "step8_feature_importance.py")],
    ["python", str(DATA / "step9_build_report.py")],
]

for cmd in steps:
    print("\n" + "-"*80)
    print("Running:", " ".join(cmd))
    print("-"*80)
    code = subprocess.call(cmd)
    if code != 0:
        sys.exit(code)

print("\n[done] Pipeline finished. See /models, /artifacts, /report.")
