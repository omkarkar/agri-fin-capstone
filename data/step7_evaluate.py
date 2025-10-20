# step7_evaluate.py — Analyze predictions and metrics
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_metrics():
    mpath = MODELS / "metrics_step6.json"
    with open(mpath, "r") as f:
        metrics = json.load(f)
    df = pd.DataFrame(metrics)
    return df


def summarize_metrics(df):
    pivot_rmse = df.pivot(index="split", columns="model", values="RMSE")
    pivot_mae = df.pivot(index="split", columns="model", values="MAE")
    pivot_r2 = df.pivot(index="split", columns="model", values="R2")

    print("\n=== RMSE ===\n", pivot_rmse.round(3))
    print("\n=== MAE ===\n", pivot_mae.round(3))
    print("\n=== R² ===\n", pivot_r2.round(3))

    pivot_rmse.to_csv(ARTIFACTS / "rmse_summary.csv")
    pivot_mae.to_csv(ARTIFACTS / "mae_summary.csv")
    pivot_r2.to_csv(ARTIFACTS / "r2_summary.csv")
    print("[saved] Summary CSVs → artifacts folder")


def analyze_predictions(split="test"):
    preds_path = MODELS / f"preds_{split}.csv"
    df = pd.read_csv(preds_path)
    df["AbsError"] = (df["Target_Yield_next"] - df["Pred_Yield_next"]).abs()
    df["SqError"] = (df["Target_Yield_next"] - df["Pred_Yield_next"]) ** 2

    # Top 20 misses
    worst = df.sort_values("AbsError", ascending=False).head(20)
    worst.to_csv(ARTIFACTS / f"top20_misses_{split}.csv", index=False)
    print(f"[saved] Top 20 misses → top20_misses_{split}.csv")

    # Error by (Area, Item)
    grp = df.groupby(["Area", "Item"]).agg(
        MAE=("AbsError", "mean"),
        RMSE=("SqError", lambda s: np.sqrt(s.mean())),
        Count=("AbsError", "size")
    ).reset_index().sort_values("RMSE", ascending=False)
    grp.to_csv(ARTIFACTS / f"group_error_{split}.csv", index=False)
    print(f"[saved] Error by crop/country → group_error_{split}.csv")

    # Residual histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["Target_Yield_next"] - df["Pred_Yield_next"], bins=40)
    plt.title(f"Residuals ({split.upper()} set)")
    plt.xlabel("Error (Target - Predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    out = ARTIFACTS / f"residual_hist_{split}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[saved] Residual plot → {out}")


def main():
    print("[1] Loading metrics...")
    dfm = load_metrics()
    summarize_metrics(dfm)

    print("\n[2] Analyzing test predictions...")
    analyze_predictions("test")

    print("\n✅ Step 7 complete. Check 'artifacts' folder for results.")


if __name__ == "__main__":
    main()
