# step9_build_report.py — Build a compact project report from artifacts
from pathlib import Path
import pandas as pd
import datetime

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve(
).parent.name == "data") else Path(__file__).resolve().parent
PROJ = ROOT / "agri-fin-capstone" if ROOT.name != "agri-fin-capstone" else ROOT
DATA = PROJ / "data"
MODELS = PROJ / "models"
ART = PROJ / "artifacts"
OUT = PROJ / "report"
OUT.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst_name: str):
    if src.exists():
        dst = OUT / dst_name
        dst.write_bytes(src.read_bytes())
        return dst
    return None


def main():
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Pull core artifacts
    rmse = ART / "rmse_summary.csv"
    mae = ART / "mae_summary.csv"
    r2 = ART / "r2_summary.csv"
    top_miss = ART / "top20_misses_test.csv"
    grp_err = ART / "group_error_test.csv"
    resid = ART / "residual_hist_test.png"
    fi_csv = ART / "feature_importance_permutation.csv"
    fi_png = ART / "feature_importance_permutation_top20.png"

    rmse_out = copy_if_exists(rmse, "rmse_summary.csv")
    mae_out = copy_if_exists(mae,  "mae_summary.csv")
    r2_out = copy_if_exists(r2,   "r2_summary.csv")
    top_out = copy_if_exists(top_miss, "top20_misses_test.csv")
    grp_out = copy_if_exists(grp_err,  "group_error_test.csv")
    resid_out = copy_if_exists(resid,    "residual_hist_test.png")
    fi_csv_out = copy_if_exists(fi_csv, "feature_importance_permutation.csv")
    fi_png_out = copy_if_exists(
        fi_png, "feature_importance_permutation_top20.png")

    # Read metrics for a one-line summary
    rmse_df = pd.read_csv(rmse_out) if rmse_out else None
    mae_df = pd.read_csv(mae_out) if mae_out else None
    r2_df = pd.read_csv(r2_out) if r2_out else None

    # Build Markdown report
    md = []
    md.append(f"# Agri-Fin Capstone — Model Report\n")
    md.append(f"_Generated: {ts}_\n")
    md.append("## Overview\n")
    md.append(
        "- Task: Predict **next-year yield (t/ha)** per (Country, Crop, Year).")
    md.append(
        "- Best model (by VALID RMSE): **Linear Regression** (saved pipeline).")
    if rmse_df is not None:
        md.append("\n## Metrics (RMSE / MAE / R²)\n")
        md.append("**RMSE**\n")
        md.append(rmse_df.to_markdown(index=False))
    if mae_df is not None:
        md.append("\n**MAE**\n")
        md.append(mae_df.to_markdown(index=False))
    if r2_df is not None:
        md.append("\n**R²**\n")
        md.append(r2_df.to_markdown(index=False))

    if fi_png_out:
        md.append("\n## Feature Importance (Permutation, ΔRMSE)\n")
        md.append("Top-20 features by how much RMSE worsens when shuffled.\n")
        md.append(f"![Feature Importance]({fi_png_out.name})\n")
    if resid_out:
        md.append("\n## Residuals (Test Set)\n")
        md.append("Should be roughly centered around 0.\n")
        md.append(f"![Residuals]({resid_out.name})\n")

    if top_out:
        md.append("\n## Top 20 Largest Errors (Test)\n")
        md.append("See CSV for details: `top20_misses_test.csv`.")
    if grp_out:
        md.append("\n## Error by Country & Crop (Test)\n")
        md.append("See CSV for details: `group_error_test.csv`.")

    md.append("\n## Data & Features\n")
    md.append("- Source: `Cleaned_AgriYield_Asia_2013_2023.csv`")
    md.append(
        "- Engineered features: growth rates, rolling 3-year means, country/crop baselines.")
    md.append("\n## Repro Steps\n")
    md.append("1) Step 5: Feature engineering → `Processed_AgriYield_Features.csv`")
    md.append("2) Step 6: Train models → saves pipeline + predictions + metrics")
    md.append("3) Step 7: Evaluate → saves artifacts")
    md.append("4) Step 8: Importance → saves permutation importance")
    md.append("5) Step 9: Build this report")

    report_md = OUT / "MODEL_REPORT.md"
    report_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {report_md}")

    print("[done] Report assembled. Open the 'report' folder.")


if __name__ == "__main__":
    main()
