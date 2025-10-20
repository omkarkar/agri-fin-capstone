# predict_cli.py — quick command-line prediction for next-year yield
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Processed_AgriYield_Features.csv"
MODELS = ROOT / "models"

CAT_FEATURES = ["Area", "Item"]
NUM_FEATURES = [
    "Year", "Production_tonnes", "Area_ha", "Yield_t_per_ha",
    "Yield_Growth_Rate", "Production_Growth_Rate", "Area_Growth_Rate",
    "Avg_Yield_Last3yrs", "Avg_Production_Last3yrs", "Country_Avg_Yield", "Crop_Avg_Yield"
]


def main(area, item, year):
    df = pd.read_csv(DATA).sort_values(["Area", "Item", "Year"])
    row = df[(df["Area"] == area) & (df["Item"] == item)
             & (df["Year"] == int(year))]
    if row.empty:
        print("No row found for that Area/Item/Year.")
        sys.exit(1)
    mdf = pd.read_json(MODELS / "metrics_step6.json")
    best = mdf.query("split=='valid'").sort_values("RMSE").iloc[0]["model"]
    pipe = joblib.load(MODELS / f"{best}_yield_next.joblib")

    feature_cols = list(dict.fromkeys(CAT_FEATURES + NUM_FEATURES))
    X = row[feature_cols]
    pred = float(pipe.predict(X)[0])
    print(
        f"Predicted next-year yield for {area} / {item} from {year} → {int(year)+1}: {pred:.2f} t/ha")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python predict_cli.py "Area" "Item" YEAR')
        sys.exit(1)
    _, area, item, year = sys.argv
    main(area, item, year)
