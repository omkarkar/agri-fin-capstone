# step5_fe.py — Feature Engineering Script
# Author: ChatGPT Guided Setup

from pathlib import Path
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "Cleaned_AgriYield_Asia_2013_2023.csv"
OUT = HERE / "Processed_AgriYield_Features.csv"


def main():
    print("[info] Working dir:", Path.cwd())
    print("[info] Script dir: ", HERE)
    print("[info] Looking for CSV:", CSV)

    # Check if the dataset exists
    if not CSV.exists():
        print("[error] CSV not found at:", CSV)
        print("[hint] Make sure the file is inside the same 'data' folder.")
        sys.exit(1)

    # 1️⃣ Load
    df = pd.read_csv(CSV)
    print("[ok] Data loaded:", df.shape)
    print(df.head(3))

    # 2️⃣ Sort
    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)

    # 3️⃣ Growth Rates
    df["Yield_Growth_Rate"] = (
        df.groupby(["Area", "Item"])["Yield_t_per_ha"].pct_change() * 100
    )
    df["Production_Growth_Rate"] = (
        df.groupby(["Area", "Item"])["Production_tonnes"].pct_change() * 100
    )
    df["Area_Growth_Rate"] = (
        df.groupby(["Area", "Item"])["Area_ha"].pct_change() * 100
    )

    # 4️⃣ Rolling 3-Year Averages
    df["Avg_Yield_Last3yrs"] = (
        df.groupby(["Area", "Item"])["Yield_t_per_ha"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["Avg_Production_Last3yrs"] = (
        df.groupby(["Area", "Item"])["Production_tonnes"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # 5️⃣ Country & Crop Averages
    df["Country_Avg_Yield"] = df.groupby(
        "Area")["Yield_t_per_ha"].transform("mean")
    df["Crop_Avg_Yield"] = df.groupby(
        "Item")["Yield_t_per_ha"].transform("mean")

    # 6️⃣ Fill Missing Growths (first year)
    for col in ["Yield_Growth_Rate", "Production_Growth_Rate", "Area_Growth_Rate"]:
        df[col] = df[col].fillna(0)

    # 7️⃣ Save
    df.to_csv(OUT, index=False)
    print(f"[saved] {OUT}")
    print("\n[done] Feature engineering complete ✅")
    print("Final columns:")
    print(list(df.columns))
    print("\nSample rows:")
    print(df.head(10))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[fatal] Error occurred:", e)
        traceback.print_exc()
        sys.exit(1)
