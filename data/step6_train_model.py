# step6_train_model.py — VERSION: sklearn-compatible (no 'squared='; robust OHE)
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "Processed_AgriYield_Features.csv"
MODELS_DIR = HERE.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def time_splits(df):
    # Train: up to 2020 (target = 2014–2021)
    # Valid: 2021 (target = 2022)
    # Test:  2022 (target = 2023)
    train = df[df["Year"] <= 2020].copy()
    valid = df[df["Year"] == 2021].copy()
    test = df[df["Year"] == 2022].copy()
    return train, valid, test


def evaluate(name, pipe, X, y):
    """Compatibility with old sklearn (no 'squared=' arg)."""
    pred = pipe.predict(X)
    mae = mean_absolute_error(y, pred)
    try:
        rmse = mean_squared_error(y, pred, squared=False)  # new sklearn
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y, pred))        # old sklearn
    r2 = r2_score(y, pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "pred": pred}


def main():
    # 1) Load
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run Step 5 first.")
    df = pd.read_csv(DATA)
    print("[ok] Loaded:", df.shape)

    # 2) Target = next year's yield per (Area, Item)
    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)
    df["Target_Yield_next"] = df.groupby(["Area", "Item"])[
        "Yield_t_per_ha"].shift(-1)
    df = df.dropna(subset=["Target_Yield_next"]).reset_index(drop=True)
    print("[ok] After target creation:", df.shape)

    # 3) Time splits
    train, valid, test = time_splits(df)
    print(
        f"[split] Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")

    # 4) Features
    cat_features = ["Area", "Item"]
    num_features = [
        "Year",
        "Production_tonnes",
        "Area_ha",
        "Yield_t_per_ha",
        "Yield_Growth_Rate",
        "Production_Growth_Rate",
        "Area_Growth_Rate",
        "Avg_Yield_Last3yrs",
        "Avg_Production_Last3yrs",
        "Country_Avg_Yield",
        "Crop_Avg_Yield",
    ]

    X_train, y_train = train[cat_features +
                             num_features], train["Target_Yield_next"]
    X_valid, y_valid = valid[cat_features +
                             num_features], valid["Target_Yield_next"]
    X_test,  y_test = test[cat_features +
                           num_features], test["Target_Yield_next"]

    # 5) Robust OneHotEncoder (old/new sklearn)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # 6) Pipelines
    lr_pipe = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    rf_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(
        n_estimators=350, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    ))])

    # 7) Train
    print("[train] Linear Regression...")
    lr_pipe.fit(X_train, y_train)

    print("[train] Random Forest...")
    rf_pipe.fit(X_train, y_train)

    # 8) Evaluate
    results = []
    for name, pipe in [("LinearRegression", lr_pipe), ("RandomForest", rf_pipe)]:
        for split, X, y in [("train", X_train, y_train),
                            ("valid", X_valid, y_valid),
                            ("test",  X_test,  y_test)]:
            m = evaluate(f"{name}_{split}", pipe, X, y)
            print(
                f"[{name:>14} | {split}]  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}")
            results.append({"model": name, "split": split,
                           "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"]})

    # 9) Select best on VALID (lowest RMSE)
    df_res = pd.DataFrame(results)
    best_row = df_res[df_res["split"] == "valid"].sort_values("RMSE").iloc[0]
    best_name = best_row["model"]
    best_pipe = lr_pipe if best_name == "LinearRegression" else rf_pipe
    print(f"[select] Best model on VALID: {best_name}")

    # 10) Save artifacts
    model_path = MODELS_DIR / f"{best_name}_yield_next.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"[saved] Model -> {model_path}")

    metrics_path = MODELS_DIR / "metrics_step6.json"
    df_res.to_json(metrics_path, orient="records", indent=2)
    print(f"[saved] Metrics -> {metrics_path}")

    # 11) Save predictions using best model
    def save_preds(split_name, X, y, frame):
        pred = best_pipe.predict(X)
        out = frame[["Area", "Item", "Year"]].copy()
        out["Target_Yield_next"] = y.values
        out["Pred_Yield_next"] = pred
        out["Error"] = out["Target_Yield_next"] - out["Pred_Yield_next"]
        out_path = MODELS_DIR / f"preds_{split_name}.csv"
        out.to_csv(out_path, index=False)
        print(f"[saved] {split_name} predictions -> {out_path}")

    save_preds("train", X_train, y_train, train)
    save_preds("valid", X_valid, y_valid, valid)
    save_preds("test",  X_test,  y_test,  test)

    print("\n[done] Step 6 complete.")


if __name__ == "__main__":
    main()
