# step8_feature_importance.py — permutation importance on saved best model (fixed)
# Works with older/newer scikit-learn. Produces CSV + bar chart of Top-20 features.

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "Processed_AgriYield_Features.csv"
MODELS = HERE.parent / "models"
ARTIFACTS = HERE.parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# -------- helpers --------
def time_splits(df: pd.DataFrame):
    """Same temporal split used in Step 6."""
    train = df[df["Year"] <= 2020].copy()
    valid = df[df["Year"] == 2021].copy()
    test = df[df["Year"] == 2022].copy()
    return train, valid, test


def load_best_model_name() -> str:
    """Read metrics_step6.json and return the best model on VALID by RMSE."""
    metrics_path = MODELS / "metrics_step6.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    mdf = pd.DataFrame(metrics)
    best_row = mdf[mdf["split"] == "valid"].sort_values("RMSE").iloc[0]
    return best_row["model"]  # "LinearRegression" or "RandomForest"


def safe_permutation_importance(estimator, X, y, n_repeats=10, random_state=42):
    """
    Prefer sklearn.inspection.permutation_importance with scoring='neg_mean_squared_error'.
    IMPORTANT: permutation_importance.importances_mean is ΔMSE (positive when worse).
    We convert to ΔRMSE = sqrt(ΔMSE). If sklearn isn't available, use a manual fallback.
    """
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            estimator,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        # Convert ΔMSE -> ΔRMSE
        imp_mean = np.sqrt(np.maximum(0.0, r.importances_mean))
        imp_std = np.sqrt(np.maximum(0.0, r.importances_std))
        return imp_mean, imp_std
    except Exception:
        # Manual fallback: shuffle each column and measure ΔRMSE directly
        from sklearn.metrics import mean_squared_error

        rng = np.random.RandomState(random_state)
        base_pred = estimator.predict(X)
        base_rmse = np.sqrt(mean_squared_error(y, base_pred))
        deltas = []
        for col in X.columns:
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred = estimator.predict(Xp)
            rmse = np.sqrt(mean_squared_error(y, pred))
            deltas.append(max(0.0, rmse - base_rmse))
        return np.array(deltas), np.zeros(len(deltas))


def main():
    # 1) Load processed data
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run Step 5 first.")
    df = pd.read_csv(DATA)

    # 2) Create next-year target exactly like Step 6
    df = df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)
    df["Target_Yield_next"] = df.groupby(["Area", "Item"])[
        "Yield_t_per_ha"].shift(-1)
    df = df.dropna(subset=["Target_Yield_next"]).reset_index(drop=True)

    # 3) Use TEST split for importance (reflects final performance)
    _, _, test = time_splits(df)
    target = "Target_Yield_next"

    # Same features as Step 6
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
    features = cat_features + num_features

    X_test = test[features].copy()
    y_test = test[target].copy()

    # 4) Load best saved model
    best = load_best_model_name()
    model_path = MODELS / f"{best}_yield_next.joblib"
    pipe = joblib.load(model_path)
    print(f"[ok] Loaded best model: {best} -> {model_path}")

    # 5) Permutation importance on TEST
    print("[info] Computing permutation importance on TEST...")
    imp_mean, imp_std = safe_permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=42
    )

    # 6) Results table
    imp_df = (
        pd.DataFrame(
            {"feature": features, "importance_rmse": imp_mean, "std": imp_std}
        )
        .sort_values("importance_rmse", ascending=False)
        .reset_index(drop=True)
    )

    # 7) Save CSV
    out_csv = ARTIFACTS / "feature_importance_permutation.csv"
    imp_df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # 8) Plot Top-20 bar chart
    top = imp_df.head(20).iloc[::-1]  # reverse for barh
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["importance_rmse"])
    plt.title("Permutation Importance (ΔRMSE when shuffled) — TEST")
    plt.xlabel("Importance (ΔRMSE)")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_png = ARTIFACTS / "feature_importance_permutation_top20.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[saved] {out_png}")

    # 9) Console preview
    print("\nTop 10 features:")
    print(imp_df.head(10).to_string(index=False))
    print("\n[done] Step 8 complete.")


if __name__ == "__main__":
    main()
