# app.py — Streamlit demo for Agri-Fin Yield Predictor (Next Year)
# Run: streamlit run app.py

from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle  # fallback loader

# ---------------- paths & constants ----------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Processed_AgriYield_Features.csv"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

CAT_FEATURES = ["Area", "Item"]
NUM_FEATURES = [
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

# ---------------- robust model loader ----------------


def load_pipeline(path: Path):
    """
    Prefer joblib; if unavailable or fails, fall back to pickle.
    Keeps the app running on environments where joblib isn't installed.
    """
    try:
        import joblib  # lazy import so it's optional at import time
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

# ---------------- cached helpers ----------------


@st.cache_data
def load_data():
    if not DATA.exists():
        st.error(
            "Processed_AgriYield_Features.csv not found. Please complete Step 5.")
        st.stop()
    df = pd.read_csv(DATA)
    return df.sort_values(["Area", "Item", "Year"]).reset_index(drop=True)


@st.cache_data
def load_best_model_name():
    mpath = MODELS / "metrics_step6.json"
    if not mpath.exists():
        st.error("metrics_step6.json not found. Please run Step 6 (training).")
        st.stop()
    mdf = pd.read_json(mpath)
    row = mdf.query("split=='valid'").sort_values("RMSE").iloc[0]
    return str(row["model"])  # "LinearRegression" or "RandomForest"


@st.cache_data
def load_best_split_metrics(best_name: str):
    """Return test metrics for the best model, if available."""
    mpath = MODELS / "metrics_step6.json"
    if not mpath.exists():
        return None
    mdf = pd.read_json(mpath)
    # pick TEST split for the chosen best model
    try:
        row = mdf[(mdf["split"] == "test") & (
            mdf["model"] == best_name)].iloc[0]
        return {"MAE": float(row["MAE"]), "RMSE": float(row["RMSE"]), "R2": float(row["R2"])}
    except Exception:
        return None


@st.cache_resource
def load_model(best_name: str):
    mpath = MODELS / f"{best_name}_yield_next.joblib"
    if not mpath.exists():
        st.error(f"Saved model not found: {mpath}")
        st.stop()
    return load_pipeline(mpath)


# ---------------- UI ----------------
st.set_page_config(page_title="Agri-Fin Yield Demo", layout="wide")
st.title("🌾 Agri-Fin Yield Predictor — Next Year (t/ha)")

df = load_data()
best_name = load_best_model_name()
pipe = load_model(best_name)

# Sidebar header with model + metrics
st.sidebar.success(f"Best model: **{best_name}**")
m = load_best_split_metrics(best_name)
if m:
    st.sidebar.caption("**Test metrics**")
    col_mae, col_rmse, col_r2 = st.sidebar.columns(3)
    col_mae.metric("MAE", f"{m['MAE']:.3f}")
    col_rmse.metric("RMSE", f"{m['RMSE']:.3f}")
    col_r2.metric("R²", f"{m['R2']:.3f}")

# ---- Quick presets (optional) ----
with st.sidebar.expander("Quick examples"):
    preset = st.selectbox(
        "Jump to a known (Area, Item)",
        [
            ("India", "Wheat"),
            ("India", "Rice, paddy"),
            ("China, mainland", "Rice, paddy"),
            ("Indonesia", "Oil palm fruit"),
            ("Pakistan", "Cotton lint"),
            ("Viet Nam", "Coffee, green"),
            ("Philippines", "Bananas"),
        ],
        index=0
    )

# ---------------- dependent dropdowns ----------------
# 1) Area first
areas = sorted(df["Area"].unique().tolist())
default_area = preset[0] if preset[0] in areas else areas[0]
area = st.sidebar.selectbox("Area (Country)", areas,
                            index=areas.index(default_area))

# 2) Items available for this Area only
items_available = sorted(df.loc[df["Area"] == area, "Item"].unique().tolist())
if not items_available:
    st.warning("No crops found for this Area. Try another Area.")
    st.stop()

default_item = preset[1] if preset[1] in items_available else items_available[0]
item = st.sidebar.selectbox(
    "Crop (Item)", items_available, index=items_available.index(default_item))

# 3) Subset rows for this (Area, Item)
sub = df[(df["Area"] == area) & (df["Item"] == item)].copy()
if sub.empty:
    st.warning("No rows for this (Area, Item). Try another crop.")
    st.stop()

# 4) Years available for this pair
min_year = int(sub["Year"].min())
max_year = int(sub["Year"].max())
year = st.sidebar.slider(
    "Base Year (we predict Year+1)",
    min_value=min_year,
    max_value=max_year,
    value=max_year,
    step=1,
)

row = sub[sub["Year"] == year]
if row.empty:
    st.warning(f"No data for {area} / {item} in {year}. Choose another year.")
    st.stop()

# ---------------- prediction ----------------
# Build feature row with UNIQUE columns (avoid duplicate names)
# preserves order, removes dups
feature_cols = list(dict.fromkeys(CAT_FEATURES + NUM_FEATURES))
X = row[feature_cols].copy()

try:
    pred = float(pipe.predict(X)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.metric(
    f"Predicted Yield for {area} — {item} in {year+1}", f"{pred:.2f} t/ha")

# Download the current prediction as CSV
out_row = row[["Area", "Item", "Year"] + NUM_FEATURES].copy()
out_row["Predicted_Yield_next"] = pred
csv_bytes = out_row.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download this prediction (CSV)",
    data=csv_bytes,
    file_name=f"prediction_{area}_{item}_{year+1}.csv",
    mime="text/csv",
)

# context metrics
country_avg = float(row["Country_Avg_Yield"].iloc[0])
crop_avg = float(row["Crop_Avg_Yield"].iloc[0])
cur_yield = float(row["Yield_t_per_ha"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("Country Avg Yield", f"{country_avg:.2f} t/ha")
c2.metric("Crop Avg Yield", f"{crop_avg:.2f} t/ha")
c3.metric("Current Year Yield", f"{cur_yield:.2f} t/ha")

# ---------------- chart: history + predicted point ----------------
st.subheader("History & Next-Year Prediction")
hist = sub[["Year", "Yield_t_per_ha"]].copy()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(hist["Year"], hist["Yield_t_per_ha"], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Yield (t/ha)")
ax.set_title(f"{area} — {item} Yield History")
# predicted point for year+1
ax.scatter([year + 1], [pred], s=60)
ax.annotate(f"Pred {year+1}: {pred:.2f}", (year + 1, pred),
            xytext=(5, 5), textcoords="offset points")
st.pyplot(fig)

# ---------------- inputs table ----------------
with st.expander("See input features used for prediction"):
    cols_for_table = ["Area", "Item", "Year"] + NUM_FEATURES
    cols_for_table = list(dict.fromkeys(cols_for_table))  # remove duplicates
    st.dataframe(row[cols_for_table].reset_index(
        drop=True), use_container_width=True)

# ---------------- actual vs predicted chart (if preds CSVs exist) ----------------
pred_files = ["preds_train.csv", "preds_valid.csv", "preds_test.csv"]
frames = []
for fn in pred_files:
    p = MODELS / fn
    if p.exists():
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass

if frames:
    comp = pd.concat(frames, ignore_index=True)
    comp = comp[(comp["Area"] == area) & (comp["Item"] == item)].copy()
    if not comp.empty:
        comp["Target_Year"] = comp["Year"] + \
            1  # we predicted next year's yield
        comp = comp.sort_values("Target_Year")

        st.subheader("Actual vs Predicted (Next-Year Yield)")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(comp["Target_Year"], comp["Target_Yield_next"],
                 marker="o", label="Actual")
        ax2.plot(comp["Target_Year"], comp["Pred_Yield_next"],
                 marker="o", linestyle="--", label="Predicted")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Yield (t/ha)")
        ax2.legend()
        ax2.set_title(f"{area} — {item}")
        st.pyplot(fig2)
    else:
        st.info(
            "No saved predictions for this Area/Crop yet. (Run Step 6 locally to generate preds_*.csv)")
else:
    st.info("Prediction CSVs not found. (Run Step 6 locally to generate preds_*.csv)")

# ---------------- optional: feature importance image ----------------
fi_png = ARTIFACTS / "feature_importance_permutation_top20.png"
if fi_png.exists():
    st.subheader("Top Features (Permutation Importance)")
    st.image(str(fi_png))
else:
    st.info("Run Step 8 to generate feature importance (optional).")
