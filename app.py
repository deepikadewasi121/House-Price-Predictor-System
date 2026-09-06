import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2540 100%);
        border: 1px solid #2c4a6e;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: #f8fafc; }
    .hero p { margin: .35rem 0 0 0; color: #b7c5d9; font-size: 1rem; }

    .price-card {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #14532d 0%, #0f3d22 100%);
        border: 1px solid #1f6b3a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .price-card .label { color: #9fd6b3; font-size: .95rem; letter-spacing: .04em; text-transform: uppercase;}
    .price-card .value { color: #eafff0; font-size: 2.8rem; font-weight: 700; margin: .2rem 0;}
    .price-card .range { color: #a9d9bb; font-size: .9rem; }

    .metric-box {
        padding: .9rem 1.1rem;
        border-radius: 14px;
        background: #161b26;
        border: 1px solid #262e3f;
    }
    .metric-box .m-label { color: #8b95a7; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;}
    .metric-box .m-value { color: #f1f5f9; font-size: 1.35rem; font-weight: 600; }

    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #262e3f;
    }
    div[data-testid="stMetricValue"] { color: #f1f5f9; }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model + metadata (cached so it only happens once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
ranges = meta["ranges"]
ocean_categories = meta["ocean_categories"]
metrics = meta["metrics"]


def default_val(col):
    return float(ranges[col]["median"])


def predict_house_price(model, longitude, latitude, housing_median_age, total_rooms,
                         total_bedrooms, population, households, median_income,
                         ocean_proximity):
    new_row = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])
    return float(model.predict(new_row)[0])


# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🏡 California House Price Predictor</h1>
    <p>Powered by a tuned HistGradientBoostingRegressor trained on the California Housing Prices dataset.</p>
</div>
""", unsafe_allow_html=True)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown(f"""<div class="metric-box"><div class="m-label">Test RMSE</div>
    <div class="m-value">${metrics['rmse']:,.0f}</div></div>""", unsafe_allow_html=True)
with top2:
    st.markdown(f"""<div class="metric-box"><div class="m-label">Test MAE</div>
    <div class="m-value">${metrics['mae']:,.0f}</div></div>""", unsafe_allow_html=True)
with top3:
    st.markdown(f"""<div class="metric-box"><div class="m-label">R² Score</div>
    <div class="m-value">{metrics['r2']:.3f}</div></div>""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# Sidebar — Inputs
# ----------------------------------------------------------------------------
st.sidebar.header("🏠 House / District Details")

with st.sidebar.expander("📍 Location", expanded=True):
    latitude = st.slider(
        "Latitude", float(meta["lat_range"][0]), float(meta["lat_range"][1]),
        value=default_val("latitude"), step=0.01,
        help="California latitude, e.g. 37.88 is near Oakland/Berkeley."
    )
    longitude = st.slider(
        "Longitude", float(meta["lon_range"][0]), float(meta["lon_range"][1]),
        value=default_val("longitude"), step=0.01,
        help="California longitude, e.g. -122.23 is near the Bay Area."
    )
    ocean_proximity = st.selectbox(
        "Ocean Proximity", ocean_categories,
        index=ocean_categories.index("NEAR BAY") if "NEAR BAY" in ocean_categories else 0
    )

with st.sidebar.expander("🏗️ Housing Characteristics", expanded=True):
    housing_median_age = st.slider(
        "Median House Age (years)",
        int(ranges["housing_median_age"]["min"]), int(ranges["housing_median_age"]["max"]),
        value=int(default_val("housing_median_age"))
    )
    total_rooms = st.number_input(
        "Total Rooms (in block)", min_value=1.0,
        max_value=float(ranges["total_rooms"]["max"]),
        value=default_val("total_rooms"), step=10.0
    )
    total_bedrooms = st.number_input(
        "Total Bedrooms (in block)", min_value=0.0,
        max_value=float(ranges["total_bedrooms"]["max"]),
        value=default_val("total_bedrooms"), step=5.0
    )
    households = st.number_input(
        "Households (in block)", min_value=1.0,
        max_value=float(ranges["households"]["max"]),
        value=default_val("households"), step=5.0
    )
    population = st.number_input(
        "Population (in block)", min_value=1.0,
        max_value=float(ranges["population"]["max"]),
        value=default_val("population"), step=10.0
    )

with st.sidebar.expander("💰 Income", expanded=True):
    median_income = st.slider(
        "Median Income (in $10,000s)",
        float(ranges["median_income"]["min"]), float(ranges["median_income"]["max"]),
        value=default_val("median_income"), step=0.1,
        help="e.g. 8.32 means roughly $83,200 median household income for the block."
    )

st.sidebar.write("---")
predict_clicked = st.sidebar.button("🔮 Predict House Price", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Main layout: map + prediction
# ----------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    st.subheader("📍 Location on Map")
    map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_df, zoom=8, size=200, color="#22c55e")

    with st.expander("ℹ️ About this app"):
        st.write(
            "This app wraps the exact preprocessing + model pipeline built in the "
            "accompanying notebook: median imputation & scaling for numeric features, "
            "most-frequent imputation & one-hot encoding for `ocean_proximity`, feeding "
            "a tuned `HistGradientBoostingRegressor`. Adjust the inputs on the left and "
            "click **Predict House Price** to estimate the median house value for that "
            "California block."
        )

with right:
    st.subheader("💵 Prediction")
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    if predict_clicked:
        pred = predict_house_price(
            model, longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income, ocean_proximity
        )
        st.session_state.prediction = pred

    if st.session_state.prediction is not None:
        pred = st.session_state.prediction
        low = pred - metrics["mae"]
        high = pred + metrics["mae"]
        st.markdown(f"""
        <div class="price-card">
            <div class="label">Estimated Median House Value</div>
            <div class="value">${pred:,.0f}</div>
            <div class="range">± ${metrics['mae']:,.0f} typical error &nbsp;•&nbsp; range ${max(low,0):,.0f} – ${high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("**Inputs used for this prediction:**")
        summary = pd.DataFrame({
            "Feature": ["Latitude", "Longitude", "Ocean Proximity", "Median Age",
                        "Total Rooms", "Total Bedrooms", "Households", "Population",
                        "Median Income"],
            "Value": [latitude, longitude, ocean_proximity, housing_median_age,
                      total_rooms, total_bedrooms, households, population,
                      f"${median_income * 10000:,.0f}"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)
    else:
        st.info("Fill in the details on the left sidebar and click **🔮 Predict House Price** to see the estimate.")

st.write("---")
st.caption("Model: tuned HistGradientBoostingRegressor · Dataset: California Housing Prices (Kaggle) · Built with Streamlit")
