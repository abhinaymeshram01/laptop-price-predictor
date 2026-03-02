import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 16px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #4338ca; }
    .price-box {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: white;
        margin-top: 20px;
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Exact Feature Columns ──────────────────────────────────────────────────────
FEATURE_COLS = [
    'Processor_Speed', 'RAM_Size', 'Storage_Capacity',
    'Screen_Size', 'Weight',
    'Brand_Asus', 'Brand_Dell', 'Brand_HP', 'Brand_Lenovo'
]

# ── Always generate sample data for evaluation ────────────────────────────────
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 600
    brands = np.random.choice(["Asus", "Dell", "HP", "Lenovo", "Other"], n)

    df = pd.DataFrame({
        "Processor_Speed":   np.round(np.random.uniform(1.5, 4.5, n), 1),
        "RAM_Size":          np.random.choice([4, 8, 16, 32, 64], n),
        "Storage_Capacity":  np.random.choice([128, 256, 512, 1024, 2048], n),
        "Screen_Size":       np.round(np.random.choice([13.3, 14.0, 15.6, 16.0, 17.3], n), 1),
        "Weight":            np.round(np.random.uniform(1.0, 3.5, n), 2),
        "Brand_Asus":        (brands == "Asus").astype(int),
        "Brand_Dell":        (brands == "Dell").astype(int),
        "Brand_HP":          (brands == "HP").astype(int),
        "Brand_Lenovo":      (brands == "Lenovo").astype(int),
    })

    brand_price = {"Asus": 500, "Dell": 600, "HP": 550, "Lenovo": 580, "Other": 470}
    df["Price"] = (
        df["Processor_Speed"] * 120
        + df["RAM_Size"] * 10
        + df["Storage_Capacity"] * 0.3
        + df["Screen_Size"] * 15
        - df["Weight"] * 30
        + pd.Series(brands).map(brand_price).values
        + np.random.normal(0, 80, n)
    ).clip(250, 5000).round(2)

    return df

# ── Load or train model — always compute MAE & R² ────────────────────────────
@st.cache_resource
def load_model():
    df = get_sample_data()
    X = df[FEATURE_COLS]
    y = df["Price"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        # Evaluate the loaded model on sample test data
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        source = "pkl"
    else:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        source = "trained"

    return model, mae, r2, source

model, mae, r2, source = load_model()

if source == "pkl":
    st.sidebar.success("✅ Loaded model from **model.pkl**")
else:
    st.sidebar.info("ℹ️ Using sample-trained model. Drop your `model.pkl` in the same folder to use your real model.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💻 Laptop Price Predictor")
st.markdown("Predict laptop prices using **Linear Regression** based on your exact feature set.")
st.markdown("---")

# ── Layout ─────────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 1], gap="large")

with col_form:
    st.subheader("🔧 Enter Laptop Specifications")

    brand = st.selectbox("🏷️ Brand", ["Asus", "Dell", "HP", "Lenovo", "Other"])

    processor_speed = st.slider(
        "⚙️ Processor Speed (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1
    )

    ram_size = st.selectbox("🧠 RAM Size (GB)", [4, 8, 16, 32, 64])

    storage_capacity = st.selectbox("💾 Storage Capacity (GB)", [128, 256, 512, 1024, 2048])

    screen_size = st.select_slider(
        "🖥️ Screen Size (inches)", options=[13.3, 14.0, 15.6, 16.0, 17.3], value=15.6
    )

    weight = st.slider("⚖️ Weight (kg)", min_value=0.8, max_value=4.0, value=2.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Price", use_container_width=True)

with col_result:
    st.subheader("📊 Model Performance")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""<div class='metric-box'>
            <h3 style='color:#4f46e5;margin:0'>{r2*100:.1f}%</h3>
            <p style='margin:0;color:gray'>R² Score</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-box'>
            <h3 style='color:#4f46e5;margin:0'>${mae:.0f}</h3>
            <p style='margin:0;color:gray'>Mean Abs. Error</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        input_data = {
            "Processor_Speed":  processor_speed,
            "RAM_Size":         ram_size,
            "Storage_Capacity": storage_capacity,
            "Screen_Size":      screen_size,
            "Weight":           weight,
            "Brand_Asus":       1 if brand == "Asus"   else 0,
            "Brand_Dell":       1 if brand == "Dell"   else 0,
            "Brand_HP":         1 if brand == "HP"     else 0,
            "Brand_Lenovo":     1 if brand == "Lenovo" else 0,
        }

        input_df = pd.DataFrame([input_data])[FEATURE_COLS]
        predicted_price = model.predict(input_df)[0]
        predicted_price = max(predicted_price, 100)

        st.markdown(f"""
        <div class='price-box'>
            <p style='font-size:18px;margin-bottom:5px;opacity:0.85'>Estimated Laptop Price</p>
            <h1 style='font-size:52px;margin:0'>${predicted_price:,.0f}</h1>
            <p style='font-size:14px;opacity:0.7;margin-top:8px'>Linear Regression Prediction</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📋 Your Configuration:**")
        config = {
            "Brand":            brand,
            "Processor Speed":  f"{processor_speed} GHz",
            "RAM":              f"{ram_size} GB",
            "Storage":          f"{storage_capacity} GB",
            "Screen Size":      f"{screen_size} inches",
            "Weight":           f"{weight} kg",
        }
        for k, v in config.items():
            st.write(f"- **{k}:** {v}")
    else:
        st.info("👈 Fill in the specs on the left and click **Predict Price**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><p style='color:gray;font-size:13px'>Built with ❤️ using Streamlit & Scikit-learn | Linear Regression Model</p></center>",
    unsafe_allow_html=True
)