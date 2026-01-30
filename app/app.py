import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# ===============================
# PREMIUM UI CSS (SINGLE BLOCK)
# ===============================
st.markdown("""
<style>

/* ----- App Background ----- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f4f6f8;
}

/* Remove header background */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Main container spacing */
.block-container {
    padding-top: 2rem;
}

/* Headings */
h1 {
    color: #3a0ca3;
}
h2, h3 {
    color: #6a040f;
}

/* White Card Style */
.card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Slider accent color */
div[data-baseweb="slider"] > div {
    color: #9d0208;
}

/* Result cards */
.result-card {
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 10px;
}

.excellent {
    background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
    color: #1b4332;
}

.good {
    background: linear-gradient(135deg, #e0fbfc, #cce3de);
    color: #005f73;
}

.average {
    background: linear-gradient(135deg, #fff3cd, #ffe8a1);
    color: #7c2d12;
}

/* Banner control */
.banner img {
    max-height: 200px;
    object-fit: cover;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_artifacts():
    with open("wine_quality_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ===============================
# TITLE
# ===============================
st.markdown('<div class="banner">', unsafe_allow_html=True)
st.image("banner.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


st.title("🍷 Wine Quality Prediction")
st.markdown(
    "Predict the **quality category of wine** (*Average, Good, Excellent*) "
    "using physicochemical properties and **wine color**."
)

st.divider()

# ===============================
# INPUT SECTION
# ===============================
st.subheader("🧪 Physicochemical Properties",text_alignment='center',divider='violet')
st.subheader("🔧 Enter Wine Characteristics")

# Wine color input
# Wine color as radio button
color = st.radio(
    "Wine Color",
    options=["Red", "White"],
    horizontal=True
)
color_encoded = 0 if color == "Red" else 1

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 0.0, 20.0, 7.4, 0.1)
    volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.7, 0.01)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, 0.01)
    residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 1.9, 0.1)
    chlorides = st.slider("Chlorides", 0.0, 1.0, 0.076, 0.001)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 11.0, 1.0)

with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0, 1.0)
    density = st.slider("Density", 0.9900, 1.0100, 0.9978, 0.0001)
    pH = st.slider("pH", 2.0, 4.5, 3.51, 0.01)
    sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56, 0.01)
    alcohol = st.slider("Alcohol (%)", 5.0, 20.0, 9.4, 0.1)


st.divider()

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Wine Quality", use_container_width=True):

    # Create input DataFrame (for display)
    color_map = {0: "Red", 1: "White"}
    input_df = pd.DataFrame({
        "Fixed Acidity": [fixed_acidity],
        "Volatile Acidity": [volatile_acidity],
        "Citric Acid": [citric_acid],
        "Residual Sugar": [residual_sugar],
        "Chlorides": [chlorides],
        "Free Sulfur Dioxide": [free_sulfur_dioxide],
        "Total Sulfur Dioxide": [total_sulfur_dioxide],
        "Density": [density],
        "pH": [pH],
        "Sulphates": [sulphates],
        "Alcohol": [alcohol],
        "Color": [color_map[color_encoded]]
    })

    # Arrange features in correct order for model
    input_array = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        color_encoded
    ]])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🍾 Prediction Result")

    if prediction == "Excellent":
        st.markdown(
        '<div class="result-card excellent">🌟 Excellent Quality Wine</div>',
        unsafe_allow_html=True
        )
    elif prediction == "Good":
        st.markdown(
        '<div class="result-card good">🍷 Good Quality Wine</div>',
        unsafe_allow_html=True
        )
    else:
        st.markdown(
        '<div class="result-card average">⚠️ Average Quality Wine</div>',
        unsafe_allow_html=True
        )


    # Show user input data
    st.subheader("🧾 Entered Wine Details")
    st.dataframe(input_df, use_container_width=True,height=100)

# ===============================
# FOOTER
# ===============================
st.divider()
st.markdown("📌 **Note:** Predictions are based on historical wine quality data.")
st.caption("📊 Machine Learning Powered | Streamlit App | Created by Pavan Ahire")
