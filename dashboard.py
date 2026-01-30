import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Wine Quality EDA Dashboard",
    page_icon="🍷",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.autolayout"] = True

# ================= LOAD DATA =================
df = pd.read_csv("winequality_clean.csv")

# ================= FEATURE GROUPS =================
num_features = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH",
    "sulphates", "alcohol"
]

cat_features = ["quality"]

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.kpi-box {
    border: 5px solid #e0e0e0;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    background-color: #ffffff;
}
.kpi-title {
    font-size: 14px;
    color: #6c757d;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>🍷 Wine Quality EDA Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= KPI SECTION =================
total_wines = df.shape[0]
avg_quality = round(df["quality"].mean(), 2)
avg_alcohol = round(df["alcohol"].mean(), 2)
avg_ph = round(df["pH"].mean(), 2)
avg_sulphates = round(df["sulphates"].mean(), 2)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Total Wines</div>
    <div class="kpi-value">{total_wines}</div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg Quality</div>
    <div class="kpi-value">{avg_quality}</div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg Alcohol (%)</div>
    <div class="kpi-value">{avg_alcohol}</div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg pH</div>
    <div class="kpi-value">{avg_ph}</div>
</div>
""", unsafe_allow_html=True)

k5.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg Sulphates</div>
    <div class="kpi-value">{avg_sulphates}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🛠️ Analysis Controls")

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

# ================= UNIVARIATE ANALYSIS =================
if analysis_type == "Univariate Analysis":

    st.subheader("📊 Univariate Analysis")

    feature = st.sidebar.selectbox(
        "Select Feature",
        num_features + cat_features
    )

    col1, col2 = st.columns(2, gap="large")

    if feature in num_features:
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[feature], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[feature], ax=ax)
            ax.set_title(f"Boxplot of {feature}")
            st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[feature], ax=ax)
        ax.set_title("Quality Distribution")
        st.pyplot(fig)

# ================= BIVARIATE ANALYSIS =================
else:

    st.subheader("📈 Bivariate Analysis")

    bi_type = st.sidebar.selectbox(
        "Select Relationship",
        ["Num vs Num", "Num vs Cat", "Cat vs Cat"]
    )

    if bi_type == "Num vs Num":
        x = st.sidebar.selectbox("X Axis", num_features)
        y = st.sidebar.selectbox("Y Axis", num_features, index=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="quality",
            palette="viridis",
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    elif bi_type == "Num vs Cat":
        num = st.sidebar.selectbox("Numerical Feature", num_features)
        cat = st.sidebar.selectbox("Categorical Feature", cat_features)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x=cat,
            y=num,
            ax=ax
        )
        ax.set_title(f"{num} by {cat}")
        st.pyplot(fig)

    else:
        x = st.sidebar.selectbox("Category", cat_features)
        y = st.sidebar.selectbox("Hue", cat_features, index=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x=x, hue=y, ax=ax)
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

# ================= FOOTER =================
st.markdown("---")
st.info(
    "This dashboard performs Exploratory Data Analysis (EDA) on the cleaned Wine Quality dataset "
    "using interactive univariate and bivariate visualizations."
)
