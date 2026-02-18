import streamlit as st
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Wine Quality Dashboard & Prediction",
    page_icon="🍷",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("winequality_clean.csv")

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    return df

df = load_data()

numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg,#4facfe,#00f2fe);
    padding:18px;
    border-radius:12px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 10px rgba(0,0,0,0.2);
}
.kpi-title{
    font-size:16px;
    font-weight:600;
}
.kpi-value{
    font-size:26px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Tabs container */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

/* Default tab style */
.stTabs [data-baseweb="tab"] {
    background-color: #f2f2f2;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 18px;
    font-weight: 500;
}

/* Selected tab style */
.stTabs [aria-selected="true"] {
    background-color: #ff4b4b !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px 8px 0px 0px;
}

</style>
""", unsafe_allow_html=True)

# ================= KPI FUNCTION =================
def show_kpis():
    col1, col2, col3, col4, col5 = st.columns(5)

    total_wines = df.shape[0]
    avg_quality = round(df["quality"].mean(), 2) if "quality" in df.columns else "-"
    avg_alcohol = round(df["alcohol"].mean(), 2) if "alcohol" in df.columns else "-"
    avg_ph = round(df["pH"].mean(), 2) if "pH" in df.columns else "-"
    avg_sulphates = round(df["sulphates"].mean(), 2) if "sulphates" in df.columns else "-"

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Wines</div>
            <div class="kpi-value">{total_wines}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Quality</div>
            <div class="kpi-value">{avg_quality}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Alcohol (%)</div>
            <div class="kpi-value">{avg_alcohol}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg pH</div>
            <div class="kpi-value">{avg_ph}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Sulphates</div>
            <div class="kpi-value">{avg_sulphates}</div>
        </div>
        """, unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Dashboard", "Model Prediction"])


# ================= DASHBOARD =================
if menu == "Dashboard":

    st.title("🍷 Wine Quality Dashboard",text_alignment ="center")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "About Dataset",
        "Univariate Analysis",
        "Bivariate Analysis"
    ])

    # ================= ABOUT DATASET =================
    with tab1:
        show_kpis()
        st.markdown("---")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("### Dataset Information")
        st.write(f"Shape of dataset: {df.shape}")
        st.write(f"Total rows: {df.shape[0]}")
        st.write(f"Total columns: {df.shape[1]}")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        st.write("### Project Objective")
        st.info("""
        The objective of this project is to analyze wine quality data using Exploratory Data Analysis (EDA). 
        This helps to understand feature distribution, detect patterns, and identify important factors 
        affecting wine quality before building Machine Learning models.
        """)

        st.write("### Why This Project?")
        st.success("""
        Dashboard demonstrates:
        - Data Cleaning
        - Exploratory Data Analysis
        - Data Visualization
        - Feature Understanding
        """)

    # ================= UNIVARIATE TAB =================
    with tab2:
        show_kpis()
        st.markdown("---")

        st.subheader("Univariate Analysis")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Count Plot"]
        )

        column = st.selectbox("Select Column", df.columns)

        # Histogram
        if chart_type == "Histogram":
            if column in numeric_cols:
                fig = px.histogram(df, x=column, title=f"Histogram of {column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Histogram supports only numeric columns.")

        # Box Plot
        elif chart_type == "Box Plot":
            if column in numeric_cols:
                fig = px.box(df, y=column, title=f"Box Plot of {column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Box Plot supports only numeric columns.")

        # Count Plot
        elif chart_type == "Count Plot":
            if column in categorical_cols:
                count_df = df[column].value_counts().reset_index()
                count_df.columns = [column, "Count"]

                fig = px.bar(
                    count_df,
                    x=column,
                    y="Count",
                    color=column,
                    title=f"Count Plot of {column}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Count Plot supports only categorical columns.")

    # ================= BIVARIATE TAB =================
    with tab3:
        show_kpis()
        st.markdown("---")
        st.subheader("Bivariate Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"]
        )

        graph_type = st.selectbox(
            "Select Graph Type",
            ["Scatter Plot", "Box Plot", "Bar Chart", "Heatmap"]
        )

        # Numeric vs Numeric
        if analysis_type == "Numeric vs Numeric":

            x = st.selectbox("Select X Feature", numeric_cols)
            y = st.selectbox("Select Y Feature", numeric_cols)

            color_col = "quality" if "quality" in df.columns else None

            if x == y:
                st.warning("Please select two different numeric features.")

                fig = px.histogram(
                    df,
                    x=x,
                    color=color_col,
                    title=f"Distribution of {x}"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                if graph_type == "Scatter Plot":
                    fig = px.scatter(df, x=x, y=y, color=color_col)
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Heatmap":
                    corr = df[[x, y]].corr()
                    fig = px.imshow(corr, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Box Plot":
                    fig = px.box(df, x=x, y=y, color=color_col)
                    st.plotly_chart(fig, use_container_width=True)

        # Numeric vs Categorical
        elif analysis_type == "Numeric vs Categorical":

            num = st.selectbox("Select Numeric Feature", numeric_cols)
            cat = st.selectbox("Select Categorical Feature", categorical_cols)

            if graph_type == "Box Plot":
                fig = px.box(df, x=cat, y=num, color=cat)
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "Bar Chart":
                temp = df.groupby(cat)[num].mean().reset_index()
                fig = px.bar(temp, x=cat, y=num, color=cat)
                st.plotly_chart(fig, use_container_width=True)

        # Categorical vs Categorical
        else:

            cat1 = st.selectbox("Select First Category", categorical_cols)
            cat2 = st.selectbox("Select Second Category", categorical_cols)

            cross_tab = pd.crosstab(df[cat1], df[cat2])

            fig = px.imshow(cross_tab, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # ================= FOOTER =================
    st.markdown("---")
    st.info(
        "This dashboard performs Exploratory Data Analysis (EDA) on the cleaned Wine Quality dataset "
        "using interactive visualizations."
    )
# ================= MODEL PREDICTION =================
elif menu == "Model Prediction":

    import pickle
    import numpy as np
    import pandas as pd

    # ===============================
    # PREMIUM UI CSS (same as app.py)
    # ===============================
    st.markdown("""
    <style>

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f4f6f8;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        padding-top: 2rem;
    }

    h1 { color: #3a0ca3; }
    h2, h3 { color: #6a040f; }

    .card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }

    div[data-baseweb="slider"] > div {
        color: #9d0208;
    }

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
    # Create two columns
    col1, col2 = st.columns([1.2, 2])   # adjust ratio if needed

    with col1:
        st.image("banner.png", use_container_width=True)

    with col2:
        st.title("🍷 Wine Quality Prediction")
        st.markdown(
        "Predict the **quality category of wine** (*Average, Good, Excellent*) "
        "using physicochemical properties and **wine color**."
        )


    st.divider()

    # ===============================
    # INPUT SECTION
    # ===============================
    st.subheader("🧪 Physicochemical Properties", divider='violet')
    st.subheader("🔧 Enter Wine Characteristics")

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
    st.markdown("""
<style>

/* Style all primary buttons */
div.stButton > button:first-child {
    background-color: #e63946;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s;
}

/* Hover effect */
div.stButton > button:first-child:hover {
    background-color: #ff7f11;
    color: white;
}

</style>
""", unsafe_allow_html=True)

    
    if st.button("🔍 Predict Wine Quality", use_container_width=True):

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

        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        st.subheader("🍾 Prediction Result")

        if prediction == "Excellent":
            st.markdown('<div class="result-card excellent">🌟 Excellent Quality Wine</div>', unsafe_allow_html=True)
        elif prediction == "Good":
            st.markdown('<div class="result-card good">🍷 Good Quality Wine</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card average">⚠️ Average Quality Wine</div>', unsafe_allow_html=True)

        st.subheader("🧾 Entered Wine Details")
        st.dataframe(input_df, use_container_width=True, height=100)
    
    st.divider()
    st.info("📌 **Note:** Predictions are based on historical wine quality data.")
    #st.info("📊 Machine Learning Powered | Streamlit App | Created by Pavan Ahire")
