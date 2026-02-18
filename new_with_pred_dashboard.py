import streamlit as st
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Wine Quality Dashboard", layout="wide")

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

    st.title("Wine Quality Dashboard")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "About Dataset",
        "Univariate Analysis",
        "Bivariate Analysis"
    ])

    # ================= ABOUT DATASET =================
    with tab1:
        show_kpis()

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # st.subheader("Dataset Information")
        # info_df = pd.DataFrame({
        #     "Column": df.columns,
        #     "Data Type": df.dtypes.values,
        #     "Non Null Count": df.count().values
        # })
        # st.dataframe(info_df)
        st.write("### Dataset Information")
        st.write(f"Shape of dataset: {df.shape}")
        st.write(f"Total rows: {df.shape[0]}")
        st.write(f"Total columns: {df.shape[1]}")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        # Project objective
        st.write("### Project Objective")
        st.info("""
        The objective of this project is to analyze wine quality data using Exploratory Data Analysis (EDA). 
        This helps to understand feature distribution, detect patterns, and identify important factors 
        affecting wine quality before building Machine Learning models.
        """)

        # Why this project is important
        st.write("### Why This Project?")
        st.success("""
        Wine quality prediction is a real-world problem in the food and beverage industry. 
        Understanding chemical properties helps producers maintain consistent product quality. 
        This project demonstrates skills in:
        - Data Cleaning
        - Exploratory Data Analysis
        - Data Visualization
        - Feature Understanding
        """)


    # ================= UNIVARIATE =================
with tab2:
    show_kpis()

    st.subheader("Univariate Analysis")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Histogram", "Box Plot"]
    )

    column = st.selectbox("Select Column", df.columns)

    # Histogram works for numeric columns
    if chart_type == "Histogram":
        if column in numeric_cols:
            fig = px.histogram(df, x=column, title=f"Histogram of {column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Histogram works only with numeric columns.")

    # Boxplot works for numeric columns
    elif chart_type == "Box Plot":
        if column in numeric_cols:
            fig = px.box(df, y=column, title=f"Box Plot of {column}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Box Plot works only with numeric columns.")


# ================= BIVARIATE =================
with tab3:
    show_kpis()

    st.subheader("Bivariate Analysis")

    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"]
    )

    graph_type = st.selectbox(
        "Select Graph Type",
        ["Scatter Plot", "Box Plot", "Bar Chart", "Heatmap"]
    )

    # -------- Numeric vs Numeric --------
    if analysis_type == "Numeric vs Numeric":

        x = st.selectbox("Select X Feature", numeric_cols)
        y = st.selectbox("Select Y Feature", numeric_cols)

        # choose color column automatically
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
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=color_col,
                    title=f"{x} vs {y}"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "Heatmap":
                corr = df[[x, y]].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "Box Plot":
                fig = px.box(
                    df,
                    x=x,
                    y=y,
                    color=color_col,
                    title=f"{x} vs {y}"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Bar chart is not suitable for Numeric vs Numeric.")


    # -------- Numeric vs Categorical --------
    elif analysis_type == "Numeric vs Categorical":

        num = st.selectbox("Select Numeric Feature", numeric_cols)
        cat = st.selectbox("Select Categorical Feature", categorical_cols)

        if graph_type == "Box Plot":
            fig = px.box(
                df,
                x=cat,
                y=num,
                color=cat,
                title=f"{num} vs {cat}"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_type == "Bar Chart":
            temp = df.groupby(cat)[num].mean().reset_index()

            fig = px.bar(
                temp,
                x=cat,
                y=num,
                color=cat,
                title=f"Average {num} by {cat}"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Use Box Plot or Bar Chart for Numeric vs Categorical.")


    # -------- Categorical vs Categorical --------
    else:

        cat1 = st.selectbox("Select First Category", categorical_cols)
        cat2 = st.selectbox("Select Second Category", categorical_cols)

        if graph_type == "Heatmap":
            cross_tab = pd.crosstab(df[cat1], df[cat2])

            fig = px.imshow(
                cross_tab,
                text_auto=True,
                title=f"{cat1} vs {cat2}"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_type == "Bar Chart":
            cross_tab = pd.crosstab(df[cat1], df[cat2]).reset_index()

            fig = px.bar(
                cross_tab,
                x=cat1,
                y=cross_tab.columns[1],
                color=cat1,
                title=f"{cat1} vs {cat2}"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Use Heatmap or Bar Chart for Categorical vs Categorical.")

# ================= FOOTER =================
st.markdown("---")
st.info(
    "This dashboard performs Exploratory Data Analysis (EDA) on the cleaned Wine Quality dataset "
    "using interactive visualizations."
)
# ================= MODEL PREDICTION =================
    

    