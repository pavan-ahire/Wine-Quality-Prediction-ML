# 🍷 Wine Quality Prediction & Analysis using Machine Learning

---

## 🔍 Project Overview
Wine quality assessment is a crucial task in the food and beverage industry, directly impacting customer satisfaction and business value.  
Traditionally, wine quality evaluation is subjective and time-consuming.

This project focuses on **predicting wine quality using machine learning** and providing **data-driven insights through interactive dashboards and a live prediction application**.

The project includes:
- End-to-end **Machine Learning pipeline**
- **Exploratory Data Analysis (EDA)**
- Multiple ML model training and comparison
- **Live wine quality prediction app**
- **Interactive EDA dashboard**

---

## 🚀 Live Deployment
The project is deployed using **Streamlit Cloud**, providing both prediction and analytics capabilities.

### 🔗 Live Links
- **Wine Quality Prediction App**  
  👉 https://wine-quality-prediction-machine-learning.streamlit.app/

- **Wine Quality EDA Dashboard**  
  👉 https://wine-quality-eda-dashboard.streamlit.app/

---

## 🎯 Objectives
- Analyze physicochemical properties of wine
- Understand factors influencing wine quality
- Perform in-depth exploratory data analysis
- Build and compare multiple ML models
- Predict wine quality for new samples
- Provide insights through visual dashboards

---

## 💼 Business Problem & Impact
Wine producers and distributors need objective, data-driven methods to ensure consistent product quality.

This project helps businesses to:
- Identify **key parameters affecting wine quality**
- Improve quality control processes
- Reduce dependency on manual testing
- Enhance product consistency and customer trust

This solution supports **quality assurance, production, and analytics teams**.

---

## 🔄 End-to-End ML Pipeline
The project follows a **production-oriented ML workflow**:

1. Data collection & understanding  
2. Data cleaning & preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature engineering  
5. Model training  
6. Model comparison & evaluation  
7. Best model selection  
8. Model persistence (`.pkl`)  
9. Deployment using Streamlit  
10. Dashboard development for analysis  

---

## 🧠 Machine Learning Models Used
The following algorithms were implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

📌 **Random Forest** was selected for deployment due to:
- Higher accuracy
- Strong performance on non-linear data
- Robustness to outliers and feature interactions

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to:
- Understand distribution of wine quality scores
- Analyze physicochemical properties of wine
- Identify correlations between features
- Detect outliers and skewed distributions
- Understand feature importance

Visualizations include:
- Quality distribution plots
- Correlation heatmaps
- Boxplots for outlier detection
- Feature vs quality analysis
- Density and histogram plots

---

## 🧩 Feature Engineering & Preprocessing
Key preprocessing steps:
- Handling missing values
- Outlier analysis
- Feature scaling using `StandardScaler`
- Ensuring feature consistency during inference
- Saving preprocessing objects (`scaler.pkl`, `feature_columns.pkl`)

This ensures **training and prediction pipelines remain identical**.

---

## 🧪 Model Evaluation Metrics
Models were evaluated using:
- Accuracy (for classification)
- Mean Squared Error (for regression)
- R² Score
- Confusion Matrix
- Cross-validation scores

---

## 🖥️ Streamlit Prediction App Features
- Simple and clean UI
- Accepts wine chemical properties as input
- Predicts wine quality score/category
- Real-time ML inference
- User-friendly and business-ready interface

---

## 📈 Streamlit EDA Dashboard Features
- Interactive charts and plots
- Wine quality distribution insights
- Feature correlation analysis
- Outlier visualization
- Dynamic and responsive layout

---

## 🛠️ Technologies Used
- **Language**: Python  
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit
- **Deployment**: Streamlit Cloud  
- **Version Control**: Git & GitHub  

---

## 📂 Project Folder Structure

```text
Wine-Quality-Prediction-ML/
│
├── app/
│   └── Streamlit app components and UI logic
│
├── dashboard/
│   └── EDA dashboard scripts and visualizations
│
├── data/
│   └── Raw and cleaned wine datasets
│
├── notebooks/
│   └── EDA and model training notebooks
│
├── model/
│   └── Saved ML model artifacts (pkl files)
│
├── README.md
├── app.py                      # Streamlit prediction app
├── dashboard.py                # Streamlit EDA dashboard
├── wine_model.pkl              # Trained ML model
├── feature_columns.pkl         # Model feature columns
├── scaler.pkl                  # Feature scaling object
├── requirements.txt            # Project dependencies
└── wine_quality.csv            # Dataset
```
---
## How to Run the Project Locally

Follow the steps below to run the project on your local machine:

### 1️⃣ Clone the Repository
```
git clone https://github.com/pavan-ahire/Wine-Quality-Prediction-ML.git
cd Wine-Quality-Prediction-ML
```

### Install Required Dependencies
- pip install -r requirements.txt
  
### Run streamlit prediction app
- streamlit run app.py
  
### Run Streamlit Dashboard
-streamlit run dashboard.py

---
## 🧠 Key Skills Demonstrated

- Machine Learning model development and evaluation
- Exploratory Data Analysis (EDA)
- Feature engineering and data preprocessing
- Model serialization and reuse (`.pkl` files)
- Deployment of ML models using Streamlit
- Dashboard creation for business insights
- End-to-end project implementation
- Version control using Git & GitHub
---
## 👨‍💻 Author

**Pavan Ahire**


 Aspiring Data Scientist | Machine Learning & Analytics Enthusiast
- [🔗 GitHub](https://github.com/pavan-ahire)
- [🔗 LinkedIn](https://www.linkedin.com/in/pavan-ahire-260940364/)
