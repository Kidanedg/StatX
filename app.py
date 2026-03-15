import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import statsmodels.api as sm
from lifelines import KaplanMeierFitter

# ---------------------------------
# App Title
# ---------------------------------

st.set_page_config(page_title="Statistical Laboratory", layout="wide")
# ---------------------------------
# Custom CSS Styling
# ---------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #0a4f9c;
}

h2, h3 {
    color: #1f77b4;
}

.sidebar .sidebar-content {
    background-color: #0a4f9c;
    color: white;
}

.statx-logo {
    position: fixed;
    top: 10px;
    right: 20px;
    font-size: 28px;
    font-weight: bold;
    color: #0a4f9c;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="statx-logo">StatX</div>', unsafe_allow_html=True)

st.title("Statistical Software Laboratory")

# ---------------------------------
# Sidebar Menu
# ---------------------------------

lab = st.sidebar.selectbox(
    "Select Laboratory",
    [
        "Welcome",
        "Data Lab",
        "Data Cleaning Lab",
        "Descriptive Statistics Lab",
        "Visualization Lab",
        "Interactive Visualization Lab",
        "Hypothesis Testing Lab",
        "Regression Lab",
        "Logistic Regression Lab",
        "ANOVA Lab",
        "Chi-Square Test Lab",
        "Time Series Analysis Lab",
        "Quality Control Lab",
        "Multivariate Analysis Lab",
        "Cluster Analysis Lab",
        "Factor Analysis Lab",
        "Structural Equation Modeling Lab",
        "Bayesian Statistics Lab",
        "Survival Analysis Lab",
        "Experimental Design Lab",
        "Machine Learning Lab",
        "Simulation Lab",
        "Report Generator",
        "Help"
    ]
)

# ---------------------------------
# ---------------------------------
# DATASET UPLOAD SYSTEM
# ---------------------------------

st.sidebar.header("Dataset Manager")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=[
        "csv","xlsx","xls",
        "txt","json",
        "parquet",
        "dta","sav"
    ]
)

df = None

if uploaded_file:

    try:

        file_name = uploaded_file.name.lower()

        # CSV
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # Excel
        elif file_name.endswith((".xlsx",".xls")):
            df = pd.read_excel(uploaded_file)

        # TXT
        elif file_name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")

        # JSON
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        # Parquet
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)

        # Stata
        elif file_name.endswith(".dta"):
            df = pd.read_stata(uploaded_file)

        # SPSS
        elif file_name.endswith(".sav"):
            import pyreadstat
            df, meta = pyreadstat.read_sav(uploaded_file)

        st.success("Dataset loaded successfully!")

    except Exception as e:

        st.error(f"Error loading file: {e}")

# ---------------------------------
# DATASET PREVIEW
# ---------------------------------

if df is not None:

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------
    # DATASET INFORMATION
    # ---------------------------------

    st.subheader("Dataset Information")

    col1,col2,col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    # ---------------------------------
    # VARIABLE TYPES
    # ---------------------------------

    st.subheader("Variable Types")

    var_types = pd.DataFrame({

        "Variable":df.columns,
        "Type":df.dtypes

    })

    st.dataframe(var_types)

    # ---------------------------------
    # MISSING VALUE DIAGNOSTICS
    # ---------------------------------

    st.subheader("Missing Value Analysis")

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({

        "Variable":missing.index,
        "Missing Count":missing.values

    })

    st.dataframe(missing_table)

    # ---------------------------------
    # DATA CLEANING OPTIONS
    # ---------------------------------

    st.subheader("Data Cleaning Options")

    option = st.selectbox(
        "Handle Missing Values",
        [
        "Do Nothing",
        "Drop Missing Rows",
        "Fill with Mean",
        "Fill with Median",
        "Fill with Mode"
        ]
    )

    if option == "Drop Missing Rows":
        df = df.dropna()

    elif option == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))

    elif option == "Fill with Median":
        df = df.fillna(df.median(numeric_only=True))

    elif option == "Fill with Mode":
        df = df.fillna(df.mode().iloc[0])

    # ---------------------------------
    # DATASET DOWNLOAD
    # ---------------------------------

    st.subheader("Download Processed Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )
# ---------------------------------
# Welcome
# ---------------------------------

if lab == "Welcome":

    st.header("Welcome to StatX Software: Statistical Weblab")

    st.markdown("""
    **Founded by:**  
    **Dr. Kidane Desta**  
    Department of Statistics  
    College of Natural and Computational Sciences  
    Aksum University

    ---

    This platform provides multiple **interactive statistical laboratories**
    designed for education, research, and advanced data analysis.

    ### How to Use
    1. Upload dataset
    2. Select a statistical laboratory
    3. Choose variables
    4. Run analysis
    5. Interpret results
    """)

# ---------------------------------
# Data Lab
# ---------------------------------

elif lab == "Data Lab":

    st.header("Data Laboratory")

    if df is not None:

        st.dataframe(df)

        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns)
        st.write("Data Types")
        st.write(df.dtypes)

    else:
        st.warning("Upload dataset first")

# ---------------------------------
# Data Cleaning
# ---------------------------------

elif lab == "Data Cleaning Lab":

    st.header("Data Cleaning")

    if df is not None:

        st.write("Missing Values")
        st.write(df.isnull().sum())

        if st.button("Drop Missing Values"):
            df = df.dropna()
            st.success("Missing values removed")

        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed")

# ---------------------------------
# Descriptive Statistics
# ---------------------------------

elif lab == "Descriptive Statistics Lab":

    st.header("Descriptive Statistics")

    if df is not None:

        st.write(df.describe())

# ---------------------------------
# Visualization
# ---------------------------------

elif lab == "Visualization Lab":

    st.header("Visualization")

    if df is not None:

        column = st.selectbox("Select Variable", df.columns)

        fig, ax = plt.subplots()

        sns.histplot(df[column], kde=True)

        st.pyplot(fig)

# ---------------------------------
# Interactive Visualization
# ---------------------------------

elif lab == "Interactive Visualization Lab":

    st.header("Interactive Visualization")

    if df is not None:

        x = st.selectbox("X Variable", df.columns)
        y = st.selectbox("Y Variable", df.columns)

        fig = px.scatter(df, x=x, y=y)

        st.plotly_chart(fig)

# ---------------------------------
# Hypothesis Testing
# ---------------------------------

elif lab == "Hypothesis Testing Lab":

    st.header("Hypothesis Testing")

    if df is not None:

        column = st.selectbox("Variable", df.columns)

        t,p = stats.ttest_1samp(df[column],0)

        st.write("T statistic:",t)
        st.write("P value:",p)

# ---------------------------------
# Regression
# ---------------------------------

elif lab == "Regression Lab":

    st.header("Linear Regression")

    if df is not None:

        x = st.selectbox("Independent Variable", df.columns)
        y = st.selectbox("Dependent Variable", df.columns)

        X = df[[x]]
        Y = df[y]

        model = LinearRegression()

        model.fit(X,Y)

        pred = model.predict(X)

        st.write("Coefficient:",model.coef_)
        st.write("Intercept:",model.intercept_)
        st.write("Mean Squared Error:",mean_squared_error(Y,pred))

# ---------------------------------
# Logistic Regression
# ---------------------------------

elif lab == "Logistic Regression Lab":

    st.header("Logistic Regression")

    if df is not None:

        x = st.selectbox("Predictor", df.columns)
        y = st.selectbox("Binary Target", df.columns)

        X = df[[x]]
        Y = df[y]

        model = LogisticRegression()

        model.fit(X,Y)

        pred = model.predict(X)

        st.write("Accuracy:",accuracy_score(Y,pred))

# ---------------------------------
# ANOVA
# ---------------------------------

elif lab == "ANOVA Lab":

    st.header("ANOVA")

    if df is not None:

        num = st.selectbox("Numeric Variable", df.columns)
        group = st.selectbox("Group Variable", df.columns)

        groups = df.groupby(group)[num].apply(list)

        f,p = stats.f_oneway(*groups)

        st.write("F statistic:",f)
        st.write("P value:",p)

# ---------------------------------
# Chi Square
# ---------------------------------

elif lab == "Chi-Square Test Lab":

    st.header("Chi-Square Test")

    if df is not None:

        col1 = st.selectbox("Variable 1", df.columns)
        col2 = st.selectbox("Variable 2", df.columns)

        table = pd.crosstab(df[col1],df[col2])

        chi,p,dof,exp = stats.chi2_contingency(table)

        st.write("Chi-square:",chi)
        st.write("P-value:",p)

# ---------------------------------
# Time Series
# ---------------------------------

elif lab == "Time Series Analysis Lab":

    st.header("Time Series Analysis")

    if df is not None:

        column = st.selectbox("Time Series Variable",df.columns)

        st.line_chart(df[column])

# ---------------------------------
# Quality Control
# ---------------------------------

elif lab == "Quality Control Lab":

    st.header("Control Chart")

    if df is not None:

        column = st.selectbox("Process Variable",df.columns)

        mean = df[column].mean()
        std = df[column].std()

        ucl = mean + 3*std
        lcl = mean - 3*std

        st.write("UCL:",ucl)
        st.write("LCL:",lcl)

        st.line_chart(df[column])

# ---------------------------------
# Multivariate Analysis
# ---------------------------------

elif lab == "Multivariate Analysis Lab":

    st.header("Correlation Matrix")

    if df is not None:

        corr = df.corr()

        fig,ax = plt.subplots()

        sns.heatmap(corr,annot=True)

        st.pyplot(fig)

# ---------------------------------
# Cluster Analysis
# ---------------------------------

elif lab == "Cluster Analysis Lab":

    st.header("KMeans Clustering")

    if df is not None:

        k = st.slider("Clusters",2,10,3)

        numeric = df.select_dtypes(include=np.number)

        model = KMeans(n_clusters=k)

        model.fit(numeric)

        st.write("Cluster Centers")

        st.write(model.cluster_centers_)

# ---------------------------------
# Factor Analysis
# ---------------------------------

elif lab == "Factor Analysis Lab":

    st.header("Factor Analysis")

    if df is not None:

        numeric = df.select_dtypes(include=np.number)

        fa = FactorAnalysis(n_components=2)

        fa.fit(numeric)

        st.write("Factor Loadings")

        st.write(fa.components_)

# ---------------------------------
# SEM
# ---------------------------------

elif lab == "Structural Equation Modeling Lab":

    st.header("Structural Equation Modeling")

    st.info("SEM requires predefined model specification.")

# ---------------------------------
# Bayesian
# ---------------------------------

elif lab == "Bayesian Statistics Lab":

    st.header("Bayesian Statistics")

    data = np.random.normal(0,1,1000)

    fig,ax = plt.subplots()

    sns.histplot(data,kde=True)

    st.pyplot(fig)

# ---------------------------------
# Survival Analysis
# ---------------------------------

elif lab == "Survival Analysis Lab":

    st.header("Survival Analysis")

    if df is not None:

        time = st.selectbox("Time Variable",df.columns)
        event = st.selectbox("Event Variable",df.columns)

        kmf = KaplanMeierFitter()

        kmf.fit(df[time], event_observed=df[event])

        kmf.plot_survival_function()

        st.pyplot(plt)

# ---------------------------------
# Experimental Design
# ---------------------------------

elif lab == "Experimental Design Lab":

    st.header("Experimental Design")

    st.write("Use randomized designs and analyze using ANOVA.")

# ---------------------------------
# Machine Learning
# ---------------------------------

elif lab == "Machine Learning Lab":

    st.header("Machine Learning")

    if df is not None:

        numeric = df.select_dtypes(include=np.number)

        X = numeric.iloc[:,:-1]
        y = numeric.iloc[:,-1]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

        model = LinearRegression()

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        st.write("MSE:",mean_squared_error(y_test,pred))

# ---------------------------------
# Simulation
# ---------------------------------

elif lab == "Simulation Lab":

    st.header("Monte Carlo Simulation")

    n = st.slider("Sample Size",100,10000,1000)

    data = np.random.normal(0,1,n)

    fig,ax = plt.subplots()

    sns.histplot(data)

    st.pyplot(fig)

# ---------------------------------
# Report Generator
# ---------------------------------

elif lab == "Report Generator":

    st.header("Report Generator")

    if df is not None:

        report = df.describe().to_string()

        st.download_button(
            "Download Report",
            report,
            "statistical_report.txt"
        )

# ---------------------------------
# Help
# ---------------------------------

elif lab == "Help":

    st.header("Help")

    st.write("""
    Instructions:

    1 Upload dataset
    2 Select laboratory
    3 Choose variables
    4 View statistical results
    """)

# ---------------------------------
# Footer
# ---------------------------------

st.markdown("""
---
© Copyright 2024–2026 Dr. Kidane Desta. All rights reserved.
""")
