import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import StatX labs
import modules.descriptive_statistics as descriptive
import modules.graph_engine as graphs
import modules.hypothesis_testing as tests
import modules.regression_lab as regression
import modules.time_series_lab as timeseries
import modules.machine_learning_lab as ml
import modules.anova_lab as anova


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="StatX Professional",
    layout="wide",
    page_icon="📊"
)

# ------------------------------------------------
# DATASET MANAGER
# ------------------------------------------------

def dataset_manager():

    st.sidebar.header("Dataset Manager")

    uploaded = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv","xlsx"]
    )

    df=None

    if uploaded:

        if uploaded.name.endswith(".csv"):
            df=pd.read_csv(uploaded)

        else:
            df=pd.read_excel(uploaded)

    return df


# ------------------------------------------------
# NAVIGATION PANEL
# ------------------------------------------------

def navigation_panel():

    st.sidebar.header("StatX Laboratories")

    module=st.sidebar.radio(

        "Select Analysis",

        [

        "Dashboard",
        "Descriptive Statistics",
        "Graph Engine",
        "Hypothesis Testing",
        "Regression Modeling",
        "ANOVA & Experimental Design",
        "Machine Learning",
        "Time Series",
        "StatX Scientific AI"

        ]

    )

    return module


# ------------------------------------------------
# PROFESSIONAL DASHBOARD
# ------------------------------------------------

def dashboard(df):

    st.title("StatX Professional Dashboard")

    if df is None:

        st.warning("Upload dataset to view dashboard")
        return

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Observations",df.shape[0])
    c2.metric("Variables",df.shape[1])
    c3.metric("Missing Values",df.isna().sum().sum())
    c4.metric("Numeric Variables",len(df.select_dtypes("number").columns))

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    st.subheader("Correlation Heatmap")

    fig=px.imshow(df.corr(numeric_only=True))

    st.plotly_chart(fig,use_container_width=True)


# ------------------------------------------------
# DRAG-DROP MODEL BUILDER
# ------------------------------------------------

def model_builder(df):

    st.subheader("Drag-and-Drop Model Builder")

    if df is None:

        st.warning("Upload dataset first")
        return

    columns=df.columns.tolist()

    target=st.selectbox("Target Variable",columns)

    predictors=st.multiselect(

        "Drag predictors here",

        [c for c in columns if c!=target]

    )

    model_type=st.selectbox(

        "Model Type",

        [

        "Linear Regression",
        "Logistic Regression",
        "Random Forest",
        "K-Means Clustering"

        ]

    )

    if st.button("Build Model"):

        st.success("Model configuration created")

        st.write({

        "Target":target,
        "Predictors":predictors,
        "Model":model_type

        })


# ------------------------------------------------
# ANALYSIS WORKSPACE
# ------------------------------------------------

def analysis_workspace(module,df):

    st.header("Analysis Workspace")

    if module=="Dashboard":

        dashboard(df)

    elif module=="Descriptive Statistics":

        descriptive.run(df)

    elif module=="Graph Engine":

        graphs.run(df)

    elif module=="Hypothesis Testing":

        tests.run(df)

    elif module=="Regression Modeling":

        regression.run(df)

    elif module=="ANOVA & Experimental Design":

        anova.run(df)

    elif module=="Machine Learning":

        ml.run(df)

    elif module=="Time Series":

        timeseries.run(df)

    else:

        st.write("AI module coming soon")


# ------------------------------------------------
# PROFESSIONAL LAYOUT
# ------------------------------------------------

def run():

    st.title("StatX Professional Statistical Platform")

    df=dataset_manager()

    module=navigation_panel()

    col1,col2=st.columns([3,1])

    with col1:

        analysis_workspace(module,df)

    with col2:

        model_builder(df)


# ------------------------------------------------
# RUN INTERFACE
# ------------------------------------------------

run()
