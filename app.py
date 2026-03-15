import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------
# IMPORT STATX MODULE REGISTRY
# ------------------------------------------------

import modules as statx


# ------------------------------------------------
# UI NAME → MODULE KEY MAP
# ------------------------------------------------

MODULE_MAP = {

    "AI Statistical Advisor": "ai_statistical_advisor",
    "AI Discovery Lab": "ai_discovery_lab",
    "Autonomous Scientific Discovery": "autonomous_scientific_discovery",

    "Descriptive Statistics": "descriptive_lab",
    "EDA": "eda_lab",
    "Data Lab": "data_lab",
    "Cleaning Lab": "cleaning_lab",
    "Visualization": "visualization_lab",

    "Hypothesis Testing": "hypothesis_lab",
    "Chi-Square Test": "chi_square_lab",
    "ANOVA": "anova_lab",
    "Regression": "regression_lab",

    "Factor Analysis": "factor_lab",
    "Cluster Analysis": "cluster_lab",
    "Multivariate Analysis": "multivariate_lab",

    "Bayesian Statistics": "bayesian_lab",
    "Simulation": "simulation_lab",
    "Time Series": "time_series_lab",
    "Spatial Statistics": "spatial_statistics_lab",
    "Survival Analysis": "survival_lab",

    "Econometrics": "econometrics_lab",
    "Machine Learning": "machine_learning_lab",

    "Biostatistics": "biostatistics",
    "Medical Biostatistics": "biostatistics_medical_lab",
    "Biometrics": "biometrics_modeling",

    "Bioinformatics": "bioinformatics",
    "Genomics DNA Engine": "genomics_dna_engine",
    "Systems Biology": "systems_biology_omics_lab",

    "Chemoinformatics": "chemoinformatics",
    "Drug Discovery": "drug_discovery_lab",

    "Statistical Physics": "statistical_physics",
    "Bioenergy": "bioenergy",

    "Global Intelligence": "global_intelligence_lab",

    "Research Paper Generator": "research_paper_generator",
    "Research Reporting": "research_reporting_lab",

    "Statistical Consultant": "stat_consultant"
}


# ------------------------------------------------
# MAIN STATX PLATFORM FUNCTION
# ------------------------------------------------

def run_statx():

    st.set_page_config(
        page_title="StatX Global Scientific Platform",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 StatX Global Scientific Platform")
    st.subheader("Unified Environment for Statistics, AI, and Scientific Discovery")

    # ------------------------------------------------
    # DATASET MANAGER
    # ------------------------------------------------

    st.sidebar.header("Dataset Manager")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv","xlsx","xls","txt","json","parquet"]
    )

    if uploaded_file is not None:

        try:

            name = uploaded_file.name.lower()

            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif name.endswith(".xlsx") or name.endswith(".xls"):
                df = pd.read_excel(uploaded_file)

            elif name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, sep=None, engine="python")

            elif name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            elif name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)

            st.session_state["dataset"] = df

            st.sidebar.success("Dataset loaded successfully")

            st.write("### Dataset Preview")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # ------------------------------------------------
    # MODULE NAVIGATION
    # ------------------------------------------------

    st.sidebar.header("StatX Modules")

    module = st.sidebar.selectbox(
        "Select Analysis Module",
        ["Home"] + list(MODULE_MAP.keys())
    )

    # ------------------------------------------------
    # HOME
    # ------------------------------------------------

    if module == "Home":

        st.write("Welcome to **StatX Scientific Platform**.")

        st.markdown("""
        **StatX integrates**

        - Advanced statistical analysis  
        - Machine learning  
        - Bioinformatics  
        - Scientific discovery AI  
        - Global scientific data networks  
        """)

        return

    # ------------------------------------------------
    # CHECK DATASET
    # ------------------------------------------------

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return

    # ------------------------------------------------
    # LOAD MODULE DYNAMICALLY
    # ------------------------------------------------

    module_key = MODULE_MAP.get(module)

    if module_key:

        mod = statx.load(module_key)

        if mod and hasattr(mod, "run"):
            mod.run()

        else:
            st.error(f"Module '{module_key}' failed to load.")
