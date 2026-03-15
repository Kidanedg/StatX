import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------
# IMPORT STATX MODULE REGISTRY
# ------------------------------------------------
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import modules as statx


# ------------------------------------------------
# MAIN STATX PLATFORM FUNCTION
# ------------------------------------------------

def run_statx():

    # ------------------------------------------------
    # PAGE CONFIGURATION
    # ------------------------------------------------

    st.set_page_config(
        page_title="StatX Global Scientific Platform",
        page_icon="📊",
        layout="wide"
    )

    # ------------------------------------------------
    # HEADER
    # ------------------------------------------------

    st.title("📊 StatX Global Scientific Platform")
    st.subheader("Unified Environment for Statistics, AI, and Scientific Discovery")

    # ------------------------------------------------
    # DATASET MANAGER
    # ------------------------------------------------

    st.sidebar.header("Dataset Manager")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls", "txt", "json", "parquet"]
    )

    df = None

    if uploaded_file:

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

            st.success("Dataset loaded successfully")

            st.write("Preview")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # ------------------------------------------------
    # SIDEBAR MODULE NAVIGATION
    # ------------------------------------------------

    st.sidebar.header("StatX Modules")

    module = st.sidebar.selectbox(
        "Select Analysis Module",
        [

            "Home",

            # Core
            "AI Statistical Advisor",
            "AI Discovery Lab",
            "Autonomous Scientific Discovery",

            # Basic Statistics
            "Descriptive Statistics",
            "EDA",
            "Data Lab",
            "Cleaning Lab",
            "Visualization",

            # Statistical Analysis
            "Hypothesis Testing",
            "Chi-Square Test",
            "ANOVA",
            "Regression",
            "Factor Analysis",
            "Cluster Analysis",
            "Multivariate Analysis",

            # Advanced Statistics
            "Bayesian Statistics",
            "Simulation",
            "Time Series",
            "Spatial Statistics",
            "Survival Analysis",

            # ML & Econometrics
            "Econometrics",
            "Machine Learning",

            # Biostatistics
            "Biostatistics",
            "Medical Biostatistics",
            "Biometrics",

            # Bioinformatics
            "Bioinformatics",
            "Genomics DNA Engine",
            "Systems Biology",

            # Chemistry
            "Chemoinformatics",
            "Drug Discovery",

            # Physics & Energy
            "Statistical Physics",
            "Bioenergy",

            # Global Systems
            "Global Intelligence",

            # Research
            "Research Paper Generator",
            "Research Reporting",

            # Consulting
            "Statistical Consultant"
        ]
    )

    # ------------------------------------------------
    # MODULE ROUTING
    # ------------------------------------------------

    if module == "Home":

        st.write("Welcome to **StatX Scientific Platform**.")

        st.write("""
        StatX integrates:

        - Advanced statistical analysis
        - Machine learning
        - Bioinformatics
        - Scientific discovery AI
        - Global scientific data networks
        """)

    # ------------------------------------------------
    # AI MODULES
    # ------------------------------------------------

    elif module == "AI Statistical Advisor":
        statx.ai_statistical_advisor(df)

    elif module == "AI Discovery Lab":
        statx.ai_discovery_lab(df)

    elif module == "Autonomous Scientific Discovery":
        statx.autonomous_scientific_discovery(df)

    # ------------------------------------------------
    # CORE STATISTICS
    # ------------------------------------------------

    elif module == "Descriptive Statistics":
        statx.descriptive_lab(df)

    elif module == "EDA":
        statx.eda_lab(df)

    elif module == "Data Lab":
        statx.data_lab(df)

    elif module == "Cleaning Lab":
        statx.cleaning_lab(df)

    elif module == "Visualization":
        statx.visualization_lab(df)

    # ------------------------------------------------
    # STATISTICAL ANALYSIS
    # ------------------------------------------------

    elif module == "Hypothesis Testing":
        statx.hypothesis_lab(df)

    elif module == "Chi-Square Test":
        statx.chi_square_lab(df)

    elif module == "ANOVA":
        statx.anova_lab(df)

    elif module == "Regression":
        statx.regression_lab(df)

    elif module == "Factor Analysis":
        statx.factor_lab(df)

    elif module == "Cluster Analysis":
        statx.cluster_lab(df)

    elif module == "Multivariate Analysis":
        statx.multivariate_lab(df)

    # ------------------------------------------------
    # ADVANCED STATISTICS
    # ------------------------------------------------

    elif module == "Bayesian Statistics":
        statx.bayesian_lab(df)

    elif module == "Simulation":
        statx.simulation_lab(df)

    elif module == "Time Series":
        statx.time_series_lab(df)

    elif module == "Spatial Statistics":
        statx.spatial_statistics_lab(df)

    elif module == "Survival Analysis":
        statx.survival_lab(df)

    # ------------------------------------------------
    # ECONOMETRICS & ML
    # ------------------------------------------------

    elif module == "Econometrics":
        statx.econometrics_lab(df)

    elif module == "Machine Learning":
        statx.machine_learning_lab(df)

    # ------------------------------------------------
    # BIOSTATISTICS
    # ------------------------------------------------

    elif module == "Biostatistics":
        statx.biostatistics(df)

    elif module == "Medical Biostatistics":
        statx.biostatistics_medical_lab(df)

    elif module == "Biometrics":
        statx.biometrics_modeling(df)

    # ------------------------------------------------
    # BIOINFORMATICS
    # ------------------------------------------------

    elif module == "Bioinformatics":
        statx.bioinformatics(df)

    elif module == "Genomics DNA Engine":
        statx.genomics_dna_engine(df)

    elif module == "Systems Biology":
        statx.systems_biology_omics_lab(df)

    # ------------------------------------------------
    # CHEMISTRY
    # ------------------------------------------------

    elif module == "Chemoinformatics":
        statx.chemoinformatics(df)

    elif module == "Drug Discovery":
        statx.drug_discovery_lab(df)

    # ------------------------------------------------
    # PHYSICS
    # ------------------------------------------------

    elif module == "Statistical Physics":
        statx.statistical_physics(df)

    elif module == "Bioenergy":
        statx.bioenergy(df)

    # ------------------------------------------------
    # GLOBAL SYSTEMS
    # ------------------------------------------------

    elif module == "Global Intelligence":
        statx.global_intelligence_lab(df)

    # ------------------------------------------------
    # RESEARCH
    # ------------------------------------------------

    elif module == "Research Paper Generator":
        statx.research_paper_generator(df)

    elif module == "Research Reporting":
        statx.research_reporting_lab(df)

    # ------------------------------------------------
    # CONSULTING
    # ------------------------------------------------

    elif module == "Statistical Consultant":
        statx.stat_consultant(df)

# ------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------

run_statx()
