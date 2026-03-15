import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------
# IMPORT STATX MODULE REGISTRY
# ------------------------------------------------

import modules as statx


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

    df = None

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

            st.sidebar.success("Dataset loaded successfully")

            st.write("### Dataset Preview")
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
            "AI Statistical Advisor",
            "AI Discovery Lab",
            "Autonomous Scientific Discovery",
            "Descriptive Statistics",
            "EDA",
            "Data Lab",
            "Cleaning Lab",
            "Visualization",
            "Hypothesis Testing",
            "Chi-Square Test",
            "ANOVA",
            "Regression",
            "Factor Analysis",
            "Cluster Analysis",
            "Multivariate Analysis",
            "Bayesian Statistics",
            "Simulation",
            "Time Series",
            "Spatial Statistics",
            "Survival Analysis",
            "Econometrics",
            "Machine Learning",
            "Biostatistics",
            "Medical Biostatistics",
            "Biometrics",
            "Bioinformatics",
            "Genomics DNA Engine",
            "Systems Biology",
            "Chemoinformatics",
            "Drug Discovery",
            "Statistical Physics",
            "Bioenergy",
            "Global Intelligence",
            "Research Paper Generator",
            "Research Reporting",
            "Statistical Consultant"
        ]
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

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    # ------------------------------------------------
    # AI MODULES
    # ------------------------------------------------

    if module == "AI Statistical Advisor":
        statx.ai_statistical_advisor.run(df)

    elif module == "AI Discovery Lab":
        statx.ai_discovery_lab.run(df)

    elif module == "Autonomous Scientific Discovery":
        statx.autonomous_scientific_discovery.run(df)

    # ------------------------------------------------
    # CORE STATISTICS
    # ------------------------------------------------

    elif module == "Descriptive Statistics":
        statx.descriptive_lab.run(df)

    elif module == "EDA":
        statx.eda_lab.run(df)

    elif module == "Data Lab":
        statx.data_lab.run(df)

    elif module == "Cleaning Lab":
        statx.cleaning_lab.run(df)

    elif module == "Visualization":
        statx.visualization_lab.run(df)

    # ------------------------------------------------
    # STATISTICAL ANALYSIS
    # ------------------------------------------------

    elif module == "Hypothesis Testing":
        statx.hypothesis_lab.run(df)

    elif module == "Chi-Square Test":
        statx.chi_square_lab.run(df)

    elif module == "ANOVA":
        statx.anova_lab.run(df)

    elif module == "Regression":
        statx.regression_lab.run(df)

    elif module == "Factor Analysis":
        statx.factor_lab.run(df)

    elif module == "Cluster Analysis":
        statx.cluster_lab.run(df)

    elif module == "Multivariate Analysis":
        statx.multivariate_lab.run(df)

    # ------------------------------------------------
    # ADVANCED STATISTICS
    # ------------------------------------------------

    elif module == "Bayesian Statistics":
        statx.bayesian_lab.run(df)

    elif module == "Simulation":
        statx.simulation_lab.run(df)

    elif module == "Time Series":
        statx.time_series_lab.run(df)

    elif module == "Spatial Statistics":
        statx.spatial_statistics_lab.run(df)

    elif module == "Survival Analysis":
        statx.survival_lab.run(df)

    # ------------------------------------------------
    # ECONOMETRICS & ML
    # ------------------------------------------------

    elif module == "Econometrics":
        statx.econometrics_lab.run(df)

    elif module == "Machine Learning":
        statx.machine_learning_lab.run(df)

    # ------------------------------------------------
    # BIOSTATISTICS
    # ------------------------------------------------

    elif module == "Biostatistics":
        statx.biostatistics.run(df)

    elif module == "Medical Biostatistics":
        statx.biostatistics_medical_lab.run(df)

    elif module == "Biometrics":
        statx.biometrics_modeling.run(df)

    # ------------------------------------------------
    # BIOINFORMATICS
    # ------------------------------------------------

    elif module == "Bioinformatics":
        statx.bioinformatics.run(df)

    elif module == "Genomics DNA Engine":
        statx.genomics_dna_engine.run(df)

    elif module == "Systems Biology":
        statx.systems_biology_omics_lab.run(df)

    # ------------------------------------------------
    # CHEMISTRY
    # ------------------------------------------------

    elif module == "Chemoinformatics":
        statx.chemoinformatics.run(df)

    elif module == "Drug Discovery":
        statx.drug_discovery_lab.run(df)

    # ------------------------------------------------
    # PHYSICS
    # ------------------------------------------------

    elif module == "Statistical Physics":
        statx.statistical_physics.run(df)

    elif module == "Bioenergy":
        statx.bioenergy.run(df)

    # ------------------------------------------------
    # GLOBAL SYSTEMS
    # ------------------------------------------------

    elif module == "Global Intelligence":
        statx.global_intelligence_lab.run(df)

    # ------------------------------------------------
    # RESEARCH
    # ------------------------------------------------

    elif module == "Research Paper Generator":
        statx.research_paper_generator.run(df)

    elif module == "Research Reporting":
        statx.research_reporting_lab.run(df)

    # ------------------------------------------------
    # CONSULTING
    # ------------------------------------------------

    elif module == "Statistical Consultant":
        statx.stat_consultant.run(df)


# ------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------

if __name__ == "__main__":
    run_statx()
