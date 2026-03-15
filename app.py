import streamlit as st
import pandas as pd

from core.data_loader import load_dataset
from core.dataset_info import show_dataset_info

from modules import (
    data_lab,
    cleaning_lab,
    descriptive_lab,
    visualization_lab,
    hypothesis_lab,
    regression_lab,
    anova_lab,
    chi_square_lab,
)

from ai import machine_learning, logistic_regression
from utils.styling import apply_style


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="StatX Statistical Laboratory",
    layout="wide"
)

apply_style()

st.title("StatX Statistical Software Laboratory")

# ---------------------------------
# Sidebar
# ---------------------------------

lab = st.sidebar.selectbox(
    "Select Laboratory",
    [
        "Welcome",
        "Data Lab",
        "Data Cleaning Lab",
        "Descriptive Statistics Lab",
        "Visualization Lab",
        "Hypothesis Testing Lab",
        "Regression Lab",
        "Logistic Regression Lab",
        "ANOVA Lab",
        "Chi-Square Test Lab",
        "Machine Learning Lab"
    ]
)

# ---------------------------------
# Dataset Upload
# ---------------------------------

df = load_dataset()

if df is not None:
    show_dataset_info(df)

# ---------------------------------
# Navigation
# ---------------------------------

if lab == "Data Lab":
    data_lab.run(df)

elif lab == "Data Cleaning Lab":
    cleaning_lab.run(df)

elif lab == "Descriptive Statistics Lab":
    descriptive_lab.run(df)

elif lab == "Visualization Lab":
    visualization_lab.run(df)

elif lab == "Hypothesis Testing Lab":
    hypothesis_lab.run(df)

elif lab == "Regression Lab":
    regression_lab.run(df)

elif lab == "Logistic Regression Lab":
    logistic_regression.run(df)

elif lab == "ANOVA Lab":
    anova_lab.run(df)

elif lab == "Chi-Square Test Lab":
    chi_square_lab.run(df)

elif lab == "Machine Learning Lab":
    machine_learning.run(df)
