import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------
# DATA SUMMARY
# ---------------------------------------------------

def dataset_summary(df):

    summary = {

        "observations":df.shape[0],
        "variables":df.shape[1],
        "missing":df.isnull().sum().sum()

    }

    return summary


# ---------------------------------------------------
# BASIC ANALYSIS
# ---------------------------------------------------

def basic_analysis(df):

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr()

    y = numeric.iloc[:,-1]
    X = numeric.iloc[:,:-1]

    model = LinearRegression()
    model.fit(X,y)

    r2 = model.score(X,y)

    return corr, r2


# ---------------------------------------------------
# ABSTRACT GENERATOR
# ---------------------------------------------------

def generate_abstract(summary, r2):

    return f"""
This study analyzes a dataset containing
{summary['observations']} observations and
{summary['variables']} variables.

Statistical analysis and predictive modeling
were conducted to identify patterns and
relationships among variables.

The developed model explained approximately
{r2*100:.1f}% of variation in the outcome variable.

The results demonstrate the potential
importance of several predictors in
explaining the studied phenomenon.
"""


# ---------------------------------------------------
# INTRODUCTION GENERATOR
# ---------------------------------------------------

def generate_introduction():

    return """
The increasing availability of large datasets has
transformed scientific research across multiple
disciplines. Statistical and machine learning
methods allow researchers to discover patterns,
test hypotheses, and build predictive models.

This study aims to explore relationships among
variables in the dataset using modern statistical
methods and to develop predictive models capable
of explaining the observed variation.
"""


# ---------------------------------------------------
# METHODS GENERATOR
# ---------------------------------------------------

def generate_methods():

    return """
Data were analyzed using statistical methods
implemented in the StatX analytical platform.

Descriptive statistics were first computed to
summarize the structure of the dataset.

Correlation analysis was then used to identify
relationships between numeric variables.

A linear regression model was constructed to
evaluate predictive relationships between
independent variables and the outcome variable.
"""


# ---------------------------------------------------
# RESULTS GENERATOR
# ---------------------------------------------------

def generate_results(r2):

    return f"""
The statistical analysis revealed several
relationships among the variables.

Regression modeling indicated that the
predictor variables collectively explain
approximately {r2*100:.1f}% of the variation
in the dependent variable.
"""


# ---------------------------------------------------
# DISCUSSION GENERATOR
# ---------------------------------------------------

def generate_discussion():

    return """
The findings highlight important statistical
relationships within the dataset.

These results may have implications for
understanding the mechanisms underlying the
observed patterns.

Further research could incorporate additional
variables, more complex models, or external
datasets to validate and extend these findings.
"""


# ---------------------------------------------------
# CONCLUSION GENERATOR
# ---------------------------------------------------

def generate_conclusion():

    return """
This study demonstrated the use of statistical
analysis and predictive modeling to explore
relationships within a dataset.

The results show that modern analytical tools
can effectively identify meaningful patterns
and generate insights that support scientific
research and decision making.
"""


# ---------------------------------------------------
# FIGURE GENERATOR
# ---------------------------------------------------

def generate_figure(df):

    numeric = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots()

    numeric.hist(ax=ax)

    return fig


# ---------------------------------------------------
# TABLE GENERATOR
# ---------------------------------------------------

def generate_table(df):

    return df.describe()


# ---------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------

def run(df):

    st.title("StatX Automatic Research Paper Generator")

    if df is None:

        st.warning("Upload dataset first")

        return

    summary = dataset_summary(df)

    corr, r2 = basic_analysis(df)

    st.header("Generated Research Paper")

# ABSTRACT

    st.subheader("Abstract")

    st.write(generate_abstract(summary, r2))

# INTRODUCTION

    st.subheader("Introduction")

    st.write(generate_introduction())

# METHODS

    st.subheader("Methods")

    st.write(generate_methods())

# RESULTS

    st.subheader("Results")

    st.write(generate_results(r2))

# TABLE

    st.subheader("Table 1: Descriptive Statistics")

    table = generate_table(df)

    st.dataframe(table)

# FIGURE

    st.subheader("Figure 1: Variable Distributions")

    fig = generate_figure(df)

    st.pyplot(fig)

# DISCUSSION

    st.subheader("Discussion")

    st.write(generate_discussion())

# CONCLUSION

    st.subheader("Conclusion")

    st.write(generate_conclusion())
