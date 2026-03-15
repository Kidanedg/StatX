import streamlit as st
import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------
# DATA TYPE DETECTION
# ---------------------------------------------------------

def detect_variable_types(df):

    types = {}

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            unique = df[col].nunique()

            if unique <= 5:
                types[col] = "Categorical (numeric-coded)"
            else:
                types[col] = "Continuous"

        else:
            types[col] = "Categorical"

    return types


# ---------------------------------------------------------
# RECOMMEND STATISTICAL TESTS
# ---------------------------------------------------------

def recommend_tests(types):

    continuous = [k for k,v in types.items() if "Continuous" in v]
    categorical = [k for k,v in types.items() if "Categorical" in v]

    recommendations = []

    if len(continuous) >= 2:

        recommendations.append("Correlation Analysis")
        recommendations.append("Simple Linear Regression")

    if len(continuous) >= 3:

        recommendations.append("Multiple Regression")

    if len(categorical) >= 2:

        recommendations.append("Chi-Square Test")

    if len(continuous) >= 1 and len(categorical) >= 1:

        recommendations.append("ANOVA")
        recommendations.append("Logistic Regression")

    return recommendations


# ---------------------------------------------------------
# NORMALITY TEST
# ---------------------------------------------------------

def check_normality(df):

    results = {}

    numeric = df.select_dtypes(include=np.number)

    for col in numeric.columns:

        stat, p = stats.shapiro(df[col].dropna())

        results[col] = {
            "Statistic":stat,
            "p-value":p,
            "Normal":p > 0.05
        }

    return results


# ---------------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------------

def detect_outliers(df):

    numeric = df.select_dtypes(include=np.number)

    outliers = {}

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        out = numeric[(numeric[col] < lower) | (numeric[col] > upper)]

        outliers[col] = len(out)

    return outliers


# ---------------------------------------------------------
# MULTICOLLINEARITY CHECK
# ---------------------------------------------------------

def multicollinearity_check(df):

    numeric = df.select_dtypes(include=np.number)

    X = numeric.dropna()

    vif = pd.DataFrame()

    vif["Variable"] = X.columns

    vif["VIF"] = [

        variance_inflation_factor(X.values,i)
        for i in range(X.shape[1])

    ]

    return vif


# ---------------------------------------------------------
# MODEL PROBLEM DETECTOR
# ---------------------------------------------------------

def detect_problems(normality, outliers, vif):

    problems = []

    for col,v in normality.items():

        if not v["Normal"]:

            problems.append(f"{col} is not normally distributed")

    for col,n in outliers.items():

        if n > 0:

            problems.append(f"{col} contains {n} potential outliers")

    for i,row in vif.iterrows():

        if row["VIF"] > 10:

            problems.append(
                f"Severe multicollinearity detected in {row['Variable']}"
            )

    return problems


# ---------------------------------------------------------
# AUTOMATIC RESEARCH INTERPRETATION
# ---------------------------------------------------------

def generate_research_interpretation(types, tests, problems):

    text = "STATISTICAL CONSULTANT REPORT\n\n"

    text += "Variable Types Detected:\n"

    for k,v in types.items():

        text += f"- {k}: {v}\n"

    text += "\nRecommended Statistical Methods:\n"

    for t in tests:

        text += f"- {t}\n"

    if problems:

        text += "\nPotential Data Issues Detected:\n"

        for p in problems:

            text += f"- {p}\n"

    else:

        text += "\nNo serious statistical problems detected.\n"

    text += """

Interpretation:

Based on the dataset structure, the recommended analyses will
allow investigation of relationships between variables,
group differences, and predictive modeling.

If assumptions such as normality or independence are violated,
robust or nonparametric alternatives should be considered.
"""

    return text


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

def run(df):

    st.title("StatX AI Statistical Consultant")

    if df is None:

        st.warning("Upload dataset first")

        return

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# ---------------------------------------------------------
# VARIABLE TYPE DETECTION
# ---------------------------------------------------------

    st.header("Automatic Data Type Detection")

    types = detect_variable_types(df)

    type_table = pd.DataFrame({

        "Variable":list(types.keys()),
        "Detected Type":list(types.values())

    })

    st.table(type_table)

# ---------------------------------------------------------
# TEST RECOMMENDATIONS
# ---------------------------------------------------------

    st.header("Recommended Statistical Analyses")

    tests = recommend_tests(types)

    for t in tests:

        st.write("✔", t)

# ---------------------------------------------------------
# NORMALITY CHECK
# ---------------------------------------------------------

    st.header("Normality Diagnostics")

    normality = check_normality(df)

    norm_table = pd.DataFrame(normality).T

    st.dataframe(norm_table)

# ---------------------------------------------------------
# OUTLIERS
# ---------------------------------------------------------

    st.header("Outlier Detection")

    outliers = detect_outliers(df)

    out_table = pd.DataFrame({

        "Variable":list(outliers.keys()),
        "Outliers":list(outliers.values())

    })

    st.table(out_table)

# ---------------------------------------------------------
# MULTICOLLINEARITY
# ---------------------------------------------------------

    st.header("Multicollinearity Diagnostics")

    vif = multicollinearity_check(df)

    st.dataframe(vif)

# ---------------------------------------------------------
# PROBLEM DETECTION
# ---------------------------------------------------------

    st.header("Model Diagnostics")

    problems = detect_problems(normality, outliers, vif)

    if problems:

        for p in problems:

            st.warning(p)

    else:

        st.success("No major statistical problems detected.")

# ---------------------------------------------------------
# FINAL AI INTERPRETATION
# ---------------------------------------------------------

    st.header("AI Statistical Interpretation")

    report = generate_research_interpretation(
        types,
        tests,
        problems
    )

    st.text(report)

    st.download_button(
        "Download Consultant Report",
        report,
        file_name="statx_consultant_report.txt"
    )
