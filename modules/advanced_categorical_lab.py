import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import StratifiedTable
import statsmodels.api as sm

# ---------------------------------------------------------
# CRAMER'S V
# ---------------------------------------------------------

def cramers_v(table):

    chi2 = sm.stats.Table(table).test_nominal_association().statistic

    n = table.sum().sum()

    r, c = table.shape

    return np.sqrt(chi2 / (n * (min(r-1, c-1))))

# ---------------------------------------------------------
# PHI COEFFICIENT
# ---------------------------------------------------------

def phi_coefficient(table):

    chi2 = sm.stats.Table(table).test_nominal_association().statistic

    n = table.sum().sum()

    return np.sqrt(chi2/n)

# ---------------------------------------------------------
# FISHER EXACT TEST
# ---------------------------------------------------------

def fisher_test(table):

    oddsratio, p = fisher_exact(table)

    return oddsratio, p

# ---------------------------------------------------------
# MCNEMAR TEST
# ---------------------------------------------------------

def run_mcnemar(table):

    result = mcnemar(table, exact=True)

    return result.statistic, result.pvalue

# ---------------------------------------------------------
# COCHRAN–MANTEL–HAENSZEL TEST
# ---------------------------------------------------------

def run_cmh(tables):

    strat_table = StratifiedTable(tables)

    result = strat_table.test_null_odds()

    return result.statistic, result.pvalue

# ---------------------------------------------------------
# LOG-LINEAR MODEL
# ---------------------------------------------------------

def run_loglinear(df, formula):

    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Poisson()
    )

    results = model.fit()

    return results

# ---------------------------------------------------------
# STREAMLIT APPLICATION
# ---------------------------------------------------------

def run(df):

    st.title("StatX Advanced Categorical Data Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    categorical = df.select_dtypes(
        include=["object","category"]
    ).columns.tolist()

    analysis = st.selectbox(

        "Select Advanced Categorical Analysis",

        [

            "Fisher Exact Test",
            "McNemar Test",
            "Cochran–Mantel–Haenszel Test",
            "Log-linear Models",
            "Association Measures (Cramer's V / Phi)"

        ]
    )

# ---------------------------------------------------------
# FISHER EXACT TEST
# ---------------------------------------------------------

    if analysis == "Fisher Exact Test":

        var1 = st.selectbox("Variable 1", categorical)

        var2 = st.selectbox("Variable 2", categorical)

        if st.button("Run Fisher Test"):

            table = pd.crosstab(df[var1], df[var2])

            if table.shape != (2,2):

                st.error("Fisher test requires a 2x2 table")

                return

            oddsratio, p = fisher_test(table)

            st.subheader("Contingency Table")

            st.dataframe(table)

            st.metric("Odds Ratio", round(oddsratio,4))

            st.metric("p-value", round(p,5))

            if p < 0.05:

                st.success("Significant association detected")

            else:

                st.info("No significant association")

# ---------------------------------------------------------
# MCNEMAR TEST
# ---------------------------------------------------------

    elif analysis == "McNemar Test":

        var1 = st.selectbox("Before Variable", categorical)

        var2 = st.selectbox("After Variable", categorical)

        if st.button("Run McNemar Test"):

            table = pd.crosstab(df[var1], df[var2])

            stat, p = run_mcnemar(table)

            st.subheader("Paired Contingency Table")

            st.dataframe(table)

            st.metric("McNemar Statistic", round(stat,4))

            st.metric("p-value", round(p,5))

# ---------------------------------------------------------
# CMH TEST
# ---------------------------------------------------------

    elif analysis == "Cochran–Mantel–Haenszel Test":

        var1 = st.selectbox("Exposure", categorical)

        var2 = st.selectbox("Outcome", categorical)

        strat = st.selectbox("Stratification Variable", categorical)

        if st.button("Run CMH Test"):

            tables = []

            for level in df[strat].unique():

                sub = df[df[strat]==level]

                table = pd.crosstab(sub[var1], sub[var2])

                if table.shape == (2,2):

                    tables.append(table.values)

            stat, p = run_cmh(tables)

            st.metric("CMH Statistic", round(stat,4))

            st.metric("p-value", round(p,5))

# ---------------------------------------------------------
# LOG-LINEAR MODEL
# ---------------------------------------------------------

    elif analysis == "Log-linear Models":

        st.write("Example formula: count ~ A + B + A:B")

        formula = st.text_input("Model Formula")

        if st.button("Run Log-linear Model"):

            results = run_loglinear(df, formula)

            st.text(results.summary())

# ---------------------------------------------------------
# ASSOCIATION MEASURES
# ---------------------------------------------------------

    elif analysis == "Association Measures (Cramer's V / Phi)":

        var1 = st.selectbox("Variable 1", categorical)

        var2 = st.selectbox("Variable 2", categorical)

        if st.button("Compute Association"):

            table = pd.crosstab(df[var1], df[var2])

            st.dataframe(table)

            if table.shape == (2,2):

                phi = phi_coefficient(table)

                st.metric("Phi Coefficient", round(phi,4))

            cv = cramers_v(table)

            st.metric("Cramer's V", round(cv,4))

    st.success("Advanced categorical analysis completed.")
