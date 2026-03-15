import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, chisquare


# ---------------------------------------------------------
# CONTINGENCY TABLE
# ---------------------------------------------------------

def contingency_table(df, row_var, col_var):

    table = pd.crosstab(df[row_var], df[col_var])

    return table


# ---------------------------------------------------------
# CHI-SQUARE TEST OF INDEPENDENCE
# ---------------------------------------------------------

def chi_square_independence(table):

    chi2, p, dof, expected = chi2_contingency(table)

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    return chi2, p, dof, expected_df


# ---------------------------------------------------------
# CHI-SQUARE TEST OF HOMOGENEITY
# ---------------------------------------------------------

def chi_square_homogeneity(table):

    chi2, p, dof, expected = chi2_contingency(table)

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    return chi2, p, dof, expected_df


# ---------------------------------------------------------
# GOODNESS OF FIT
# ---------------------------------------------------------

def chi_square_gof(observed, expected):

    chi2, p = chisquare(f_obs=observed, f_exp=expected)

    return chi2, p


# ---------------------------------------------------------
# RESIDUALS
# ---------------------------------------------------------

def standardized_residuals(observed, expected):

    residuals = (observed - expected) / np.sqrt(expected)

    return residuals


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

def run(df):

    st.title("StatX Categorical & Contingency Analysis")

    if df is None:
        st.warning("Upload dataset first")
        return

    categorical = df.select_dtypes(include=["object","category"]).columns.tolist()

    analysis = st.selectbox(

        "Select Analysis",

        [

            "Chi-Square Test of Independence",
            "Chi-Square Test of Homogeneity",
            "Chi-Square Goodness of Fit"

        ]
    )


# ---------------------------------------------------------
# CHI-SQUARE INDEPENDENCE
# ---------------------------------------------------------

    if analysis == "Chi-Square Test of Independence":

        row_var = st.selectbox("Row Variable", categorical)

        col_var = st.selectbox("Column Variable", categorical)

        if st.button("Run Test"):

            table = contingency_table(df, row_var, col_var)

            chi2, p, dof, expected = chi_square_independence(table)

            st.subheader("Contingency Table")
            st.dataframe(table)

            st.subheader("Expected Frequencies")
            st.dataframe(expected)

            st.metric("Chi-Square Statistic", round(chi2,4))
            st.metric("p-value", round(p,5))
            st.metric("Degrees of Freedom", dof)

            residuals = standardized_residuals(table, expected)

            st.subheader("Standardized Residuals")

            st.dataframe(pd.DataFrame(residuals,
                                      index=table.index,
                                      columns=table.columns))

            fig, ax = plt.subplots()

            sns.heatmap(residuals,
                        annot=True,
                        center=0)

            ax.set_title("Residual Heatmap")

            st.pyplot(fig)

            if p < 0.05:

                st.success("Variables are dependent (association exists).")

            else:

                st.info("No evidence of association.")


# ---------------------------------------------------------
# HOMOGENEITY
# ---------------------------------------------------------

    elif analysis == "Chi-Square Test of Homogeneity":

        group = st.selectbox("Population/Group Variable", categorical)

        outcome = st.selectbox("Outcome Variable", categorical)

        if st.button("Run Homogeneity Test"):

            table = contingency_table(df, group, outcome)

            chi2, p, dof, expected = chi_square_homogeneity(table)

            st.subheader("Observed Frequencies")
            st.dataframe(table)

            st.subheader("Expected Frequencies")
            st.dataframe(expected)

            st.metric("Chi-Square", round(chi2,4))
            st.metric("p-value", round(p,5))

            if p < 0.05:

                st.success("Distributions differ across populations.")

            else:

                st.info("Distributions are homogeneous.")


# ---------------------------------------------------------
# GOODNESS OF FIT
# ---------------------------------------------------------

    elif analysis == "Chi-Square Goodness of Fit":

        var = st.selectbox("Categorical Variable", categorical)

        counts = df[var].value_counts()

        st.subheader("Observed Frequencies")

        st.write(counts)

        expected_type = st.radio(

            "Expected Distribution",

            ["Equal", "Custom"]

        )

        if expected_type == "Equal":

            expected = np.repeat(counts.sum()/len(counts), len(counts))

        else:

            expected = st.text_input(

                "Enter expected frequencies separated by commas"

            )

            if expected:

                expected = np.array(

                    [float(x) for x in expected.split(",")]

                )

        if st.button("Run Goodness-of-Fit Test"):

            chi2, p = chi_square_gof(counts.values, expected)

            st.metric("Chi-Square", round(chi2,4))
            st.metric("p-value", round(p,5))

            if p < 0.05:

                st.success("Observed distribution differs from expected.")

            else:

                st.info("Observed distribution fits expected pattern.")

    st.success("Categorical analysis completed.")
