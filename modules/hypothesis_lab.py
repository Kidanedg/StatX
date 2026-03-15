import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.weightstats as smstats
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# INTERPRETATION ENGINE
# -----------------------------------------------------

def interpret_result(pvalue, alpha=0.05):

    if pvalue < alpha:
        return f"""
        **Decision:** Reject the null hypothesis (H₀)

        **Interpretation:**  
        The p-value ({pvalue:.5f}) is less than the significance level (α = {alpha}).  
        Therefore, there is **statistically significant evidence** against H₀.
        """

    else:
        return f"""
        **Decision:** Fail to reject the null hypothesis (H₀)

        **Interpretation:**  
        The p-value ({pvalue:.5f}) is greater than the significance level (α = {alpha}).  
        Therefore, there is **insufficient statistical evidence** to reject H₀.
        """

# -----------------------------------------------------
# CONFIDENCE INTERVAL
# -----------------------------------------------------

def confidence_interval(data, confidence):

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    ci = stats.t.interval(
        confidence,
        n-1,
        loc=mean,
        scale=std/np.sqrt(n)
    )

    return mean, ci

# -----------------------------------------------------
# ONE SAMPLE T TEST
# -----------------------------------------------------

def one_sample_ttest(data, mu):

    tstat, pvalue = stats.ttest_1samp(data, mu)

    return tstat, pvalue


# -----------------------------------------------------
# TWO SAMPLE T TEST
# -----------------------------------------------------

def two_sample_ttest(data1, data2):

    tstat, pvalue = stats.ttest_ind(data1, data2)

    return tstat, pvalue


# -----------------------------------------------------
# PAIRED T TEST
# -----------------------------------------------------

def paired_ttest(data1, data2):

    tstat, pvalue = stats.ttest_rel(data1, data2)

    return tstat, pvalue


# -----------------------------------------------------
# Z TEST
# -----------------------------------------------------

def z_test(data, mu):

    zstat, pvalue = smstats.ztest(data, value=mu)

    return zstat, pvalue


# -----------------------------------------------------
# PROPORTION TEST
# -----------------------------------------------------

def proportion_test(success, n, p0):

    zstat = (success/n - p0)/np.sqrt(p0*(1-p0)/n)

    pvalue = 2*(1-stats.norm.cdf(abs(zstat)))

    return zstat, pvalue


# -----------------------------------------------------
# CHI SQUARE TEST
# -----------------------------------------------------

def chi_square_test(table):

    chi2, pvalue, dof, expected = stats.chi2_contingency(table)

    return chi2, pvalue, dof, expected


# -----------------------------------------------------
# F TEST (Variance Test)
# -----------------------------------------------------

def variance_test(data1, data2):

    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)

    f = var1/var2

    dfn = len(data1)-1
    dfd = len(data2)-1

    pvalue = 1 - stats.f.cdf(f, dfn, dfd)

    return f, pvalue


# -----------------------------------------------------
# ANOVA
# -----------------------------------------------------

def one_way_anova(groups):

    fstat, pvalue = stats.f_oneway(*groups)

    return fstat, pvalue


# -----------------------------------------------------
# NONPARAMETRIC TESTS
# -----------------------------------------------------

def mann_whitney(data1, data2):

    stat, pvalue = stats.mannwhitneyu(data1, data2)

    return stat, pvalue


def wilcoxon_test(data1, data2):

    stat, pvalue = stats.wilcoxon(data1, data2)

    return stat, pvalue


def kruskal_test(groups):

    stat, pvalue = stats.kruskal(*groups)

    return stat, pvalue


# -----------------------------------------------------
# CORRELATION TEST
# -----------------------------------------------------

def correlation_test(x, y):

    r, pvalue = stats.pearsonr(x, y)

    return r, pvalue


# -----------------------------------------------------
# MAIN STREAMLIT MODULE
# -----------------------------------------------------

def run(df):

    st.title("StatX Hypothesis Testing & Estimation Lab")

    if df is None:
        st.warning("Upload dataset first")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    test_type = st.selectbox(
        "Select Statistical Procedure",
        [
            "Confidence Interval",
            "One Sample t-test",
            "Two Sample t-test",
            "Paired t-test",
            "Z Test",
            "Proportion Test",
            "Chi-Square Test",
            "Variance F Test",
            "One Way ANOVA",
            "Mann Whitney Test",
            "Wilcoxon Test",
            "Kruskal Wallis Test",
            "Correlation Test"
        ]
    )

    alpha = st.sidebar.slider("Significance Level",0.01,0.10,0.05)

# -----------------------------------------------------
# CONFIDENCE INTERVAL
# -----------------------------------------------------

    if test_type == "Confidence Interval":

        var = st.selectbox("Variable", numeric_cols)
        conf = st.slider("Confidence Level",0.80,0.99,0.95)

        mean, ci = confidence_interval(df[var].dropna(), conf)

        st.write("Sample Mean:", mean)
        st.write("Confidence Interval:", ci)

# -----------------------------------------------------
# ONE SAMPLE T TEST
# -----------------------------------------------------

    elif test_type == "One Sample t-test":

        var = st.selectbox("Variable", numeric_cols)
        mu = st.number_input("Hypothesized Mean")

        tstat, pvalue = one_sample_ttest(df[var].dropna(), mu)

        st.write("t Statistic:", tstat)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------
# TWO SAMPLE T TEST
# -----------------------------------------------------

    elif test_type == "Two Sample t-test":

        var1 = st.selectbox("Sample 1", numeric_cols)
        var2 = st.selectbox("Sample 2", numeric_cols)

        tstat, pvalue = two_sample_ttest(df[var1].dropna(), df[var2].dropna())

        st.write("t Statistic:", tstat)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------
# PAIRED T TEST
# -----------------------------------------------------

    elif test_type == "Paired t-test":

        var1 = st.selectbox("Before", numeric_cols)
        var2 = st.selectbox("After", numeric_cols)

        tstat, pvalue = paired_ttest(df[var1], df[var2])

        st.write("t Statistic:", tstat)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------
# Z TEST
# -----------------------------------------------------

    elif test_type == "Z Test":

        var = st.selectbox("Variable", numeric_cols)
        mu = st.number_input("Hypothesized Mean")

        zstat, pvalue = z_test(df[var], mu)

        st.write("Z Statistic:", zstat)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------
# CHI SQUARE
# -----------------------------------------------------

    elif test_type == "Chi-Square Test":

        cat1 = st.selectbox("Row Variable", df.columns)
        cat2 = st.selectbox("Column Variable", df.columns)

        table = pd.crosstab(df[cat1], df[cat2])

        chi2, pvalue, dof, expected = chi_square_test(table)

        st.write("Contingency Table")
        st.write(table)

        st.write("Chi-square:", chi2)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------
# CORRELATION
# -----------------------------------------------------

    elif test_type == "Correlation Test":

        x = st.selectbox("Variable X", numeric_cols)
        y = st.selectbox("Variable Y", numeric_cols)

        r, pvalue = correlation_test(df[x], df[y])

        st.write("Correlation Coefficient:", r)
        st.write("p-value:", pvalue)

        st.write(interpret_result(pvalue, alpha))

# -----------------------------------------------------

    st.success("Analysis completed successfully")
