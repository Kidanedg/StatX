import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# ---------------------------------------------------------
# CONTROL LIMIT CALCULATIONS
# ---------------------------------------------------------

def control_limits(series):

    mean = series.mean()
    std = series.std()

    ucl = mean + 3*std
    lcl = mean - 3*std

    return mean, ucl, lcl


# ---------------------------------------------------------
# XBAR CHART
# ---------------------------------------------------------

def xbar_chart(df, column):

    data = df[column]

    mean, ucl, lcl = control_limits(data)

    fig, ax = plt.subplots()

    ax.plot(data, marker='o')

    ax.axhline(mean)
    ax.axhline(ucl)
    ax.axhline(lcl)

    ax.set_title("X-bar Control Chart")

    return fig, mean, ucl, lcl


# ---------------------------------------------------------
# RANGE CHART
# ---------------------------------------------------------

def r_chart(df, column):

    data = df[column].rolling(2).apply(lambda x: max(x)-min(x))

    mean, ucl, lcl = control_limits(data.dropna())

    fig, ax = plt.subplots()

    ax.plot(data)

    ax.axhline(mean)
    ax.axhline(ucl)
    ax.axhline(lcl)

    ax.set_title("Range Chart")

    return fig


# ---------------------------------------------------------
# P CHART
# ---------------------------------------------------------

def p_chart(defects, n):

    p_bar = defects.mean()/n

    ucl = p_bar + 3*np.sqrt(p_bar*(1-p_bar)/n)
    lcl = p_bar - 3*np.sqrt(p_bar*(1-p_bar)/n)

    return p_bar, ucl, lcl


# ---------------------------------------------------------
# PROCESS CAPABILITY
# ---------------------------------------------------------

def process_capability(data, usl, lsl):

    mean = data.mean()
    std = data.std()

    cp = (usl - lsl)/(6*std)

    cpk = min(
        (usl - mean)/(3*std),
        (mean - lsl)/(3*std)
    )

    return cp, cpk


# ---------------------------------------------------------
# SIX SIGMA LEVEL
# ---------------------------------------------------------

def sigma_level(defects, opportunities):

    dpm = defects/opportunities * 1e6

    sigma = stats.norm.ppf(1 - dpm/1e6) + 1.5

    return dpm, sigma


# ---------------------------------------------------------
# RELIABILITY ANALYSIS
# ---------------------------------------------------------

def reliability_analysis(times):

    mean_life = np.mean(times)

    failure_rate = 1/mean_life

    return mean_life, failure_rate


# ---------------------------------------------------------
# DESIGN OF EXPERIMENTS (2^k)
# ---------------------------------------------------------

def factorial_effects(df, factors, response):

    effects = {}

    for f in factors:

        group = df.groupby(f)[response].mean()

        effects[f] = group.iloc[1] - group.iloc[0]

    return effects


# ---------------------------------------------------------
# PARETO ANALYSIS
# ---------------------------------------------------------

def pareto_chart(df, column):

    counts = df[column].value_counts()

    fig, ax = plt.subplots()

    counts.plot(kind='bar', ax=ax)

    ax.set_title("Pareto Chart")

    return fig


# ---------------------------------------------------------
# CAUSE-EFFECT DIAGRAM (TEXT)
# ---------------------------------------------------------

def cause_effect():

    text = """
Possible causes of process variation:

Man
Machine
Material
Method
Measurement
Environment
"""

    return text


# ---------------------------------------------------------
# INTERPRETATION ENGINE
# ---------------------------------------------------------

def interpret_cp(cp, cpk):

    if cp >= 1.33 and cpk >= 1.33:

        return "Process is capable and centered."

    elif cp >= 1 and cpk >= 1:

        return "Process is marginally capable."

    else:

        return "Process capability is poor. Improvement required."


# ---------------------------------------------------------
# MAIN STREAMLIT APP
# ---------------------------------------------------------

def run(df):

    st.title("StatX Quality Control & Statistical Engineering Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    option = st.selectbox(

        "Select Analysis",

        [

            "X-bar Control Chart",
            "Range Chart",
            "p Control Chart",
            "Process Capability",
            "Six Sigma Analysis",
            "Reliability Analysis",
            "Design of Experiments",
            "Pareto Analysis",
            "Cause-Effect Analysis"
        ]
    )

# ---------------------------------------------------------
# XBAR
# ---------------------------------------------------------

    if option == "X-bar Control Chart":

        var = st.selectbox("Variable", numeric)

        fig, mean, ucl, lcl = xbar_chart(df, var)

        st.pyplot(fig)

        st.write("Mean:", mean)
        st.write("UCL:", ucl)
        st.write("LCL:", lcl)

# ---------------------------------------------------------
# RANGE
# ---------------------------------------------------------

    elif option == "Range Chart":

        var = st.selectbox("Variable", numeric)

        fig = r_chart(df, var)

        st.pyplot(fig)

# ---------------------------------------------------------
# P CHART
# ---------------------------------------------------------

    elif option == "p Control Chart":

        defects = st.number_input("Number of Defects")

        n = st.number_input("Sample Size")

        p_bar, ucl, lcl = p_chart(defects, n)

        st.write("Average defect rate:", p_bar)
        st.write("UCL:", ucl)
        st.write("LCL:", lcl)

# ---------------------------------------------------------
# PROCESS CAPABILITY
# ---------------------------------------------------------

    elif option == "Process Capability":

        var = st.selectbox("Variable", numeric)

        usl = st.number_input("Upper Spec Limit")

        lsl = st.number_input("Lower Spec Limit")

        cp, cpk = process_capability(df[var], usl, lsl)

        st.metric("Cp", cp)
        st.metric("Cpk", cpk)

        st.write(interpret_cp(cp, cpk))

# ---------------------------------------------------------
# SIX SIGMA
# ---------------------------------------------------------

    elif option == "Six Sigma Analysis":

        defects = st.number_input("Defects")

        opportunities = st.number_input("Opportunities")

        dpm, sigma = sigma_level(defects, opportunities)

        st.metric("Defects per million", dpm)

        st.metric("Sigma level", sigma)

# ---------------------------------------------------------
# RELIABILITY
# ---------------------------------------------------------

    elif option == "Reliability Analysis":

        var = st.selectbox("Failure Times", numeric)

        mean_life, rate = reliability_analysis(df[var])

        st.metric("Mean Life", mean_life)

        st.metric("Failure Rate", rate)

# ---------------------------------------------------------
# DOE
# ---------------------------------------------------------

    elif option == "Design of Experiments":

        response = st.selectbox("Response Variable", numeric)

        factors = st.multiselect("Factors", df.columns)

        effects = factorial_effects(df, factors, response)

        st.write(effects)

# ---------------------------------------------------------
# PARETO
# ---------------------------------------------------------

    elif option == "Pareto Analysis":

        var = st.selectbox("Defect Type", df.columns)

        fig = pareto_chart(df, var)

        st.pyplot(fig)

# ---------------------------------------------------------
# CAUSE EFFECT
# ---------------------------------------------------------

    elif option == "Cause-Effect Analysis":

        st.text(cause_effect())

    st.success("Quality control analysis completed")
