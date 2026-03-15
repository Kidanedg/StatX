import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import coint
from arch import arch_model


# ---------------------------------------------------------
# PANEL DATA MODELS
# ---------------------------------------------------------

def run_panel_model(df, entity, time, y, x_vars):

    df_panel = df.set_index([entity, time])

    Y = df_panel[y]
    X = df_panel[x_vars]

    model = PanelOLS(Y, X, entity_effects=True)

    results = model.fit()

    return results


# ---------------------------------------------------------
# ARDL MODEL
# ---------------------------------------------------------

def run_ardl(df, y, x, lags):

    model = ARDL(df[y], lags, df[x])

    results = model.fit()

    return results


# ---------------------------------------------------------
# COINTEGRATION TEST
# ---------------------------------------------------------

def run_cointegration(df, var1, var2):

    score, pvalue, crit = coint(df[var1], df[var2])

    return score, pvalue, crit


# ---------------------------------------------------------
# GARCH VOLATILITY MODEL
# ---------------------------------------------------------

def run_garch(series):

    model = arch_model(series, vol="Garch", p=1, q=1)

    result = model.fit(disp="off")

    return result


# ---------------------------------------------------------
# VALUE AT RISK
# ---------------------------------------------------------

def calculate_var(returns, alpha=0.05):

    var = np.percentile(returns, 100*alpha)

    return var


# ---------------------------------------------------------
# CONDITIONAL VALUE AT RISK
# ---------------------------------------------------------

def calculate_cvar(returns, alpha=0.05):

    var = calculate_var(returns, alpha)

    cvar = returns[returns <= var].mean()

    return cvar


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

def run(df):

    st.title("StatX Econometrics & Financial Statistics Laboratory")

    if df is None:
        st.warning("Upload dataset first")
        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Econometric Analysis",

        [

            "Panel Data Models",
            "ARDL Models",
            "Cointegration Test",
            "GARCH Volatility Model",
            "Financial Risk Metrics"

        ]
    )


# ---------------------------------------------------------
# PANEL DATA
# ---------------------------------------------------------

    if analysis == "Panel Data Models":

        entity = st.selectbox("Entity Variable (Cross-section)", df.columns)

        time = st.selectbox("Time Variable", df.columns)

        y = st.selectbox("Dependent Variable", numeric)

        x = st.multiselect("Independent Variables", numeric)

        if st.button("Run Panel Regression"):

            results = run_panel_model(df, entity, time, y, x)

            st.text(results.summary)

            st.write("""
Interpretation:
Panel models analyze both cross-sectional and time-series variation.
Entity effects control for unobserved heterogeneity.
""")


# ---------------------------------------------------------
# ARDL
# ---------------------------------------------------------

    elif analysis == "ARDL Models":

        y = st.selectbox("Dependent Variable", numeric)

        x = st.multiselect("Independent Variables", numeric)

        lag = st.slider("Lag Order", 1, 5, 1)

        if st.button("Run ARDL"):

            res = run_ardl(df, y, x, lag)

            st.text(res.summary())

            st.write("""
Interpretation:
ARDL models estimate short-run and long-run relationships
between variables with different integration orders.
""")


# ---------------------------------------------------------
# COINTEGRATION
# ---------------------------------------------------------

    elif analysis == "Cointegration Test":

        var1 = st.selectbox("Variable 1", numeric)

        var2 = st.selectbox("Variable 2", numeric)

        if st.button("Run Cointegration Test"):

            score, pvalue, crit = run_cointegration(df, var1, var2)

            st.write("Test Statistic:", score)
            st.write("p-value:", pvalue)
            st.write("Critical Values:", crit)

            if pvalue < 0.05:
                st.success("Variables are cointegrated.")
            else:
                st.warning("No cointegration detected.")

            st.write("""
Interpretation:
Cointegration indicates a long-run equilibrium
relationship between non-stationary variables.
""")


# ---------------------------------------------------------
# GARCH
# ---------------------------------------------------------

    elif analysis == "GARCH Volatility Model":

        var = st.selectbox("Financial Return Series", numeric)

        if st.button("Run GARCH Model"):

            res = run_garch(df[var].dropna())

            st.text(res.summary())

            volatility = res.conditional_volatility

            fig, ax = plt.subplots()

            ax.plot(volatility)

            ax.set_title("Conditional Volatility")

            st.pyplot(fig)

            st.write("""
Interpretation:
GARCH models capture volatility clustering
in financial time series.
""")


# ---------------------------------------------------------
# FINANCIAL RISK
# ---------------------------------------------------------

    elif analysis == "Financial Risk Metrics":

        var = st.selectbox("Return Series", numeric)

        alpha = st.slider("Risk Level", 0.01, 0.10, 0.05)

        returns = df[var].dropna()

        if st.button("Calculate Risk Metrics"):

            var_val = calculate_var(returns, alpha)

            cvar_val = calculate_cvar(returns, alpha)

            st.metric("Value at Risk (VaR)", round(var_val,4))

            st.metric("Conditional VaR (CVaR)", round(cvar_val,4))

            st.write("""
Interpretation:
VaR estimates the maximum expected loss
at a given confidence level.
CVaR measures the expected loss beyond VaR.
""")

    st.success("Econometric analysis completed.")
