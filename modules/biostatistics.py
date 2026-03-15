import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter
from lifelines import AalenAdditiveFitter
from lifelines import KaplanMeierFitter
from lifelines import FineAndGrayFitter


# ---------------------------------------------------------
# COX PROPORTIONAL HAZARDS MODEL
# ---------------------------------------------------------

def run_cox_model(df, duration, event, covariates):

    model = CoxPHFitter()

    model.fit(
        df[[duration, event] + covariates],
        duration_col=duration,
        event_col=event
    )

    return model


# ---------------------------------------------------------
# LOG-RANK TEST
# ---------------------------------------------------------

def run_logrank(df, duration, event, group):

    groups = df[group].unique()

    if len(groups) != 2:

        return None

    g1 = df[df[group]==groups[0]]

    g2 = df[df[group]==groups[1]]

    result = logrank_test(
        g1[duration],
        g2[duration],
        event_observed_A=g1[event],
        event_observed_B=g2[event]
    )

    return result


# ---------------------------------------------------------
# NELSON-AALEN ESTIMATOR
# ---------------------------------------------------------

def run_nelson_aalen(df, duration, event):

    naf = NelsonAalenFitter()

    naf.fit(df[duration], event_observed=df[event])

    return naf


# ---------------------------------------------------------
# COMPETING RISKS MODEL
# ---------------------------------------------------------

def run_competing_risks(df, duration, event, covariates):

    model = FineAndGrayFitter()

    model.fit(
        df,
        duration_col=duration,
        event_col=event
    )

    return model


# ---------------------------------------------------------
# SURVIVAL REGRESSION (AALEN ADDITIVE)
# ---------------------------------------------------------

def run_survival_regression(df, duration, event, covariates):

    model = AalenAdditiveFitter()

    model.fit(
        df[[duration, event] + covariates],
        duration_col=duration,
        event_col=event
    )

    return model


# ---------------------------------------------------------
# STREAMLIT APPLICATION
# ---------------------------------------------------------

def run(df):

    st.title("StatX Survival Analysis & Biostatistics Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Survival Analysis",

        [

            "Cox Proportional Hazards Model",
            "Log-Rank Test",
            "Nelson-Aalen Estimator",
            "Competing Risks Model",
            "Survival Regression"

        ]
    )

# ---------------------------------------------------------
# COX MODEL
# ---------------------------------------------------------

    if analysis == "Cox Proportional Hazards Model":

        duration = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Indicator (1=event,0=censored)", numeric)

        covariates = st.multiselect("Covariates", numeric)

        if st.button("Run Cox Model"):

            model = run_cox_model(df, duration, event, covariates)

            st.text(model.summary)

            st.write("""
Interpretation:
Hazard ratios >1 indicate higher risk,
<1 indicate protective effect.
""")


# ---------------------------------------------------------
# LOG-RANK TEST
# ---------------------------------------------------------

    elif analysis == "Log-Rank Test":

        duration = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Indicator", numeric)

        group = st.selectbox("Group Variable", df.columns)

        if st.button("Run Log-Rank Test"):

            result = run_logrank(df, duration, event, group)

            if result:

                st.metric("Test Statistic", round(result.test_statistic,4))

                st.metric("p-value", round(result.p_value,5))

                if result.p_value < 0.05:

                    st.success("Survival curves differ significantly")

                else:

                    st.info("No significant survival difference")


# ---------------------------------------------------------
# NELSON-AALEN
# ---------------------------------------------------------

    elif analysis == "Nelson-Aalen Estimator":

        duration = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Indicator", numeric)

        if st.button("Estimate Hazard Function"):

            naf = run_nelson_aalen(df, duration, event)

            fig, ax = plt.subplots()

            naf.plot(ax=ax)

            ax.set_title("Cumulative Hazard Function")

            st.pyplot(fig)


# ---------------------------------------------------------
# COMPETING RISKS
# ---------------------------------------------------------

    elif analysis == "Competing Risks Model":

        duration = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Type Variable", numeric)

        covariates = st.multiselect("Covariates", numeric)

        if st.button("Run Competing Risks Model"):

            model = run_competing_risks(df, duration, event, covariates)

            st.text(model.summary)


# ---------------------------------------------------------
# SURVIVAL REGRESSION
# ---------------------------------------------------------

    elif analysis == "Survival Regression":

        duration = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Indicator", numeric)

        covariates = st.multiselect("Covariates", numeric)

        if st.button("Run Survival Regression"):

            model = run_survival_regression(df, duration, event, covariates)

            st.text(model.summary)

    st.success("Survival analysis completed.")
