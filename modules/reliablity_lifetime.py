import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter
from lifelines import CoxPHFitter


# ---------------------------------------------------------
# WEIBULL ANALYSIS
# ---------------------------------------------------------

def weibull_analysis(data):

    wf = WeibullFitter()

    wf.fit(data)

    shape = wf.rho_
    scale = wf.lambda_

    return wf, shape, scale


# ---------------------------------------------------------
# HAZARD FUNCTION
# ---------------------------------------------------------

def hazard_function(data):

    wf = WeibullFitter()

    wf.fit(data)

    hazard = wf.hazard_at_times(np.linspace(min(data), max(data), 100))

    return hazard


# ---------------------------------------------------------
# KAPLAN-MEIER SURVIVAL
# ---------------------------------------------------------

def kaplan_meier(times, events):

    km = KaplanMeierFitter()

    km.fit(times, event_observed=events)

    return km


# ---------------------------------------------------------
# ACCELERATED LIFE TEST MODEL
# ---------------------------------------------------------

def accelerated_life_model(df, time_col, covariates):

    model = CoxPHFitter()

    model.fit(df[[time_col] + covariates], duration_col=time_col)

    return model


# ---------------------------------------------------------
# FAILURE PREDICTION
# ---------------------------------------------------------

def failure_probability(shape, scale, t):

    prob = 1 - np.exp(-(t/scale)**shape)

    return prob


# ---------------------------------------------------------
# RELIABILITY FUNCTION
# ---------------------------------------------------------

def reliability_function(shape, scale, t):

    R = np.exp(-(t/scale)**shape)

    return R


# ---------------------------------------------------------
# INTERPRETATION ENGINE
# ---------------------------------------------------------

def interpret_weibull(shape):

    if shape < 1:
        return "Failure rate decreases over time (infant mortality failures)."

    elif shape == 1:
        return "Constant failure rate (random failures)."

    else:
        return "Failure rate increases over time (wear-out failures)."


# ---------------------------------------------------------
# STREAMLIT APPLICATION
# ---------------------------------------------------------

def run(df):

    st.title("StatX Reliability & Lifetime Analysis Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Reliability Analysis",

        [

            "Weibull Analysis",
            "Kaplan-Meier Survival Curve",
            "Hazard Function",
            "Accelerated Life Testing",
            "Failure Probability Prediction",
            "Reliability Function"
        ]

    )

# ---------------------------------------------------------
# WEIBULL ANALYSIS
# ---------------------------------------------------------

    if analysis == "Weibull Analysis":

        var = st.selectbox("Lifetime Variable", numeric)

        wf, shape, scale = weibull_analysis(df[var])

        st.metric("Shape Parameter (β)", shape)

        st.metric("Scale Parameter (η)", scale)

        st.write(interpret_weibull(shape))

        fig, ax = plt.subplots()

        wf.plot_survival_function(ax=ax)

        ax.set_title("Weibull Survival Curve")

        st.pyplot(fig)


# ---------------------------------------------------------
# KAPLAN MEIER
# ---------------------------------------------------------

    elif analysis == "Kaplan-Meier Survival Curve":

        time = st.selectbox("Time Variable", numeric)

        event = st.selectbox("Event Indicator (1=failure)", df.columns)

        km = kaplan_meier(df[time], df[event])

        fig, ax = plt.subplots()

        km.plot_survival_function(ax=ax)

        ax.set_title("Kaplan-Meier Survival Curve")

        st.pyplot(fig)

        st.write("Median Survival Time:", km.median_survival_time_)


# ---------------------------------------------------------
# HAZARD FUNCTION
# ---------------------------------------------------------

    elif analysis == "Hazard Function":

        var = st.selectbox("Lifetime Variable", numeric)

        wf, shape, scale = weibull_analysis(df[var])

        t = np.linspace(min(df[var]), max(df[var]), 100)

        hazard = (shape/scale)*(t/scale)**(shape-1)

        fig, ax = plt.subplots()

        ax.plot(t, hazard)

        ax.set_title("Hazard Function")

        st.pyplot(fig)

        st.write(interpret_weibull(shape))


# ---------------------------------------------------------
# ACCELERATED LIFE TEST
# ---------------------------------------------------------

    elif analysis == "Accelerated Life Testing":

        time = st.selectbox("Lifetime Variable", numeric)

        covariates = st.multiselect("Stress Factors", numeric)

        if len(covariates) > 0:

            model = accelerated_life_model(df, time, covariates)

            st.text(model.summary)


# ---------------------------------------------------------
# FAILURE PREDICTION
# ---------------------------------------------------------

    elif analysis == "Failure Probability Prediction":

        shape = st.number_input("Weibull Shape β", value=1.5)

        scale = st.number_input("Weibull Scale η", value=1000.0)

        t = st.number_input("Time", value=500.0)

        prob = failure_probability(shape, scale, t)

        st.metric("Probability of Failure", prob)


# ---------------------------------------------------------
# RELIABILITY FUNCTION
# ---------------------------------------------------------

    elif analysis == "Reliability Function":

        shape = st.number_input("Weibull Shape β", value=1.5)

        scale = st.number_input("Weibull Scale η", value=1000.0)

        t = st.number_input("Time", value=500.0)

        R = reliability_function(shape, scale, t)

        st.metric("Reliability R(t)", R)

    st.success("Reliability analysis completed.")
