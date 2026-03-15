import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pymc as pm
import arviz as az

from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# ---------------------------------------------------------
# PRIOR BUILDER
# ---------------------------------------------------------

def build_prior(prior_type, mean=0, sd=1):

    if prior_type == "Normal":

        return pm.Normal("prior", mu=mean, sigma=sd)

    elif prior_type == "Uniform":

        return pm.Uniform("prior", lower=mean, upper=sd)

    elif prior_type == "Exponential":

        return pm.Exponential("prior", lam=1/sd)


# ---------------------------------------------------------
# BAYESIAN INFERENCE
# ---------------------------------------------------------

def bayesian_mean_inference(data):

    with pm.Model():

        mu = pm.Normal("mu", mu=np.mean(data), sigma=np.std(data))

        sigma = pm.HalfNormal("sigma", sigma=10)

        likelihood = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=data
        )

        trace = pm.sample(2000, tune=1000)

    return trace


# ---------------------------------------------------------
# BAYESIAN REGRESSION
# ---------------------------------------------------------

def bayesian_regression(df, x_vars, y_var):

    X = df[x_vars].values
    Y = df[y_var].values

    with pm.Model():

        beta = pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])

        intercept = pm.Normal("intercept", mu=0, sigma=10)

        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = intercept + pm.math.dot(X, beta)

        likelihood = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=Y
        )

        trace = pm.sample(2000, tune=1000)

    return trace


# ---------------------------------------------------------
# MCMC SIMULATION
# ---------------------------------------------------------

def run_mcmc(data):

    trace = bayesian_mean_inference(data)

    return trace


# ---------------------------------------------------------
# POSTERIOR VISUALIZATION
# ---------------------------------------------------------

def posterior_plots(trace):

    fig = az.plot_posterior(trace)

    return fig


# ---------------------------------------------------------
# TRACE PLOT
# ---------------------------------------------------------

def trace_plots(trace):

    fig = az.plot_trace(trace)

    return fig


# ---------------------------------------------------------
# BAYESIAN NETWORK
# ---------------------------------------------------------

def build_bayesian_network(df, edges):

    model = BayesianNetwork(edges)

    model.fit(df, estimator=MaximumLikelihoodEstimator)

    inference = VariableElimination(model)

    return model, inference


# ---------------------------------------------------------
# STREAMLIT APPLICATION
# ---------------------------------------------------------

def run(df):

    st.title("StatX Bayesian Statistics Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Bayesian Analysis",

        [

            "Bayesian Inference",
            "Bayesian Regression",
            "MCMC Simulation",
            "Posterior Distribution Visualization",
            "Bayesian Networks"

        ]
    )


# ---------------------------------------------------------
# BAYESIAN INFERENCE
# ---------------------------------------------------------

    if analysis == "Bayesian Inference":

        var = st.selectbox("Variable", numeric)

        data = df[var].dropna()

        if st.button("Run Bayesian Inference"):

            trace = bayesian_mean_inference(data)

            st.subheader("Posterior Summary")

            st.write(az.summary(trace))

            st.subheader("Posterior Distribution")

            fig = az.plot_posterior(trace)

            st.pyplot(fig)


# ---------------------------------------------------------
# BAYESIAN REGRESSION
# ---------------------------------------------------------

    elif analysis == "Bayesian Regression":

        y = st.selectbox("Dependent Variable", numeric)

        x = st.multiselect("Predictors", numeric)

        if len(x) > 0 and st.button("Run Bayesian Regression"):

            trace = bayesian_regression(df, x, y)

            st.write(az.summary(trace))

            fig = az.plot_trace(trace)

            st.pyplot(fig)


# ---------------------------------------------------------
# MCMC SIMULATION
# ---------------------------------------------------------

    elif analysis == "MCMC Simulation":

        var = st.selectbox("Variable", numeric)

        data = df[var].dropna()

        if st.button("Run MCMC"):

            trace = run_mcmc(data)

            st.subheader("Trace Plot")

            fig = az.plot_trace(trace)

            st.pyplot(fig)

            st.subheader("Posterior Density")

            fig2 = az.plot_posterior(trace)

            st.pyplot(fig2)


# ---------------------------------------------------------
# POSTERIOR VISUALIZATION
# ---------------------------------------------------------

    elif analysis == "Posterior Distribution Visualization":

        var = st.selectbox("Variable", numeric)

        data = df[var].dropna()

        trace = bayesian_mean_inference(data)

        fig = posterior_plots(trace)

        st.pyplot(fig)


# ---------------------------------------------------------
# BAYESIAN NETWORKS
# ---------------------------------------------------------

    elif analysis == "Bayesian Networks":

        st.write("Define causal relationships between variables")

        cols = df.columns.tolist()

        parent = st.selectbox("Parent Variable", cols)

        child = st.selectbox("Child Variable", cols)

        if st.button("Build Network"):

            edges = [(parent, child)]

            encoded_df = df.copy()

            for c in encoded_df.columns:

                if encoded_df[c].dtype == "object":

                    le = LabelEncoder()

                    encoded_df[c] = le.fit_transform(encoded_df[c])

            model, inference = build_bayesian_network(encoded_df, edges)

            st.write("Bayesian Network constructed")

            st.write("Edges:", model.edges())

    st.success("Bayesian analysis completed.")
