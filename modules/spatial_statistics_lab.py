import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from scipy.spatial.distance import pdist, squareform

import libpysal
import esda
import spreg


# ---------------------------------------------------------
# SPATIAL WEIGHT MATRIX
# ---------------------------------------------------------

def build_spatial_weights(df, x_coord, y_coord):

    coords = list(zip(df[x_coord], df[y_coord]))

    w = libpysal.weights.KNN.from_array(coords, k=5)

    return w


# ---------------------------------------------------------
# MORAN'S I (SPATIAL AUTOCORRELATION)
# ---------------------------------------------------------

def moran_test(df, variable, weights):

    y = df[variable].values

    mi = esda.Moran(y, weights)

    return mi.I, mi.p_sim


# ---------------------------------------------------------
# GEARY'S C
# ---------------------------------------------------------

def geary_test(df, variable, weights):

    y = df[variable].values

    gc = esda.Geary(y, weights)

    return gc.C, gc.p_sim


# ---------------------------------------------------------
# SPATIAL LAG MODEL
# ---------------------------------------------------------

def spatial_lag_model(df, y_var, x_vars, weights):

    y = df[y_var].values.reshape(-1,1)

    X = df[x_vars].values

    model = spreg.ML_Lag(y, X, w=weights)

    return model


# ---------------------------------------------------------
# SPATIAL ERROR MODEL
# ---------------------------------------------------------

def spatial_error_model(df, y_var, x_vars, weights):

    y = df[y_var].values.reshape(-1,1)

    X = df[x_vars].values

    model = spreg.ML_Error(y, X, w=weights)

    return model


# ---------------------------------------------------------
# KRIGING / GAUSSIAN PROCESS
# ---------------------------------------------------------

def spatial_kriging(df, x_coord, y_coord, variable):

    coords = df[[x_coord, y_coord]].values

    y = df[variable].values

    kernel = RBF(length_scale=1)

    gp = GaussianProcessRegressor(kernel=kernel)

    gp.fit(coords, y)

    return gp


# ---------------------------------------------------------
# SPATIO-TEMPORAL TREND MODEL
# ---------------------------------------------------------

def spatiotemporal_regression(df, y_var, x_vars):

    import statsmodels.api as sm

    X = df[x_vars]

    X = sm.add_constant(X)

    model = sm.OLS(df[y_var], X).fit()

    return model


# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

def run(df):

    st.title("StatX Geostatistics & Spatio-Temporal Modeling Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Spatial Analysis",

        [

            "Spatial Autocorrelation (Moran's I)",
            "Geary's C Test",
            "Spatial Lag Model",
            "Spatial Error Model",
            "Spatial Kriging",
            "Spatio-Temporal Regression"

        ]
    )

# ---------------------------------------------------------
# MORAN
# ---------------------------------------------------------

    if analysis == "Spatial Autocorrelation (Moran's I)":

        var = st.selectbox("Variable", numeric)

        x = st.selectbox("Longitude/X Coordinate", numeric)

        y = st.selectbox("Latitude/Y Coordinate", numeric)

        if st.button("Run Moran Test"):

            w = build_spatial_weights(df, x, y)

            I, p = moran_test(df, var, w)

            st.metric("Moran's I", round(I,4))

            st.metric("p-value", round(p,5))


# ---------------------------------------------------------
# GEARY
# ---------------------------------------------------------

    elif analysis == "Geary's C Test":

        var = st.selectbox("Variable", numeric)

        x = st.selectbox("X Coordinate", numeric)

        y = st.selectbox("Y Coordinate", numeric)

        if st.button("Run Geary Test"):

            w = build_spatial_weights(df, x, y)

            C, p = geary_test(df, var, w)

            st.metric("Geary's C", round(C,4))

            st.metric("p-value", round(p,5))


# ---------------------------------------------------------
# SPATIAL LAG
# ---------------------------------------------------------

    elif analysis == "Spatial Lag Model":

        y_var = st.selectbox("Dependent Variable", numeric)

        x_vars = st.multiselect("Predictors", numeric)

        x = st.selectbox("X Coordinate", numeric)

        y = st.selectbox("Y Coordinate", numeric)

        if st.button("Run Spatial Lag Model"):

            w = build_spatial_weights(df, x, y)

            model = spatial_lag_model(df, y_var, x_vars, w)

            st.text(model.summary)


# ---------------------------------------------------------
# SPATIAL ERROR
# ---------------------------------------------------------

    elif analysis == "Spatial Error Model":

        y_var = st.selectbox("Dependent Variable", numeric)

        x_vars = st.multiselect("Predictors", numeric)

        x = st.selectbox("X Coordinate", numeric)

        y = st.selectbox("Y Coordinate", numeric)

        if st.button("Run Spatial Error Model"):

            w = build_spatial_weights(df, x, y)

            model = spatial_error_model(df, y_var, x_vars, w)

            st.text(model.summary)


# ---------------------------------------------------------
# KRIGING
# ---------------------------------------------------------

    elif analysis == "Spatial Kriging":

        var = st.selectbox("Variable", numeric)

        x = st.selectbox("X Coordinate", numeric)

        y = st.selectbox("Y Coordinate", numeric)

        if st.button("Run Kriging Model"):

            model = spatial_kriging(df, x, y, var)

            st.write("Gaussian Process Model fitted.")


# ---------------------------------------------------------
# SPATIO-TEMPORAL
# ---------------------------------------------------------

    elif analysis == "Spatio-Temporal Regression":

        y_var = st.selectbox("Dependent Variable", numeric)

        x_vars = st.multiselect("Predictors", numeric)

        if st.button("Run Spatio-Temporal Model"):

            model = spatiotemporal_regression(df, y_var, x_vars)

            st.text(model.summary())

    st.success("Spatial analysis completed.")
