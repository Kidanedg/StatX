import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------------------------------
# STYLING
# -----------------------------------------------------

def plot_series(ts):

    fig, ax = plt.subplots(figsize=(10,4))

    ts.plot(ax=ax,color="navy")

    ax.set_title("Time Series Plot")

    st.pyplot(fig)


# -----------------------------------------------------
# STATIONARITY TESTS
# -----------------------------------------------------

def adf_test(ts):

    result = adfuller(ts.dropna())

    return {
        "ADF Statistic": result[0],
        "p-value": result[1]
    }


def kpss_test(ts):

    result = kpss(ts.dropna())

    return {
        "KPSS Statistic": result[0],
        "p-value": result[1]
    }


# -----------------------------------------------------
# INTERPRETATION ENGINE
# -----------------------------------------------------

def interpret_stationarity(adf_p, kpss_p):

    text = ""

    if adf_p < 0.05:
        text += "ADF test suggests the series is stationary.\n"
    else:
        text += "ADF test suggests the series is non-stationary.\n"

    if kpss_p > 0.05:
        text += "KPSS confirms stationarity.\n"
    else:
        text += "KPSS suggests non-stationarity."

    return text


# -----------------------------------------------------
# ACF PACF
# -----------------------------------------------------

def acf_pacf_plots(ts):

    fig, ax = plt.subplots()

    plot_acf(ts, ax=ax)

    st.pyplot(fig)

    fig2, ax2 = plt.subplots()

    plot_pacf(ts, ax=ax2)

    st.pyplot(fig2)


# -----------------------------------------------------
# FORECAST EVALUATION
# -----------------------------------------------------

def forecast_metrics(true, pred):

    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)

    return rmse, mae


# -----------------------------------------------------
# ARIMA MODEL
# -----------------------------------------------------

def run_arima(ts, p, d, q):

    model = ARIMA(ts, order=(p,d,q)).fit()

    return model


# -----------------------------------------------------
# SARIMA MODEL
# -----------------------------------------------------

def run_sarima(ts, p,d,q,P,D,Q,s):

    model = SARIMAX(
        ts,
        order=(p,d,q),
        seasonal_order=(P,D,Q,s)
    ).fit()

    return model


# -----------------------------------------------------
# EXPONENTIAL SMOOTHING
# -----------------------------------------------------

def exp_smoothing(ts):

    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal=None
    ).fit()

    return model


# -----------------------------------------------------
# HOLT WINTERS
# -----------------------------------------------------

def holt_winters(ts, seasonal_periods):

    model = ExponentialSmoothing(
        ts,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods
    ).fit()

    return model


# -----------------------------------------------------
# FORECAST PLOT
# -----------------------------------------------------

def forecast_plot(ts, forecast):

    fig, ax = plt.subplots(figsize=(10,4))

    ts.plot(ax=ax,label="Observed")

    forecast.plot(ax=ax,label="Forecast")

    plt.legend()

    st.pyplot(fig)


# -----------------------------------------------------
# AUTOMATIC MODEL SELECTION
# -----------------------------------------------------

def auto_arima(ts):

    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(3):
        for d in range(2):
            for q in range(3):

                try:

                    model = ARIMA(ts,order=(p,d,q)).fit()

                    if model.aic < best_aic:

                        best_aic = model.aic
                        best_order = (p,d,q)
                        best_model = model

                except:
                    continue

    return best_model, best_order


# -----------------------------------------------------
# MAIN MODULE
# -----------------------------------------------------

def run(df):

    st.title("StatX Time Series & Forecasting Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    time_col = st.selectbox("Time Index", df.columns)

    value_col = st.selectbox("Series Variable", numeric)

    df = df.sort_values(time_col)

    ts = df[value_col]

    plot_series(ts)

    analysis = st.selectbox(

        "Select Analysis",

        [

            "Stationarity Tests",
            "ACF / PACF Diagnostics",
            "ARIMA",
            "SARIMA",
            "Exponential Smoothing",
            "Holt Winters",
            "Automatic Model Selection"
        ]

    )

# -----------------------------------------------------
# STATIONARITY
# -----------------------------------------------------

    if analysis == "Stationarity Tests":

        adf = adf_test(ts)
        kps = kpss_test(ts)

        st.write("ADF Test", adf)
        st.write("KPSS Test", kps)

        st.write(

            interpret_stationarity(
                adf["p-value"],
                kps["p-value"]
            )

        )

# -----------------------------------------------------
# ACF PACF
# -----------------------------------------------------

    elif analysis == "ACF / PACF Diagnostics":

        acf_pacf_plots(ts)

# -----------------------------------------------------
# ARIMA
# -----------------------------------------------------

    elif analysis == "ARIMA":

        p = st.slider("p",0,5,1)
        d = st.slider("d",0,2,1)
        q = st.slider("q",0,5,1)

        model = run_arima(ts,p,d,q)

        st.text(model.summary())

        forecast = model.forecast(10)

        forecast_plot(ts,forecast)

# -----------------------------------------------------
# SARIMA
# -----------------------------------------------------

    elif analysis == "SARIMA":

        p = st.slider("p",0,3,1)
        d = st.slider("d",0,2,1)
        q = st.slider("q",0,3,1)

        P = st.slider("P",0,2,1)
        D = st.slider("D",0,1,0)
        Q = st.slider("Q",0,2,1)

        s = st.number_input("Season length",12)

        model = run_sarima(ts,p,d,q,P,D,Q,s)

        st.text(model.summary())

        forecast = model.forecast(10)

        forecast_plot(ts,forecast)

# -----------------------------------------------------
# EXPONENTIAL SMOOTHING
# -----------------------------------------------------

    elif analysis == "Exponential Smoothing":

        model = exp_smoothing(ts)

        st.text(model.summary())

        forecast = model.forecast(10)

        forecast_plot(ts,forecast)

# -----------------------------------------------------
# HOLT WINTERS
# -----------------------------------------------------

    elif analysis == "Holt Winters":

        s = st.number_input("Season length",12)

        model = holt_winters(ts,s)

        st.text(model.summary())

        forecast = model.forecast(10)

        forecast_plot(ts,forecast)

# -----------------------------------------------------
# AUTO MODEL
# -----------------------------------------------------

    elif analysis == "Automatic Model Selection":

        model, order = auto_arima(ts)

        st.write("Best ARIMA Order:", order)

        st.text(model.summary())

        forecast = model.forecast(10)

        forecast_plot(ts,forecast)

    st.success("Time series analysis completed")
