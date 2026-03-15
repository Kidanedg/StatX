import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from pandas.plotting import scatter_matrix


def run(df):

    st.title("📊 StatX Graphs & Visualization Laboratory")

    if df is None:
        st.warning("Upload dataset first.")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.divider()

# -----------------------------------------
# GRAPH SETTINGS
# -----------------------------------------

    st.sidebar.header("Graph Style Settings")

    color_theme = st.sidebar.selectbox(
        "Color Theme",
        [
            "viridis","plasma","coolwarm","magma",
            "Set1","Set2","tab10","Dark2"
        ]
    )

    graph_engine = st.sidebar.selectbox(
        "Visualization Engine",
        ["Matplotlib / Seaborn", "Plotly Interactive"]
    )

# -----------------------------------------
# GRAPH TYPE SELECTION
# -----------------------------------------

    st.header("Select Graph Type")

    graph_type = st.selectbox(
        "Graph Type",
        [
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Density Plot",
            "Scatter Plot",
            "Scatter Matrix",
            "Line Plot",
            "Bar Chart",
            "Pie Chart",
            "Area Chart",
            "Correlation Heatmap",
            "Pair Plot",
            "QQ Plot",
            "Residual Plot",
            "3D Scatter Plot",
            "Bubble Plot",
            "Time Series Plot"
        ]
    )

# -----------------------------------------
# HISTOGRAM
# -----------------------------------------

    if graph_type == "Histogram":

        var = st.selectbox("Variable", numeric_cols)

        bins = st.slider("Number of bins", 5, 50, 20)

        if graph_engine == "Matplotlib / Seaborn":

            fig, ax = plt.subplots()

            sns.histplot(df[var], bins=bins, kde=True, cmap=color_theme)

            ax.set_title("Histogram")

            st.pyplot(fig)

        else:

            fig = px.histogram(
                df,
                x=var,
                nbins=bins,
                color_discrete_sequence=px.colors.qualitative.Set2
            )

            st.plotly_chart(fig)

# -----------------------------------------
# BOX PLOT
# -----------------------------------------

    elif graph_type == "Box Plot":

        y = st.selectbox("Numeric Variable", numeric_cols)

        if len(cat_cols) > 0:
            x = st.selectbox("Group Variable", cat_cols)
        else:
            x = None

        fig = px.box(df, x=x, y=y, color=x)

        st.plotly_chart(fig)

# -----------------------------------------
# VIOLIN PLOT
# -----------------------------------------

    elif graph_type == "Violin Plot":

        y = st.selectbox("Variable", numeric_cols)

        fig = px.violin(df, y=y, box=True, points="all")

        st.plotly_chart(fig)

# -----------------------------------------
# DENSITY PLOT
# -----------------------------------------

    elif graph_type == "Density Plot":

        var = st.selectbox("Variable", numeric_cols)

        fig, ax = plt.subplots()

        sns.kdeplot(df[var], fill=True, cmap=color_theme)

        st.pyplot(fig)

# -----------------------------------------
# SCATTER PLOT
# -----------------------------------------

    elif graph_type == "Scatter Plot":

        x = st.selectbox("X Variable", numeric_cols)
        y = st.selectbox("Y Variable", numeric_cols)

        color = st.selectbox("Color Group", [None]+cat_cols)

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            trendline="ols"
        )

        st.plotly_chart(fig)

# -----------------------------------------
# SCATTER MATRIX
# -----------------------------------------

    elif graph_type == "Scatter Matrix":

        vars = st.multiselect("Variables", numeric_cols, default=numeric_cols[:4])

        fig = px.scatter_matrix(df, dimensions=vars)

        st.plotly_chart(fig)

# -----------------------------------------
# LINE PLOT
# -----------------------------------------

    elif graph_type == "Line Plot":

        x = st.selectbox("X Variable", df.columns)
        y = st.selectbox("Y Variable", numeric_cols)

        fig = px.line(df, x=x, y=y)

        st.plotly_chart(fig)

# -----------------------------------------
# BAR CHART
# -----------------------------------------

    elif graph_type == "Bar Chart":

        x = st.selectbox("Category", cat_cols)
        y = st.selectbox("Value", numeric_cols)

        fig = px.bar(df, x=x, y=y, color=x)

        st.plotly_chart(fig)

# -----------------------------------------
# PIE CHART
# -----------------------------------------

    elif graph_type == "Pie Chart":

        var = st.selectbox("Category Variable", cat_cols)

        fig = px.pie(df, names=var)

        st.plotly_chart(fig)

# -----------------------------------------
# AREA CHART
# -----------------------------------------

    elif graph_type == "Area Chart":

        x = st.selectbox("X Variable", df.columns)
        y = st.selectbox("Y Variable", numeric_cols)

        fig = px.area(df, x=x, y=y)

        st.plotly_chart(fig)

# -----------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------

    elif graph_type == "Correlation Heatmap":

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots()

        sns.heatmap(
            corr,
            annot=True,
            cmap=color_theme
        )

        st.pyplot(fig)

# -----------------------------------------
# PAIR PLOT
# -----------------------------------------

    elif graph_type == "Pair Plot":

        vars = st.multiselect("Variables", numeric_cols, default=numeric_cols[:4])

        fig = sns.pairplot(df[vars])

        st.pyplot(fig)

# -----------------------------------------
# QQ PLOT
# -----------------------------------------

    elif graph_type == "QQ Plot":

        var = st.selectbox("Variable", numeric_cols)

        fig = sm.qqplot(df[var].dropna(), line="s")

        st.pyplot(fig)

# -----------------------------------------
# RESIDUAL PLOT
# -----------------------------------------

    elif graph_type == "Residual Plot":

        x = st.selectbox("X Variable", numeric_cols)
        y = st.selectbox("Y Variable", numeric_cols)

        fig, ax = plt.subplots()

        sns.residplot(x=df[x], y=df[y], ax=ax)

        st.pyplot(fig)

# -----------------------------------------
# 3D SCATTER
# -----------------------------------------

    elif graph_type == "3D Scatter Plot":

        x = st.selectbox("X", numeric_cols)
        y = st.selectbox("Y", numeric_cols)
        z = st.selectbox("Z", numeric_cols)

        fig = px.scatter_3d(df, x=x, y=y, z=z)

        st.plotly_chart(fig)

# -----------------------------------------
# BUBBLE PLOT
# -----------------------------------------

    elif graph_type == "Bubble Plot":

        x = st.selectbox("X Variable", numeric_cols)
        y = st.selectbox("Y Variable", numeric_cols)
        size = st.selectbox("Bubble Size", numeric_cols)

        fig = px.scatter(df, x=x, y=y, size=size)

        st.plotly_chart(fig)

# -----------------------------------------
# TIME SERIES
# -----------------------------------------

    elif graph_type == "Time Series Plot":

        x = st.selectbox("Time Variable", df.columns)
        y = st.selectbox("Series Variable", numeric_cols)

        fig = px.line(df, x=x, y=y)

        st.plotly_chart(fig)

    st.success("Graph generated successfully.")
