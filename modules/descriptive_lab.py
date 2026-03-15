import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px


def run(df):

    st.title("📊 StatX Descriptive Statistics Laboratory")

    if df is None:
        st.warning("⚠ Upload dataset first.")
        return

    # ------------------------------
    # DATASET OVERVIEW
    # ------------------------------
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    with st.expander("Preview Dataset"):
        st.dataframe(df)

    with st.expander("Variable Types"):
        st.write(df.dtypes)

    st.divider()

    # ------------------------------
    # VARIABLE SELECTION
    # ------------------------------
    st.header("Variable Selection")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.error("Dataset has no numeric variables.")
        return

    selected_vars = st.multiselect(
        "Select variables for analysis",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )

    if not selected_vars:
        st.info("Select variables to begin analysis.")
        return

    data = df[selected_vars]

    st.divider()

    # ------------------------------
    # SUMMARY STATISTICS
    # ------------------------------
    st.header("Summary Statistics")

    summary = pd.DataFrame({
        "Mean": data.mean(),
        "Median": data.median(),
        "Mode": data.mode().iloc[0],
        "Std Dev": data.std(),
        "Variance": data.var(),
        "Min": data.min(),
        "Max": data.max(),
        "Range": data.max() - data.min(),
        "Skewness": data.skew(),
        "Kurtosis": data.kurt(),
        "Missing": data.isna().sum()
    })

    st.dataframe(summary)

    st.download_button(
        "Download Summary",
        summary.to_csv().encode("utf-8"),
        "summary_statistics.csv",
        "text/csv"
    )

    st.divider()

    # ------------------------------
    # QUANTILES
    # ------------------------------
    st.header("Quantile Statistics")

    quantiles = data.quantile([0.05,0.25,0.5,0.75,0.95]).T

    st.dataframe(quantiles)

    st.divider()

    # ------------------------------
    # NORMALITY TESTS
    # ------------------------------
    st.header("Normality Tests")

    normality_results = []

    for col in selected_vars:

        try:
            shapiro = stats.shapiro(data[col].dropna())
            ks = stats.kstest(
                (data[col]-data[col].mean())/data[col].std(),
                'norm'
            )

            normality_results.append({
                "Variable": col,
                "Shapiro p-value": shapiro.pvalue,
                "KS p-value": ks.pvalue
            })

        except:
            pass

    normality_df = pd.DataFrame(normality_results)

    st.dataframe(normality_df)

    st.divider()

    # ------------------------------
    # CORRELATION ANALYSIS
    # ------------------------------
    st.header("Correlation Analysis")

    if len(selected_vars) > 1:

        corr = data.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig)

    else:
        st.info("Select at least two variables.")

    st.divider()

    # ------------------------------
    # OUTLIER DETECTION
    # ------------------------------
    st.header("Outlier Detection (IQR Method)")

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    outliers = ((data < (Q1 - 1.5*IQR)) |
               (data > (Q3 + 1.5*IQR)))

    st.write("Outliers per variable")

    st.dataframe(outliers.sum())

    st.divider()

    # ------------------------------
    # VISUALIZATION LAB
    # ------------------------------
    st.header("Visualization Lab")

    var = st.selectbox("Choose variable", selected_vars)

    plot_type = st.selectbox(
        "Select visualization",
        [
            "Histogram",
            "Boxplot",
            "Density Plot",
            "Violin Plot",
            "QQ Plot"
        ]
    )

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(data[var], kde=True, ax=ax)

    elif plot_type == "Boxplot":
        sns.boxplot(x=data[var], ax=ax)

    elif plot_type == "Density Plot":
        sns.kdeplot(data[var], fill=True, ax=ax)

    elif plot_type == "Violin Plot":
        sns.violinplot(x=data[var], ax=ax)

    elif plot_type == "QQ Plot":
        stats.probplot(data[var].dropna(), plot=ax)

    st.pyplot(fig)

    st.divider()

    # ------------------------------
    # GROUP STATISTICS
    # ------------------------------
    st.header("Group Statistics")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) > 0:

        group_var = st.selectbox("Grouping variable", cat_cols)

        target_var = st.selectbox("Numeric variable", selected_vars)

        group_stats = df.groupby(group_var)[target_var].describe()

        st.dataframe(group_stats)

        fig2 = px.box(
            df,
            x=group_var,
            y=target_var
        )

        st.plotly_chart(fig2)

    else:
        st.info("No categorical variables found.")

    st.divider()

    # ------------------------------
    # DISTRIBUTION SHAPE INTERPRETATION
    # ------------------------------
    st.header("Distribution Shape Interpretation")

    skewness = data.skew()

    interpretation = []

    for var in skewness.index:

        if skewness[var] > 1:
            shape = "Highly Positively Skewed"
        elif skewness[var] > 0.5:
            shape = "Moderately Positively Skewed"
        elif skewness[var] < -1:
            shape = "Highly Negatively Skewed"
        elif skewness[var] < -0.5:
            shape = "Moderately Negatively Skewed"
        else:
            shape = "Approximately Symmetric"

        interpretation.append({
            "Variable": var,
            "Skewness": skewness[var],
            "Interpretation": shape
        })

    st.dataframe(pd.DataFrame(interpretation))

    st.success("StatX descriptive analysis completed successfully.")
