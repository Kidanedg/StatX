import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# VIF CALCULATION
# ---------------------------------------------------------

def calculate_vif(X):

    vif_data = pd.DataFrame()

    vif_data["Variable"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    vif_data["Tolerance"] = 1 / vif_data["VIF"]

    return vif_data


# ---------------------------------------------------------
# CONDITION INDEX & EIGENVALUES
# ---------------------------------------------------------

def condition_index(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    corr_matrix = np.corrcoef(X_scaled, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    condition_index = np.sqrt(max(eigenvalues) / eigenvalues)

    result = pd.DataFrame({
        "Eigenvalue": eigenvalues,
        "Condition Index": condition_index
    })

    return result


# ---------------------------------------------------------
# VARIANCE DECOMPOSITION
# ---------------------------------------------------------

def variance_decomposition(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    U, s, V = np.linalg.svd(X_scaled)

    phi = (V.T ** 2)

    variance_prop = phi / phi.sum(axis=0)

    variance_prop = pd.DataFrame(
        variance_prop,
        columns=X.columns
    )

    return variance_prop


# ---------------------------------------------------------
# CORRELATION CHECK
# ---------------------------------------------------------

def high_correlation(df, threshold=0.8):

    corr = df.corr().abs()

    high_corr = []

    for i in range(len(corr.columns)):
        for j in range(i):

            if corr.iloc[i, j] > threshold:

                high_corr.append(
                    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                )

    return high_corr


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def correlation_heatmap(X):

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        X.corr(),
        annot=True,
        cmap="coolwarm"
    )

    st.pyplot(fig)


def vif_barplot(vif_df):

    fig, ax = plt.subplots()

    sns.barplot(
        x="VIF",
        y="Variable",
        data=vif_df
    )

    plt.title("Variance Inflation Factor")

    st.pyplot(fig)


# ---------------------------------------------------------
# INTERPRETATION ENGINE
# ---------------------------------------------------------

def interpret_vif(vif_df):

    st.subheader("Multicollinearity Interpretation")

    for i in range(len(vif_df)):

        var = vif_df.loc[i,"Variable"]
        vif = vif_df.loc[i,"VIF"]

        if vif < 5:

            st.write(
                f"{var}: VIF={vif:.2f} → No serious multicollinearity"
            )

        elif vif < 10:

            st.write(
                f"{var}: VIF={vif:.2f} → Moderate multicollinearity"
            )

        else:

            st.write(
                f"{var}: VIF={vif:.2f} → Severe multicollinearity problem"
            )


def interpret_condition_index(ci_df):

    st.subheader("Condition Index Interpretation")

    max_ci = ci_df["Condition Index"].max()

    st.write("Maximum Condition Index:", max_ci)

    if max_ci < 10:

        st.write("No multicollinearity problem detected.")

    elif max_ci < 30:

        st.write("Moderate multicollinearity detected.")

    else:

        st.write("Severe multicollinearity detected.")


# ---------------------------------------------------------
# MAIN MODULE
# ---------------------------------------------------------

def run(df):

    st.title("StatX Multicollinearity Diagnostics Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    predictors = st.multiselect(
        "Select Predictor Variables",
        numeric_cols
    )

    if len(predictors) < 2:

        st.info("Select at least two predictors")

        return

    X = df[predictors].dropna()

    st.subheader("Dataset Preview")

    st.dataframe(X.head())

# ---------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------

    st.subheader("Correlation Matrix")

    correlation_heatmap(X)

# ---------------------------------------------------------
# HIGH CORRELATION WARNING
# ---------------------------------------------------------

    st.subheader("High Pairwise Correlation")

    high_corr = high_correlation(X)

    if len(high_corr) == 0:

        st.success("No high correlations above threshold")

    else:

        for pair in high_corr:

            st.write(
                f"{pair[0]} vs {pair[1]} → correlation = {pair[2]:.3f}"
            )

# ---------------------------------------------------------
# VIF
# ---------------------------------------------------------

    st.subheader("Variance Inflation Factor")

    vif_df = calculate_vif(X)

    st.dataframe(vif_df)

    vif_barplot(vif_df)

    interpret_vif(vif_df)

# ---------------------------------------------------------
# CONDITION INDEX
# ---------------------------------------------------------

    st.subheader("Eigenvalues and Condition Index")

    ci_df = condition_index(X)

    st.dataframe(ci_df)

    interpret_condition_index(ci_df)

# ---------------------------------------------------------
# VARIANCE DECOMPOSITION
# ---------------------------------------------------------

    st.subheader("Variance Decomposition Proportions")

    var_dec = variance_decomposition(X)

    st.dataframe(var_dec)

    st.success("Multicollinearity diagnostics completed")
