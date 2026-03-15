import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.cross_decomposition import CCA
import statsmodels.api as sm


# ----------------------------------------------------------
# PCA WITH ADVANCED DIAGNOSTICS
# ----------------------------------------------------------

def run_pca(df, variables):

    X = df[variables]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    components = pca.fit_transform(X_scaled)

    eigenvalues = pca.explained_variance_
    variance_ratio = pca.explained_variance_ratio_

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(variables))],
        index=variables
    )

    return components, eigenvalues, variance_ratio, loadings


# ----------------------------------------------------------
# CANONICAL CORRELATION ANALYSIS
# ----------------------------------------------------------

def run_cca(df, set1, set2):

    X = df[set1]
    Y = df[set2]

    cca = CCA(n_components=2)

    X_c, Y_c = cca.fit_transform(X, Y)

    correlations = [
        np.corrcoef(X_c[:,i], Y_c[:,i])[0,1]
        for i in range(X_c.shape[1])
    ]

    return correlations


# ----------------------------------------------------------
# DISCRIMINANT ANALYSIS
# ----------------------------------------------------------

def run_lda(df, predictors, target):

    X = df[predictors]
    y = df[target]

    lda = LinearDiscriminantAnalysis()

    lda.fit(X, y)

    score = lda.score(X, y)

    return lda, score


# ----------------------------------------------------------
# MULTIDIMENSIONAL SCALING
# ----------------------------------------------------------

def run_mds(df, variables):

    X = df[variables]

    mds = MDS(n_components=2)

    coords = mds.fit_transform(X)

    return coords


# ----------------------------------------------------------
# CLUSTER VALIDATION
# ----------------------------------------------------------

def cluster_validation(df, variables):

    X = df[variables]

    scores = []

    for k in range(2,10):

        model = KMeans(n_clusters=k)

        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)

        scores.append((k, sil))

    return scores


# ----------------------------------------------------------
# STRUCTURAL EQUATION MODELING (SIMPLIFIED)
# ----------------------------------------------------------

def run_sem(df, y, x_vars):

    X = df[x_vars]

    X = sm.add_constant(X)

    model = sm.OLS(df[y], X).fit()

    return model


# ----------------------------------------------------------
# STREAMLIT INTERFACE
# ----------------------------------------------------------

def run(df):

    st.title("StatX Advanced Multivariate Statistics Laboratory")

    if df is None:
        st.warning("Upload dataset first")
        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Multivariate Analysis",

        [

            "Principal Component Analysis",
            "Canonical Correlation Analysis",
            "Discriminant Analysis",
            "Multidimensional Scaling",
            "Cluster Validation",
            "Structural Equation Modeling"

        ]
    )


# ----------------------------------------------------------
# PCA
# ----------------------------------------------------------

    if analysis == "Principal Component Analysis":

        vars = st.multiselect("Variables", numeric)

        if st.button("Run PCA"):

            comp, eig, var_ratio, loadings = run_pca(df, vars)

            st.subheader("Eigenvalues")
            st.write(eig)

            st.subheader("Explained Variance Ratio")
            st.write(var_ratio)

            st.subheader("Component Loadings")
            st.dataframe(loadings)

            fig, ax = plt.subplots()

            ax.plot(var_ratio, marker="o")

            ax.set_title("Scree Plot")

            st.pyplot(fig)

            st.write("""
Interpretation:
Components with high eigenvalues explain the majority
of variance in the dataset.
""")


# ----------------------------------------------------------
# CCA
# ----------------------------------------------------------

    elif analysis == "Canonical Correlation Analysis":

        set1 = st.multiselect("Variable Set 1", numeric)

        set2 = st.multiselect("Variable Set 2", numeric)

        if st.button("Run CCA"):

            corr = run_cca(df, set1, set2)

            st.write("Canonical Correlations")

            st.write(corr)

            st.write("""
Interpretation:
High canonical correlation indicates strong relationships
between the two variable sets.
""")


# ----------------------------------------------------------
# DISCRIMINANT
# ----------------------------------------------------------

    elif analysis == "Discriminant Analysis":

        target = st.selectbox("Group Variable", df.columns)

        predictors = st.multiselect("Predictors", numeric)

        if st.button("Run Discriminant Analysis"):

            lda, score = run_lda(df, predictors, target)

            st.metric("Classification Accuracy", score)

            st.write("""
Interpretation:
Discriminant analysis finds linear combinations
of predictors that best separate groups.
""")


# ----------------------------------------------------------
# MDS
# ----------------------------------------------------------

    elif analysis == "Multidimensional Scaling":

        vars = st.multiselect("Variables", numeric)

        if st.button("Run MDS"):

            coords = run_mds(df, vars)

            fig, ax = plt.subplots()

            ax.scatter(coords[:,0], coords[:,1])

            ax.set_title("MDS Map")

            st.pyplot(fig)

            st.write("""
Interpretation:
Points closer together represent observations
with similar characteristics.
""")


# ----------------------------------------------------------
# CLUSTER VALIDATION
# ----------------------------------------------------------

    elif analysis == "Cluster Validation":

        vars = st.multiselect("Variables", numeric)

        if st.button("Evaluate Clusters"):

            scores = cluster_validation(df, vars)

            ks = [k for k,_ in scores]
            sil = [s for _,s in scores]

            fig, ax = plt.subplots()

            ax.plot(ks, sil, marker="o")

            ax.set_title("Silhouette Score vs Clusters")

            st.pyplot(fig)

            st.write("""
Interpretation:
Higher silhouette score indicates better cluster separation.
""")


# ----------------------------------------------------------
# SEM
# ----------------------------------------------------------

    elif analysis == "Structural Equation Modeling":

        y = st.selectbox("Dependent Variable", numeric)

        x = st.multiselect("Predictors", numeric)

        if st.button("Run SEM"):

            model = run_sem(df, y, x)

            st.text(model.summary())

            st.write("""
Interpretation:
Structural equation modeling evaluates relationships
between latent and observed variables.
""")

    st.success("Multivariate analysis completed.")
