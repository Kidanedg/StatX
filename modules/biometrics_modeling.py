import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean, mahalanobis


# ---------------------------------------------------
# BIOMETRIC FEATURE EXTRACTION
# ---------------------------------------------------

def biometric_features(df):

    numeric = df.select_dtypes(include=np.number)

    stats = pd.DataFrame({
        "mean":numeric.mean(),
        "std":numeric.std(),
        "variance":numeric.var(),
        "min":numeric.min(),
        "max":numeric.max()
    })

    return stats


# ---------------------------------------------------
# BIOMETRIC DISTANCE MATRIX
# ---------------------------------------------------

def biometric_distance_matrix(df):

    num = df.select_dtypes(include=np.number)

    matrix = np.zeros((len(num), len(num)))

    for i in range(len(num)):
        for j in range(len(num)):

            matrix[i,j] = euclidean(num.iloc[i], num.iloc[j])

    return matrix


# ---------------------------------------------------
# MAHALANOBIS DISTANCE
# ---------------------------------------------------

def mahalanobis_distance(df):

    num = df.select_dtypes(include=np.number)

    cov = np.cov(num.values.T)

    inv_cov = np.linalg.inv(cov)

    distances = []

    mean = num.mean().values

    for row in num.values:

        d = mahalanobis(row, mean, inv_cov)

        distances.append(d)

    return distances


# ---------------------------------------------------
# BIOMETRIC CLASSIFICATION MODEL
# ---------------------------------------------------

def biometric_classifier(df, target):

    X = df.drop(columns=[target])

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)

    return acc, cm


# ---------------------------------------------------
# PCA BIOMETRIC DIMENSION REDUCTION
# ---------------------------------------------------

def biometric_pca(df):

    num = df.select_dtypes(include=np.number)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(num)

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    variance = pca.explained_variance_ratio_

    return components, variance


# ---------------------------------------------------
# LDA DISCRIMINANT BIOMETRIC MODEL
# ---------------------------------------------------

def lda_biometric(df, target):

    X = df.drop(columns=[target])

    y = df[target]

    lda = LinearDiscriminantAnalysis()

    X_lda = lda.fit_transform(X,y)

    return X_lda


# ---------------------------------------------------
# BIOMETRIC INTERPRETATION
# ---------------------------------------------------

def interpret_biometrics(acc):

    return f"""
Biometric classification accuracy = {acc:.3f}

Interpretation:

The biometric model successfully identifies individuals
or biological classes using quantitative features.

Accuracy above 0.80 indicates strong discriminatory
power of the biometric variables.
"""


# ---------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------

def run(df):

    st.title("StatX Biometrics Modeling Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

# ---------------------------------------------------
# FEATURE ANALYSIS
# ---------------------------------------------------

    st.header("Biometric Feature Analysis")

    stats = biometric_features(df)

    st.dataframe(stats)

# ---------------------------------------------------
# DISTANCE MATRIX
# ---------------------------------------------------

    st.header("Biometric Distance Matrix")

    dist = biometric_distance_matrix(df)

    st.write(dist)

# ---------------------------------------------------
# MAHALANOBIS DISTANCE
# ---------------------------------------------------

    st.header("Mahalanobis Distance Analysis")

    md = mahalanobis_distance(df)

    st.write(md)

# ---------------------------------------------------
# PCA
# ---------------------------------------------------

    st.header("Biometric PCA Visualization")

    comps, var = biometric_pca(df)

    fig, ax = plt.subplots()

    ax.scatter(comps[:,0], comps[:,1])

    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    st.pyplot(fig)

    st.write("Explained variance:", var)

# ---------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------

    st.header("Biometric Classification")

    target = st.selectbox("Select Target Variable", df.columns)

    if st.button("Run Biometric Classification"):

        acc, cm = biometric_classifier(df, target)

        st.metric("Accuracy", round(acc,3))

        st.write("Confusion Matrix")

        st.write(cm)

        st.write(interpret_biometrics(acc))

# ---------------------------------------------------
# LDA
# ---------------------------------------------------

    st.header("Linear Discriminant Analysis")

    target = st.selectbox("Target for LDA", df.columns)

    if st.button("Run LDA"):

        lda_data = lda_biometric(df, target)

        fig, ax = plt.subplots()

        ax.scatter(lda_data[:,0], lda_data[:,0])

        st.pyplot(fig)
