import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis

from sklearn.neural_network import MLPClassifier

from scipy.cluster.hierarchy import dendrogram, linkage


# -------------------------------------------------------
# INTERPRETATION ENGINE
# -------------------------------------------------------

def interpret_accuracy(acc):

    if acc > 0.9:
        return "Excellent predictive performance."

    elif acc > 0.8:
        return "Very good predictive performance."

    elif acc > 0.7:
        return "Acceptable predictive performance."

    else:
        return "Model performance is weak."


def interpret_clusters(k):

    return f"""
The algorithm divided the observations into **{k} clusters**.

Clusters represent groups of observations with similar
characteristics in the feature space.
"""


def interpret_pca(var):

    return f"""
The first components explain **{var:.2%} of the total variance**.

This indicates that the dataset can be represented with
fewer dimensions while retaining most of the information.
"""


# -------------------------------------------------------
# DECISION TREE
# -------------------------------------------------------

def decision_tree(X_train, X_test, y_train, y_test):

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc, pred


# -------------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------------

def random_forest(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc


# -------------------------------------------------------
# GRADIENT BOOSTING
# -------------------------------------------------------

def gradient_boost(X_train, X_test, y_train, y_test):

    model = GradientBoostingClassifier()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc


# -------------------------------------------------------
# SVM
# -------------------------------------------------------

def svm_model(X_train, X_test, y_train, y_test):

    model = SVC(kernel="rbf")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc


# -------------------------------------------------------
# KMEANS
# -------------------------------------------------------

def kmeans_model(X, k):

    model = KMeans(n_clusters=k)

    clusters = model.fit_predict(X)

    return clusters


# -------------------------------------------------------
# HIERARCHICAL CLUSTERING
# -------------------------------------------------------

def hierarchical_model(X, k):

    model = AgglomerativeClustering(n_clusters=k)

    clusters = model.fit_predict(X)

    return clusters


def dendrogram_plot(X):

    Z = linkage(X, method="ward")

    fig, ax = plt.subplots()

    dendrogram(Z)

    st.pyplot(fig)


# -------------------------------------------------------
# PCA
# -------------------------------------------------------

def pca_model(X):

    pca = PCA()

    components = pca.fit_transform(X)

    variance = pca.explained_variance_ratio_

    return components, variance


# -------------------------------------------------------
# FACTOR ANALYSIS
# -------------------------------------------------------

def factor_model(X, k):

    fa = FactorAnalysis(n_components=k)

    factors = fa.fit_transform(X)

    return factors


# -------------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------------

def neural_network(X_train, X_test, y_train, y_test):

    model = MLPClassifier(hidden_layer_sizes=(50,50))

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc


# -------------------------------------------------------
# MAIN MODULE
# -------------------------------------------------------

def run(df):

    st.title("StatX Machine Learning & AI Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Machine Learning Method",

        [

            "Decision Tree",

            "Random Forest",

            "Gradient Boosting",

            "Support Vector Machine",

            "K-Means Clustering",

            "Hierarchical Clustering",

            "Principal Component Analysis",

            "Factor Analysis",

            "Neural Network"
        ]

    )

# -------------------------------------------------------
# SUPERVISED LEARNING
# -------------------------------------------------------

    if analysis in [

        "Decision Tree",

        "Random Forest",

        "Gradient Boosting",

        "Support Vector Machine",

        "Neural Network"
    ]:

        X_vars = st.multiselect("Predictor Variables", numeric)

        y_var = st.selectbox("Target Variable", numeric)

        X = df[X_vars]

        Y = df[y_var]

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3
        )

# -------------------------------------------------------
# DECISION TREE
# -------------------------------------------------------

        if analysis == "Decision Tree":

            model, acc, pred = decision_tree(
                X_train, X_test, y_train, y_test
            )

            st.write("Accuracy:", acc)

            st.write(interpret_accuracy(acc))

            fig, ax = plt.subplots(figsize=(8,5))

            plot_tree(model, filled=True)

            st.pyplot(fig)

# -------------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------------

        elif analysis == "Random Forest":

            model, acc = random_forest(
                X_train, X_test, y_train, y_test
            )

            st.write("Accuracy:", acc)

            st.write(interpret_accuracy(acc))

# -------------------------------------------------------
# GRADIENT BOOST
# -------------------------------------------------------

        elif analysis == "Gradient Boosting":

            model, acc = gradient_boost(
                X_train, X_test, y_train, y_test
            )

            st.write("Accuracy:", acc)

            st.write(interpret_accuracy(acc))

# -------------------------------------------------------
# SVM
# -------------------------------------------------------

        elif analysis == "Support Vector Machine":

            model, acc = svm_model(
                X_train, X_test, y_train, y_test
            )

            st.write("Accuracy:", acc)

            st.write(interpret_accuracy(acc))

# -------------------------------------------------------
# NEURAL NETWORK
# -------------------------------------------------------

        elif analysis == "Neural Network":

            model, acc = neural_network(
                X_train, X_test, y_train, y_test
            )

            st.write("Accuracy:", acc)

            st.write(interpret_accuracy(acc))

# -------------------------------------------------------
# KMEANS
# -------------------------------------------------------

    elif analysis == "K-Means Clustering":

        vars = st.multiselect("Variables", numeric)

        k = st.slider("Clusters",2,10,3)

        X = df[vars]

        clusters = kmeans_model(X,k)

        df["Cluster"] = clusters

        st.write(interpret_clusters(k))

        fig, ax = plt.subplots()

        sns.scatterplot(
            x=X.iloc[:,0],
            y=X.iloc[:,1],
            hue=clusters
        )

        st.pyplot(fig)

# -------------------------------------------------------
# HIERARCHICAL
# -------------------------------------------------------

    elif analysis == "Hierarchical Clustering":

        vars = st.multiselect("Variables", numeric)

        k = st.slider("Clusters",2,10,3)

        X = df[vars]

        clusters = hierarchical_model(X,k)

        df["Cluster"] = clusters

        dendrogram_plot(X)

# -------------------------------------------------------
# PCA
# -------------------------------------------------------

    elif analysis == "Principal Component Analysis":

        vars = st.multiselect("Variables", numeric)

        X = df[vars]

        components, var = pca_model(X)

        st.write("Explained Variance")

        st.write(var)

        st.write(interpret_pca(var[0]))

# -------------------------------------------------------
# FACTOR
# -------------------------------------------------------

    elif analysis == "Factor Analysis":

        vars = st.multiselect("Variables", numeric)

        k = st.slider("Factors",1,5,2)

        X = df[vars]

        factors = factor_model(X,k)

        st.write("Factor Scores")

        st.write(pd.DataFrame(factors))

    st.success("Machine learning analysis completed")
