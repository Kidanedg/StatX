import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from scipy.stats import pearsonr


# ---------------------------------------------------
# DATA PROFILING
# ---------------------------------------------------

def dataset_profile(df):

    profile = {

        "rows":df.shape[0],
        "columns":df.shape[1],
        "numeric":len(df.select_dtypes(include=np.number).columns),
        "categorical":len(df.select_dtypes(exclude=np.number).columns),
        "missing":df.isnull().sum().sum()

    }

    return profile


# ---------------------------------------------------
# CORRELATION DISCOVERY
# ---------------------------------------------------

def discover_correlations(df):

    num = df.select_dtypes(include=np.number)

    corr = num.corr()

    strong = []

    for i in corr.columns:

        for j in corr.columns:

            if i != j:

                r = corr.loc[i,j]

                if abs(r) > 0.7:

                    strong.append((i,j,r))

    return corr, strong


# ---------------------------------------------------
# CLUSTER DISCOVERY
# ---------------------------------------------------

def discover_clusters(df):

    num = df.select_dtypes(include=np.number)

    if num.shape[1] < 2:

        return None

    model = KMeans(n_clusters=3)

    clusters = model.fit_predict(num)

    return clusters


# ---------------------------------------------------
# PCA PATTERN DISCOVERY
# ---------------------------------------------------

def pca_patterns(df):

    num = df.select_dtypes(include=np.number)

    pca = PCA(n_components=2)

    comps = pca.fit_transform(num)

    var = pca.explained_variance_ratio_

    return comps, var


# ---------------------------------------------------
# AUTOMATIC MODEL BUILDING
# ---------------------------------------------------

def auto_model(df):

    num = df.select_dtypes(include=np.number)

    if num.shape[1] < 2:

        return None

    y = num.iloc[:,-1]
    X = num.iloc[:,:-1]

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.3, random_state=42

    )

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)

    return model, r2


# ---------------------------------------------------
# HYPOTHESIS GENERATION
# ---------------------------------------------------

def generate_hypotheses(strong_corr):

    hypotheses = []

    for a,b,r in strong_corr:

        if r > 0:

            hypotheses.append(

f"H1: Variable '{a}' is positively associated with '{b}'."

            )

        else:

            hypotheses.append(

f"H1: Variable '{a}' is negatively associated with '{b}'."

            )

    return hypotheses


# ---------------------------------------------------
# INTERPRETATION ENGINE
# ---------------------------------------------------

def interpret_profile(profile):

    return f"""
Dataset contains {profile['rows']} observations
and {profile['columns']} variables.

Numeric variables: {profile['numeric']}
Categorical variables: {profile['categorical']}

Missing values detected: {profile['missing']}

Interpretation:
Dataset structure is suitable for statistical
modeling and exploratory data analysis.
"""


def interpret_patterns(strong):

    if len(strong)==0:

        return "No strong correlations detected."

    text="Strong statistical relationships detected:\n\n"

    for a,b,r in strong[:5]:

        text += f"{a} vs {b} (r={r:.2f})\n"

    return text


def interpret_model(r2):

    return f"""
Automatic predictive model constructed.

Model performance:

R² = {r2:.3f}

Interpretation:
The model explains approximately {r2*100:.1f}% of
variation in the target variable.
"""


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------

def run(df):

    st.title("StatX Autonomous Scientific Discovery Engine")

    if df is None:

        st.warning("Upload dataset first")

        return

# ---------------------------------------------------
# DATA PROFILING
# ---------------------------------------------------

    profile = dataset_profile(df)

    st.subheader("Dataset Profiling")

    st.write(profile)

    st.write(interpret_profile(profile))


# ---------------------------------------------------
# CORRELATION DISCOVERY
# ---------------------------------------------------

    st.subheader("Pattern Discovery")

    corr, strong = discover_correlations(df)

    st.write("Correlation Matrix")

    st.dataframe(corr)

    st.write(interpret_patterns(strong))


# ---------------------------------------------------
# CLUSTER DISCOVERY
# ---------------------------------------------------

    clusters = discover_clusters(df)

    if clusters is not None:

        st.subheader("Cluster Discovery")

        st.write(clusters)


# ---------------------------------------------------
# PCA
# ---------------------------------------------------

    comps, var = pca_patterns(df)

    fig, ax = plt.subplots()

    ax.scatter(comps[:,0], comps[:,1])

    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    st.pyplot(fig)

    st.write(

f"First two components explain {np.sum(var)*100:.2f}% variance."

    )


# ---------------------------------------------------
# AUTO MODEL
# ---------------------------------------------------

    model, r2 = auto_model(df)

    st.subheader("Automatic Predictive Modeling")

    st.metric("Model R²", round(r2,3))

    st.write(interpret_model(r2))


# ---------------------------------------------------
# HYPOTHESIS GENERATION
# ---------------------------------------------------

    st.subheader("Generated Research Hypotheses")

    hyps = generate_hypotheses(strong)

    for h in hyps[:10]:

        st.write("-",h)
