import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import scipy.stats as stats


# ------------------------------------------------
# DATA TYPE DETECTION
# ------------------------------------------------

def detect_data_structure(df):

    numeric = df.select_dtypes(include=np.number).columns
    categorical = df.select_dtypes(exclude=np.number).columns

    structure = {

        "observations":df.shape[0],
        "variables":df.shape[1],
        "numeric_variables":len(numeric),
        "categorical_variables":len(categorical)

    }

    return structure, numeric, categorical


# ------------------------------------------------
# EXPERIMENT DESIGN ENGINE
# ------------------------------------------------

def design_experiment(df):

    factors = df.columns[:-1]
    response = df.columns[-1]

    design = {

        "factors":list(factors),
        "response_variable":response,
        "recommended_design":"factorial experiment"
    }

    return design


# ------------------------------------------------
# MODEL SELECTION ENGINE
# ------------------------------------------------

def choose_model(df, target):

    y = df[target]

    if len(y.unique()) <= 2:

        return "logistic_regression"

    elif len(y.unique()) < 10:

        return "classification"

    else:

        return "regression"


# ------------------------------------------------
# AUTOMATED MODEL FITTING
# ------------------------------------------------

def build_model(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.3,random_state=1
    )

    if len(y.unique()) <= 2:

        model = LogisticRegression(max_iter=200)

        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        score = accuracy_score(y_test,preds)

        model_type="Logistic Regression"

    else:

        model = RandomForestRegressor()

        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        score = r2_score(y_test,preds)

        model_type="Random Forest Regression"

    return model_type, score


# ------------------------------------------------
# STATISTICAL ANALYSIS ENGINE
# ------------------------------------------------

def statistical_analysis(df):

    corr = df.corr(numeric_only=True)

    pvalues = {}

    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i+1,len(cols)):

            r,p = stats.pearsonr(df[cols[i]],df[cols[j]])

            pvalues[(cols[i],cols[j])] = p

    return corr, pvalues


# ------------------------------------------------
# SCIENTIFIC INTERPRETATION ENGINE
# ------------------------------------------------

def interpret_results(model_type, score):

    if "Regression" in model_type:

        return f"""
Model type: {model_type}

Model explains approximately {score*100:.2f}% of variation.

Interpretation:

The predictors demonstrate measurable influence on the response
variable. The model provides a reasonable representation of
the observed data structure.
"""

    else:

        return f"""
Model type: {model_type}

Prediction accuracy: {score*100:.2f}%

Interpretation:

The classification model successfully distinguishes
between outcome categories with high predictive accuracy.
"""


# ------------------------------------------------
# HYPOTHESIS GENERATOR
# ------------------------------------------------

def generate_hypotheses(corr):

    hypotheses=[]

    for col1 in corr.columns:

        for col2 in corr.columns:

            if col1 != col2:

                r=corr.loc[col1,col2]

                if abs(r)>0.6:

                    hypotheses.append(

f"Hypothesis: {col1} is significantly associated with {col2}."

                    )

    return list(set(hypotheses))


# ------------------------------------------------
# RESEARCH DIRECTIONS
# ------------------------------------------------

def suggest_research(hypotheses):

    directions=[]

    for h in hypotheses[:5]:

        directions.append(

f"Future research should investigate the causal mechanisms underlying: {h}"

        )

    return directions


# ------------------------------------------------
# STREAMLIT INTERFACE
# ------------------------------------------------

def run(df):

    st.title("StatX Scientific Super AI")

    if df is None:

        st.warning("Upload dataset first")
        return

# ------------------------------------------------
# DATA STRUCTURE
# ------------------------------------------------

    structure, num, cat = detect_data_structure(df)

    st.header("Dataset Intelligence")

    st.write(structure)


# ------------------------------------------------
# EXPERIMENT DESIGN
# ------------------------------------------------

    st.header("AI Experiment Design")

    design = design_experiment(df)

    st.write(design)


# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------

    st.header("Automatic Model Selection")

    target = st.selectbox("Select Target Variable", df.columns)

    model_choice = choose_model(df,target)

    st.write("Recommended Model:", model_choice)


# ------------------------------------------------
# MODEL BUILDING
# ------------------------------------------------

    st.header("Automated Modeling")

    if st.button("Run AI Modeling"):

        model_type, score = build_model(df,target)

        st.metric("Model Performance",round(score,3))

        st.write(interpret_results(model_type,score))


# ------------------------------------------------
# STATISTICAL ANALYSIS
# ------------------------------------------------

    st.header("AI Statistical Analysis")

    corr, pvals = statistical_analysis(df)

    st.dataframe(corr)


# ------------------------------------------------
# HYPOTHESES
# ------------------------------------------------

    st.header("Generated Scientific Hypotheses")

    hyps = generate_hypotheses(corr)

    for h in hyps[:10]:

        st.write("-",h)


# ------------------------------------------------
# FUTURE RESEARCH
# ------------------------------------------------

    st.header("Suggested Research Directions")

    directions = suggest_research(hyps)

    for d in directions:

        st.write("-",d)
