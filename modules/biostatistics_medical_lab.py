import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


# ---------------------------------------------------
# CLINICAL TRIAL ANALYSIS
# ---------------------------------------------------

def clinical_trial_analysis(df, treatment, outcome):

    groups = df.groupby(treatment)[outcome].mean()

    return groups


def interpret_clinical_trial(groups):

    return f"""
Clinical trial results indicate differences in outcome
between treatment groups.

Average outcomes:

{groups}

Interpretation:
If treatment group mean is substantially higher (or lower),
the treatment may have a significant clinical effect.
"""


# ---------------------------------------------------
# EPIDEMIOLOGICAL MODELING
# ---------------------------------------------------

def epidemiology_rates(df, disease, population):

    incidence = df[disease].sum() / df[population].sum()

    prevalence = df[disease].mean()

    return incidence, prevalence


def interpret_epidemiology(inc, prev):

    return f"""
Incidence rate = {inc:.4f}
Prevalence = {prev:.4f}

Interpretation:

Incidence represents the probability of new disease cases
in the population during the study period.

Prevalence indicates the proportion of individuals
currently affected by the disease.
"""


# ---------------------------------------------------
# DIAGNOSTIC TEST EVALUATION
# ---------------------------------------------------

def diagnostic_test(df, actual, predicted):

    cm = confusion_matrix(df[actual], df[predicted])

    TN, FP, FN, TP = cm.ravel()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity, cm


def roc_analysis(df, actual, prob):

    fpr, tpr, thresholds = roc_curve(df[actual], df[prob])

    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# ---------------------------------------------------
# DISEASE RISK MODEL (LOGISTIC)
# ---------------------------------------------------

def disease_risk_model(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]

    return model, probs, y_test


def interpret_risk():

    return """
Logistic regression estimates disease risk
based on predictor variables.

Positive coefficients indicate increased
disease risk, while negative coefficients
suggest protective effects.
"""


# ---------------------------------------------------
# SURVIVAL ANALYSIS
# ---------------------------------------------------

def kaplan_meier(df, time, event):

    kmf = KaplanMeierFitter()

    kmf.fit(df[time], event_observed=df[event])

    return kmf


def cox_model(df, time, event):

    cph = CoxPHFitter()

    cph.fit(df, duration_col=time, event_col=event)

    return cph


# ---------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------

def run(df):

    st.title("StatX Biostatistics & Medical Statistics Laboratory")

    if df is None:

        st.warning("Upload dataset first")
        return

# ---------------------------------------------------
# CLINICAL TRIAL
# ---------------------------------------------------

    st.header("Clinical Trial Analysis")

    treatment = st.selectbox("Treatment Variable", df.columns)
    outcome = st.selectbox("Outcome Variable", df.columns)

    if st.button("Run Clinical Trial Analysis"):

        groups = clinical_trial_analysis(df, treatment, outcome)

        st.write(groups)

        st.write(interpret_clinical_trial(groups))

# ---------------------------------------------------
# EPIDEMIOLOGY
# ---------------------------------------------------

    st.header("Epidemiological Analysis")

    disease = st.selectbox("Disease Variable", df.columns)
    population = st.selectbox("Population Variable", df.columns)

    if st.button("Compute Epidemiology Metrics"):

        inc, prev = epidemiology_rates(df, disease, population)

        st.metric("Incidence Rate", round(inc,4))
        st.metric("Prevalence", round(prev,4))

        st.write(interpret_epidemiology(inc, prev))

# ---------------------------------------------------
# DIAGNOSTIC TEST
# ---------------------------------------------------

    st.header("Diagnostic Test Evaluation")

    actual = st.selectbox("Actual Disease Status", df.columns)
    predicted = st.selectbox("Predicted Diagnosis", df.columns)

    if st.button("Evaluate Diagnostic Test"):

        sens, spec, cm = diagnostic_test(df, actual, predicted)

        st.metric("Sensitivity", round(sens,3))
        st.metric("Specificity", round(spec,3))

        st.write("Confusion Matrix")
        st.write(cm)

# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------

    st.header("ROC Curve Analysis")

    prob = st.selectbox("Prediction Probability Column", df.columns)

    if st.button("Plot ROC Curve"):

        fpr, tpr, roc_auc = roc_analysis(df, actual, prob)

        fig, ax = plt.subplots()

        ax.plot(fpr, tpr)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")

        st.pyplot(fig)

# ---------------------------------------------------
# DISEASE RISK MODEL
# ---------------------------------------------------

    st.header("Disease Risk Modeling")

    target = st.selectbox("Disease Outcome Variable", df.columns)

    if st.button("Run Logistic Risk Model"):

        model, probs, y_test = disease_risk_model(df, target)

        st.write("Predicted Risk Probabilities")

        st.write(probs[:20])

        st.write(interpret_risk())

# ---------------------------------------------------
# SURVIVAL ANALYSIS
# ---------------------------------------------------

    st.header("Medical Survival Analysis")

    time = st.selectbox("Survival Time Variable", df.columns)
    event = st.selectbox("Event Indicator (1=death)", df.columns)

    if st.button("Kaplan-Meier Survival Curve"):

        kmf = kaplan_meier(df, time, event)

        fig, ax = plt.subplots()

        kmf.plot(ax=ax)

        ax.set_title("Kaplan-Meier Survival Curve")

        st.pyplot(fig)

# ---------------------------------------------------
# COX MODEL
# ---------------------------------------------------

    st.header("Cox Proportional Hazards Model")

    if st.button("Run Cox Model"):

        cph = cox_model(df, time, event)

        st.write(cph.summary)
