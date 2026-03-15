import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem


# -----------------------------------------------------
# MOLECULAR DESCRIPTORS
# -----------------------------------------------------

def compute_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    descriptors = {

        "MolWt":Descriptors.MolWt(mol),
        "LogP":Descriptors.MolLogP(mol),
        "HDonors":Descriptors.NumHDonors(mol),
        "HAcceptors":Descriptors.NumHAcceptors(mol),
        "TPSA":Descriptors.TPSA(mol)

    }

    return descriptors


# -----------------------------------------------------
# MOLECULAR DOCKING SCORE (SIMPLIFIED)
# -----------------------------------------------------

def docking_score(smiles):

    mol = Chem.MolFromSmiles(smiles)

    score = Descriptors.MolWt(mol)*0.01 - Descriptors.MolLogP(mol)

    return score


# -----------------------------------------------------
# QSAR MODEL
# -----------------------------------------------------

def train_qsar(df):

    X = df.drop(columns=["Activity"])
    y = df["Activity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, r2, rmse


# -----------------------------------------------------
# ADMET PREDICTION
# -----------------------------------------------------

def admet_prediction(smiles):

    desc = compute_descriptors(smiles)

    logp = desc["LogP"]
    mw = desc["MolWt"]

    toxicity = "Low"

    if logp > 5 or mw > 500:
        toxicity = "High"

    absorption = "Good"

    if desc["TPSA"] > 140:
        absorption = "Poor"

    return toxicity, absorption


# -----------------------------------------------------
# VIRTUAL SCREENING
# -----------------------------------------------------

def virtual_screening(smiles_list):

    scores = []

    for s in smiles_list:

        score = docking_score(s)

        scores.append(score)

    df = pd.DataFrame({

        "SMILES":smiles_list,
        "DockingScore":scores

    })

    df = df.sort_values("DockingScore")

    return df


# -----------------------------------------------------
# INTERPRETATIONS
# -----------------------------------------------------

def interpret_docking(score):

    return f"""
Docking score = {score:.3f}

Interpretation:
Lower docking scores indicate stronger predicted
binding affinity between ligand and target protein.

Compounds with favorable scores may represent
potential drug candidates.
"""


def interpret_qsar(r2, rmse):

    return f"""
QSAR model performance:

R² = {r2:.3f}
RMSE = {rmse:.3f}

Interpretation:
The model predicts biological activity based
on molecular descriptors.

Higher R² indicates stronger predictive power
for drug activity.
"""


def interpret_admet(tox, absb):

    return f"""
ADMET prediction results:

Toxicity: {tox}
Absorption: {absb}

Interpretation:
These pharmacokinetic properties influence
drug safety and bioavailability.
"""


def interpret_screening(df):

    top = df.iloc[0]

    return f"""
Virtual screening completed.

Best candidate molecule:
{top['SMILES']}

Interpretation:
This compound shows the most favorable
predicted docking score and may warrant
further experimental validation.
"""


# -----------------------------------------------------
# STREAMLIT INTERFACE
# -----------------------------------------------------

def run():

    st.title("StatX Computational Drug Discovery Laboratory")

    analysis = st.selectbox(

        "Select Drug Discovery Analysis",

        [

            "Molecular Docking",
            "Ligand Descriptor Analysis",
            "QSAR Modeling",
            "ADMET Prediction",
            "Virtual Drug Screening"

        ]

    )

# -----------------------------------------------------
# DESCRIPTOR ANALYSIS
# -----------------------------------------------------

    if analysis == "Ligand Descriptor Analysis":

        smiles = st.text_input("Enter SMILES string")

        if st.button("Compute Descriptors"):

            desc = compute_descriptors(smiles)

            st.write(desc)


# -----------------------------------------------------
# DOCKING
# -----------------------------------------------------

    elif analysis == "Molecular Docking":

        smiles = st.text_input("Enter Ligand SMILES")

        if st.button("Compute Docking Score"):

            score = docking_score(smiles)

            st.metric("Docking Score", round(score,3))

            st.write(interpret_docking(score))


# -----------------------------------------------------
# QSAR
# -----------------------------------------------------

    elif analysis == "QSAR Modeling":

        file = st.file_uploader("Upload descriptor dataset (CSV)")

        if file and st.button("Train QSAR Model"):

            df = pd.read_csv(file)

            model, r2, rmse = train_qsar(df)

            st.metric("R²", round(r2,3))

            st.metric("RMSE", round(rmse,3))

            st.write(interpret_qsar(r2, rmse))


# -----------------------------------------------------
# ADMET
# -----------------------------------------------------

    elif analysis == "ADMET Prediction":

        smiles = st.text_input("Enter SMILES")

        if st.button("Predict ADMET"):

            tox, absb = admet_prediction(smiles)

            st.metric("Toxicity", tox)

            st.metric("Absorption", absb)

            st.write(interpret_admet(tox, absb))


# -----------------------------------------------------
# VIRTUAL SCREENING
# -----------------------------------------------------

    elif analysis == "Virtual Drug Screening":

        smiles_text = st.text_area("Enter SMILES list (one per line)")

        if st.button("Run Screening"):

            smiles_list = smiles_text.split("\n")

            df = virtual_screening(smiles_list)

            st.dataframe(df)

            st.write(interpret_screening(df))

            fig, ax = plt.subplots()

            ax.bar(range(len(df)), df["DockingScore"])

            ax.set_xlabel("Compound")

            ax.set_ylabel("Docking Score")

            st.pyplot(fig)
