import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# -------------------------------------------------------
# LOAD PDB STRUCTURE
# -------------------------------------------------------

def load_structure(pdb_file):

    parser = PDBParser()

    structure = parser.get_structure("protein", pdb_file)

    return structure


# -------------------------------------------------------
# AMINO ACID COMPOSITION
# -------------------------------------------------------

def amino_acid_composition(sequence):

    analysis = ProteinAnalysis(sequence)

    comp = analysis.get_amino_acids_percent()

    return comp


# -------------------------------------------------------
# SECONDARY STRUCTURE ESTIMATION
# -------------------------------------------------------

def estimate_secondary_structure(sequence):

    helix = sequence.count("A") + sequence.count("L") + sequence.count("M")
    sheet = sequence.count("V") + sequence.count("I") + sequence.count("F")

    total = len(sequence)

    helix_p = helix/total*100
    sheet_p = sheet/total*100
    coil_p = 100 - helix_p - sheet_p

    return helix_p, sheet_p, coil_p


# -------------------------------------------------------
# MOLECULAR CONTACT MAP
# -------------------------------------------------------

def contact_map(structure, cutoff=5.0):

    atoms = [atom for atom in structure.get_atoms()]

    coords = np.array([atom.coord for atom in atoms])

    dist = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))

    contact = dist < cutoff

    return contact


# -------------------------------------------------------
# PROTEIN PHYSICOCHEMICAL PROPERTIES
# -------------------------------------------------------

def protein_properties(sequence):

    analysis = ProteinAnalysis(sequence)

    mw = analysis.molecular_weight()
    pi = analysis.isoelectric_point()
    aromaticity = analysis.aromaticity()
    instability = analysis.instability_index()

    return mw, pi, aromaticity, instability


# -------------------------------------------------------
# INTERPRETATIONS
# -------------------------------------------------------

def interpret_amino(comp):

    dominant = max(comp, key=comp.get)

    return f"""
Amino acid composition analysis completed.

The most abundant amino acid is {dominant}.

Interpretation:
Amino acid composition influences protein folding,
stability, and biological function.
"""


def interpret_secondary(helix, sheet, coil):

    return f"""
Secondary structure estimation:

Alpha helix: {helix:.2f}%
Beta sheet: {sheet:.2f}%
Random coil: {coil:.2f}%

Interpretation:
These structural components determine
the 3D conformation of the protein.
"""


def interpret_properties(mw, pi):

    return f"""
Protein physicochemical properties estimated.

Molecular Weight: {mw:.2f} Da
Isoelectric Point: {pi:.2f}

Interpretation:
These parameters influence protein solubility,
folding behavior, and biochemical activity.
"""


def interpret_contacts():

    return """
Contact map generated.

Interpretation:
Residue contact maps reveal spatial
interactions within proteins and
help identify structural domains.
"""


# -------------------------------------------------------
# STREAMLIT INTERFACE
# -------------------------------------------------------

def run():

    st.title("StatX Biomolecular Modeling Laboratory")

    analysis = st.selectbox(

        "Select Biomolecular Analysis",

        [

            "Amino Acid Composition",
            "Secondary Structure Estimation",
            "Protein Physicochemical Properties",
            "Molecular Contact Map"

        ]
    )

# -------------------------------------------------------
# SEQUENCE INPUT
# -------------------------------------------------------

    sequence = st.text_area("Enter Protein Sequence")

# -------------------------------------------------------
# AMINO ACID ANALYSIS
# -------------------------------------------------------

    if analysis == "Amino Acid Composition":

        if st.button("Analyze Composition"):

            comp = amino_acid_composition(sequence)

            df = pd.DataFrame(

                comp.items(),

                columns=["Amino Acid","Frequency"]

            )

            st.dataframe(df)

            fig, ax = plt.subplots()

            ax.bar(df["Amino Acid"], df["Frequency"])

            st.pyplot(fig)

            st.write(interpret_amino(comp))


# -------------------------------------------------------
# SECONDARY STRUCTURE
# -------------------------------------------------------

    elif analysis == "Secondary Structure Estimation":

        if st.button("Estimate Structure"):

            helix, sheet, coil = estimate_secondary_structure(sequence)

            st.metric("Alpha Helix (%)", round(helix,2))
            st.metric("Beta Sheet (%)", round(sheet,2))
            st.metric("Random Coil (%)", round(coil,2))

            st.write(interpret_secondary(helix, sheet, coil))


# -------------------------------------------------------
# PROTEIN PROPERTIES
# -------------------------------------------------------

    elif analysis == "Protein Physicochemical Properties":

        if st.button("Calculate Properties"):

            mw, pi, aro, inst = protein_properties(sequence)

            st.metric("Molecular Weight", round(mw,2))
            st.metric("Isoelectric Point", round(pi,2))
            st.metric("Aromaticity", round(aro,3))
            st.metric("Instability Index", round(inst,2))

            st.write(interpret_properties(mw, pi))


# -------------------------------------------------------
# CONTACT MAP
# -------------------------------------------------------

    elif analysis == "Molecular Contact Map":

        pdb_file = st.file_uploader("Upload PDB File")

        if pdb_file and st.button("Generate Contact Map"):

            structure = load_structure(pdb_file)

            cmap = contact_map(structure)

            fig, ax = plt.subplots()

            ax.imshow(cmap)

            ax.set_title("Residue Contact Map")

            st.pyplot(fig)

            st.write(interpret_contacts())
