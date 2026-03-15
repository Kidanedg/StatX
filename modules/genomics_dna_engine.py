import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio import AlignIO
from Bio.Phylo import draw
from io import StringIO


# ---------------------------------------------------------
# GC CONTENT ANALYSIS
# ---------------------------------------------------------

def gc_content(sequence):

    sequence = sequence.upper()

    g = sequence.count("G")
    c = sequence.count("C")

    gc = (g + c) / len(sequence) * 100

    return gc


# ---------------------------------------------------------
# DNA SEQUENCE ALIGNMENT
# ---------------------------------------------------------

def align_sequences(seq1, seq2):

    matrix = substitution_matrices.load("NUC.4.4")

    alignments = pairwise2.align.globalds(seq1, seq2, matrix, -10, -0.5)

    return alignments[0]


# ---------------------------------------------------------
# MOTIF DETECTION
# ---------------------------------------------------------

def motif_search(sequence, motif):

    sequence = sequence.upper()
    motif = motif.upper()

    positions = []

    for i in range(len(sequence)-len(motif)+1):

        if sequence[i:i+len(motif)] == motif:

            positions.append(i)

    return positions


# ---------------------------------------------------------
# PROTEIN TRANSLATION
# ---------------------------------------------------------

def translate_dna(sequence):

    seq = Seq(sequence)

    protein = seq.translate(to_stop=True)

    return str(protein)


# ---------------------------------------------------------
# CODON USAGE ANALYSIS
# ---------------------------------------------------------

def codon_usage(sequence):

    sequence = sequence.upper()

    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]

    counts = Counter(codons)

    total = sum(counts.values())

    usage = {codon: count/total for codon, count in counts.items()}

    return usage


# ---------------------------------------------------------
# PHYLOGENETIC TREE
# ---------------------------------------------------------

def phylogenetic_tree(fasta_text):

    alignment = AlignIO.read(StringIO(fasta_text), "fasta")

    calculator = DistanceCalculator("identity")

    dm = calculator.get_distance(alignment)

    constructor = DistanceTreeConstructor()

    tree = constructor.nj(dm)

    return tree


# ---------------------------------------------------------
# INTERPRETATIONS
# ---------------------------------------------------------

def interpret_gc(gc):

    if gc < 40:

        return f"""
GC content = {gc:.2f}%.

Interpretation:
The sequence has relatively low GC content,
which may indicate AT-rich genomic regions.

Such regions are often associated with
promoters or regulatory elements.
"""

    elif gc < 60:

        return f"""
GC content = {gc:.2f}%.

Interpretation:
The sequence has balanced GC composition,
typical for many bacterial and eukaryotic genes.
"""

    else:

        return f"""
GC content = {gc:.2f}%.

Interpretation:
The sequence has high GC content,
which may indicate structural stability
or GC-rich genomic domains.
"""


def interpret_alignment(score):

    return f"""
Sequence alignment completed.

Alignment score = {score}

Interpretation:
Higher scores indicate greater sequence similarity.

Closely related sequences may share
common evolutionary ancestry or similar biological functions.
"""


def interpret_motif(motif, positions):

    n = len(positions)

    return f"""
Motif '{motif}' detected {n} times.

Motifs often represent regulatory elements
such as transcription factor binding sites.

Their locations may influence gene expression.
"""


def interpret_protein(protein):

    return f"""
Protein translation completed.

Protein length = {len(protein)} amino acids.

Interpretation:
The translated protein sequence may represent
the functional product encoded by the DNA sequence.
"""


def interpret_codon_usage():

    return """
Codon usage analysis completed.

Interpretation:
Different organisms prefer specific codons
for encoding the same amino acid.

Codon bias can influence gene expression
and protein synthesis efficiency.
"""


def interpret_phylogeny():

    return """
Phylogenetic tree constructed.

Interpretation:
The tree represents evolutionary relationships
between DNA sequences.

Sequences that cluster together likely share
a recent common ancestor.
"""
    

# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

def run():

    st.title("StatX Genomics & DNA Sequence Analysis Engine")

    analysis = st.selectbox(

        "Select Genomics Analysis",

        [

            "GC Content Analysis",
            "DNA Sequence Alignment",
            "Motif Detection",
            "Protein Translation",
            "Codon Usage Analysis",
            "Phylogenetic Tree Construction"

        ]
    )

# ---------------------------------------------------------
# GC CONTENT
# ---------------------------------------------------------

    if analysis == "GC Content Analysis":

        seq = st.text_area("Enter DNA Sequence")

        if st.button("Analyze GC Content"):

            gc = gc_content(seq)

            st.metric("GC Content (%)", round(gc,2))

            st.write(interpret_gc(gc))


# ---------------------------------------------------------
# ALIGNMENT
# ---------------------------------------------------------

    elif analysis == "DNA Sequence Alignment":

        seq1 = st.text_area("Sequence 1")

        seq2 = st.text_area("Sequence 2")

        if st.button("Run Alignment"):

            alignment = align_sequences(seq1, seq2)

            st.text(alignment)

            st.write(interpret_alignment(alignment.score))


# ---------------------------------------------------------
# MOTIF
# ---------------------------------------------------------

    elif analysis == "Motif Detection":

        seq = st.text_area("DNA Sequence")

        motif = st.text_input("Motif")

        if st.button("Search Motif"):

            pos = motif_search(seq, motif)

            st.write("Motif positions:", pos)

            st.write(interpret_motif(motif, pos))


# ---------------------------------------------------------
# PROTEIN
# ---------------------------------------------------------

    elif analysis == "Protein Translation":

        seq = st.text_area("DNA Sequence")

        if st.button("Translate DNA"):

            protein = translate_dna(seq)

            st.text(protein)

            st.write(interpret_protein(protein))


# ---------------------------------------------------------
# CODON USAGE
# ---------------------------------------------------------

    elif analysis == "Codon Usage Analysis":

        seq = st.text_area("DNA Sequence")

        if st.button("Analyze Codons"):

            usage = codon_usage(seq)

            df = pd.DataFrame(

                list(usage.items()),

                columns=["Codon","Frequency"]

            )

            st.dataframe(df)

            fig, ax = plt.subplots()

            ax.bar(df["Codon"], df["Frequency"])

            plt.xticks(rotation=90)

            st.pyplot(fig)

            st.write(interpret_codon_usage())


# ---------------------------------------------------------
# PHYLOGENY
# ---------------------------------------------------------

    elif analysis == "Phylogenetic Tree Construction":

        fasta = st.text_area("Paste FASTA Alignment")

        if st.button("Build Tree"):

            tree = phylogenetic_tree(fasta)

            fig = plt.figure()

            draw(tree)

            st.pyplot(fig)

            st.write(interpret_phylogeny())
