import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


# -----------------------------------------------------
# GENE REGULATORY NETWORK
# -----------------------------------------------------

def build_grn(expression_data, threshold=0.7):

    genes = expression_data.columns
    G = nx.Graph()

    for g in genes:
        G.add_node(g)

    for i in range(len(genes)):
        for j in range(i+1, len(genes)):

            r,_ = pearsonr(expression_data[genes[i]],
                           expression_data[genes[j]])

            if abs(r) >= threshold:
                G.add_edge(genes[i], genes[j], weight=r)

    return G


# -----------------------------------------------------
# PROTEIN-PROTEIN INTERACTION NETWORK
# -----------------------------------------------------

def build_ppi(protein_df):

    G = nx.Graph()

    for _,row in protein_df.iterrows():

        p1 = row["Protein1"]
        p2 = row["Protein2"]
        score = row["Score"]

        G.add_edge(p1, p2, weight=score)

    return G


# -----------------------------------------------------
# METABOLIC PATHWAY NETWORK
# -----------------------------------------------------

def build_metabolic_network(pathway_df):

    G = nx.DiGraph()

    for _,row in pathway_df.iterrows():

        substrate = row["Substrate"]
        product = row["Product"]

        G.add_edge(substrate, product)

    return G


# -----------------------------------------------------
# RNA-SEQ DIFFERENTIAL EXPRESSION
# -----------------------------------------------------

def rnaseq_differential(counts, group):

    genes = counts.columns
    groups = group.unique()

    g1 = counts[group==groups[0]]
    g2 = counts[group==groups[1]]

    pvals=[]
    logfc=[]

    for gene in genes:

        t,p = ttest_ind(g1[gene], g2[gene])

        fc = np.log2(g1[gene].mean()+1) - np.log2(g2[gene].mean()+1)

        pvals.append(p)
        logfc.append(fc)

    res = pd.DataFrame({

        "Gene":genes,
        "log2FC":logfc,
        "p-value":pvals

    })

    return res.sort_values("p-value")


# -----------------------------------------------------
# MULTI-OMICS INTEGRATION
# -----------------------------------------------------

def multiomics_integration(df):

    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    pca = PCA(n_components=2)

    comps = pca.fit_transform(X)

    return comps, pca.explained_variance_ratio_


# -----------------------------------------------------
# INTERPRETATIONS
# -----------------------------------------------------

def interpret_grn(G):

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    return f"""
Gene Regulatory Network constructed.

Nodes (genes): {n_nodes}
Edges (interactions): {n_edges}

Interpretation:
Genes connected in the network may regulate
each other's expression through transcription
factors or signaling pathways.

Highly connected genes (hubs) may represent
key regulatory genes.
"""


def interpret_ppi(G):

    return f"""
Protein interaction network constructed.

Number of proteins: {G.number_of_nodes()}
Interactions: {G.number_of_edges()}

Interpretation:
Proteins that interact physically often
participate in the same biological pathway
or cellular process.
"""


def interpret_metabolic():

    return """
Metabolic pathway network generated.

Interpretation:
Nodes represent metabolites and edges
represent biochemical reactions.

The pathway structure reveals metabolic
flows within the biological system.
"""


def interpret_rnaseq(res):

    sig = res[res["p-value"] < 0.05]

    return f"""
RNA-seq differential expression analysis completed.

Significant genes (p < 0.05): {len(sig)}

Interpretation:
These genes show statistically significant
expression differences between biological
conditions and may represent biomarkers
or key biological pathways.
"""


def interpret_multiomics(var):

    explained = np.sum(var)*100

    return f"""
Multi-omics integration using PCA completed.

The first two components explain {explained:.2f}% of
the combined variation across omics datasets.

Interpretation:
Different biological layers (genomics,
transcriptomics, proteomics, metabolomics)
are summarized into major biological patterns.
"""


# -----------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------

def draw_network(G):

    fig, ax = plt.subplots(figsize=(8,6))

    pos = nx.spring_layout(G)

    nx.draw(G,
            pos,
            with_labels=True,
            node_size=300,
            font_size=8,
            ax=ax)

    return fig


# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------

def run(df):

    st.title("StatX Systems Biology & Omics Integration Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    analysis = st.selectbox(

        "Select Systems Biology Analysis",

        [

            "Gene Regulatory Network",
            "Protein-Protein Interaction Network",
            "Metabolic Pathway Network",
            "RNA-seq Differential Expression",
            "Multi-Omics Integration"

        ]
    )

# -----------------------------------------------------
# GRN
# -----------------------------------------------------

    if analysis == "Gene Regulatory Network":

        threshold = st.slider("Correlation Threshold",0.3,0.95,0.7)

        if st.button("Build Network"):

            G = build_grn(df, threshold)

            fig = draw_network(G)

            st.pyplot(fig)

            st.write(interpret_grn(G))


# -----------------------------------------------------
# PPI
# -----------------------------------------------------

    elif analysis == "Protein-Protein Interaction Network":

        st.info("Dataset must contain Protein1, Protein2, Score")

        if st.button("Build PPI Network"):

            G = build_ppi(df)

            fig = draw_network(G)

            st.pyplot(fig)

            st.write(interpret_ppi(G))


# -----------------------------------------------------
# METABOLIC
# -----------------------------------------------------

    elif analysis == "Metabolic Pathway Network":

        st.info("Dataset must contain Substrate and Product")

        if st.button("Build Metabolic Network"):

            G = build_metabolic_network(df)

            fig = draw_network(G)

            st.pyplot(fig)

            st.write(interpret_metabolic())


# -----------------------------------------------------
# RNASEQ
# -----------------------------------------------------

    elif analysis == "RNA-seq Differential Expression":

        group_col = st.selectbox("Group Variable", df.columns)

        if st.button("Run RNA-seq Model"):

            counts = df.drop(columns=[group_col])

            group = df[group_col]

            res = rnaseq_differential(counts, group)

            st.dataframe(res)

            st.write(interpret_rnaseq(res))


# -----------------------------------------------------
# MULTIOMICS
# -----------------------------------------------------

    elif analysis == "Multi-Omics Integration":

        if st.button("Integrate Omics Data"):

            comps, var = multiomics_integration(df)

            fig, ax = plt.subplots()

            ax.scatter(comps[:,0], comps[:,1])

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            st.pyplot(fig)

            st.write(interpret_multiomics(var))
