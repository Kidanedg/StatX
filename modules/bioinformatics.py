import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind


# -----------------------------------------------------
# DIFFERENTIAL GENE EXPRESSION
# -----------------------------------------------------

def differential_expression(df, group_col):

    genes = df.drop(columns=[group_col])
    groups = df[group_col].unique()

    g1 = df[df[group_col]==groups[0]].drop(columns=[group_col])
    g2 = df[df[group_col]==groups[1]].drop(columns=[group_col])

    pvals = []
    fold_change = []

    for gene in genes.columns:

        t,p = ttest_ind(g1[gene], g2[gene])

        pvals.append(p)

        fc = np.mean(g1[gene]) - np.mean(g2[gene])

        fold_change.append(fc)

    results = pd.DataFrame({

        "Gene":genes.columns,
        "p-value":pvals,
        "FoldChange":fold_change

    })

    return results.sort_values("p-value")


# -----------------------------------------------------
# PCA FOR GENE EXPRESSION
# -----------------------------------------------------

def pca_analysis(df):

    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(X)

    return pcs, pca.explained_variance_ratio_


# -----------------------------------------------------
# GENE CLUSTERING
# -----------------------------------------------------

def gene_clustering(df):

    model = KMeans(n_clusters=3)

    clusters = model.fit_predict(df)

    return clusters


# -----------------------------------------------------
# HIERARCHICAL CLUSTERING
# -----------------------------------------------------

def hierarchical_cluster(df):

    Z = linkage(df, method="ward")

    return Z


# -----------------------------------------------------
# INTERPRETATIONS
# -----------------------------------------------------

def interpret_dge(results):

    sig = results[results["p-value"] < 0.05]

    n = len(sig)

    return f"""
Differential expression analysis identified {n} genes
with statistically significant expression differences
between biological groups (p < 0.05).

These genes may represent potential biomarkers or
biological pathways associated with the experimental condition.
"""


def interpret_pca(var_ratio):

    total = np.sum(var_ratio)*100

    return f"""
Principal Component Analysis was applied to reduce the
dimensionality of gene expression data.

The first two principal components explain {total:.2f}% of
the total variance in the dataset.

This indicates that the major biological variation
can be summarized using a small number of components.
"""


def interpret_clusters():

    return """
Clustering analysis groups genes or samples with similar
expression patterns.

Genes within the same cluster may share similar biological
functions or regulatory mechanisms.
"""


# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------

def run(df):

    st.title("StatX Bioinformatics Modeling Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    analysis = st.selectbox(

        "Select Bioinformatics Analysis",

        [

            "Differential Gene Expression",
            "Gene Expression PCA",
            "Gene Clustering",
            "Hierarchical Clustering"

        ]
    )

# -----------------------------------------------------
# DIFFERENTIAL EXPRESSION
# -----------------------------------------------------

    if analysis == "Differential Gene Expression":

        group_col = st.selectbox("Group Variable", df.columns)

        if st.button("Run Analysis"):

            results = differential_expression(df, group_col)

            st.subheader("Results")

            st.dataframe(results)

            st.subheader("Interpretation")

            st.write(interpret_dge(results))


# -----------------------------------------------------
# PCA
# -----------------------------------------------------

    elif analysis == "Gene Expression PCA":

        if st.button("Run PCA"):

            pcs, var = pca_analysis(df)

            fig, ax = plt.subplots()

            ax.scatter(pcs[:,0], pcs[:,1])

            ax.set_xlabel("PC1")

            ax.set_ylabel("PC2")

            st.pyplot(fig)

            st.subheader("Interpretation")

            st.write(interpret_pca(var))


# -----------------------------------------------------
# KMEANS
# -----------------------------------------------------

    elif analysis == "Gene Clustering":

        if st.button("Run Clustering"):

            clusters = gene_clustering(df)

            st.write("Cluster Assignments")

            st.write(clusters)

            st.subheader("Interpretation")

            st.write(interpret_clusters())


# -----------------------------------------------------
# HIERARCHICAL
# -----------------------------------------------------

    elif analysis == "Hierarchical Clustering":

        if st.button("Run Hierarchical Clustering"):

            Z = hierarchical_cluster(df)

            fig, ax = plt.subplots()

            dendrogram(Z)

            st.pyplot(fig)

            st.subheader("Interpretation")

            st.write(interpret_clusters())
