import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ---------------------------------------------
# WORKFLOW NODE DEFINITIONS
# ---------------------------------------------

class WorkflowNode:

    def __init__(self, name, function):
        self.name = name
        self.function = function


# ---------------------------------------------
# NODE FUNCTIONS
# ---------------------------------------------

def dataset_node(df):

    return df


def clean_data_node(df):

    df = df.dropna()

    return df


def scale_features_node(df):

    num = df.select_dtypes(include=np.number)

    scaler = StandardScaler()

    df[num.columns] = scaler.fit_transform(num)

    return df


def regression_node(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)

    return r2


# ---------------------------------------------
# WORKFLOW EXECUTION ENGINE
# ---------------------------------------------

class WorkflowEngine:

    def __init__(self, nodes):
        self.nodes = nodes

    def run(self, df, target):

        data = df

        for node in self.nodes:

            if node.name == "Regression":

                result = node.function(data, target)

            else:

                data = node.function(data)

        return result


# ---------------------------------------------
# STREAMLIT VISUAL WORKFLOW BUILDER
# ---------------------------------------------

def run(df):

    st.title("StatX Visual Workflow Builder")

    if df is None:

        st.warning("Upload dataset first")
        return

# ---------------------------------------------
# AVAILABLE BLOCKS
# ---------------------------------------------

    st.sidebar.header("Workflow Blocks")

    blocks = [

        "Dataset",
        "Data Cleaning",
        "Feature Scaling",
        "Regression Model"

    ]

    selected_blocks = st.sidebar.multiselect(

        "Drag blocks into workflow",

        blocks

    )

# ---------------------------------------------
# WORKFLOW DISPLAY
# ---------------------------------------------

    st.subheader("Workflow Pipeline")

    if len(selected_blocks) == 0:

        st.info("Add blocks to create workflow")

    else:

        workflow_text = " → ".join(selected_blocks)

        st.code(workflow_text)

# ---------------------------------------------
# TARGET VARIABLE
# ---------------------------------------------

    target = st.selectbox("Target Variable", df.columns)

# ---------------------------------------------
# BUILD WORKFLOW NODES
# ---------------------------------------------

    node_objects = []

    for block in selected_blocks:

        if block == "Dataset":

            node_objects.append(
                WorkflowNode("Dataset", dataset_node)
            )

        elif block == "Data Cleaning":

            node_objects.append(
                WorkflowNode("Cleaning", clean_data_node)
            )

        elif block == "Feature Scaling":

            node_objects.append(
                WorkflowNode("Scaling", scale_features_node)
            )

        elif block == "Regression Model":

            node_objects.append(
                WorkflowNode("Regression", regression_node)
            )

# ---------------------------------------------
# EXECUTE WORKFLOW
# ---------------------------------------------

    if st.button("Run Workflow"):

        engine = WorkflowEngine(node_objects)

        result = engine.run(df, target)

        st.success("Workflow executed")

        st.metric("Model R²", round(result,3))
