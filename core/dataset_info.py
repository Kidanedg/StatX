import streamlit as st
import pandas as pd


def show_dataset_info(df):

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1,col2,col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", df.isna().sum().sum())

    st.subheader("Variable Types")

    types = pd.DataFrame({
        "Variable": df.columns,
        "Type": df.dtypes
    })

    st.dataframe(types)
