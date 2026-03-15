import streamlit as st
import pandas as pd


def load_dataset():

    st.sidebar.header("Dataset Manager")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv","xlsx","xls","txt","json","parquet","dta","sav"]
    )

    if uploaded_file:

        file_name = uploaded_file.name.lower()

        try:

            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif file_name.endswith(("xlsx","xls")):
                df = pd.read_excel(uploaded_file)

            elif file_name.endswith(".txt"):
                df = pd.read_csv(uploaded_file, sep=None, engine="python")

            elif file_name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            elif file_name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)

            elif file_name.endswith(".dta"):
                df = pd.read_stata(uploaded_file)

            elif file_name.endswith(".sav"):
                import pyreadstat
                df, meta = pyreadstat.read_sav(uploaded_file)

            st.success("Dataset loaded successfully!")

            return df

        except Exception as e:
            st.error(f"Error loading file: {e}")

    return None
