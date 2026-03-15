import streamlit as st
import app

st.set_page_config(
    page_title="StatX Scientific Platform",
    layout="wide"
)

st.title("StatX Scientific Platform")

app.run_statx()
