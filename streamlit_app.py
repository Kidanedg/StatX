import streamlit as st
import requests
import app

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="StatX Scientific Platform",
    layout="wide"
)

st.title("StatX Scientific Platform")

menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Register"]
)

# -------------------
# REGISTER
# -------------------

if menu == "Register":

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):

        r = requests.post(
            f"{API}/register",
            json={"username": username, "password": password}
        )

        if r.status_code == 200:
            st.success("User created")
        else:
            st.error("Registration failed")

# -------------------
# LOGIN
# -------------------

if menu == "Login":

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        r = requests.post(
            f"{API}/login",
            json={"username": username, "password": password}
        )

        if r.status_code == 200:

            token = r.json()["token"]

            st.session_state["token"] = token
            st.session_state["logged_in"] = True

            st.success("Login successful")

        else:
            st.error("Invalid credentials")

# -------------------
# AFTER LOGIN
# -------------------

if st.session_state.get("logged_in"):

    st.sidebar.success("Logged in")

    app.run_statx()
