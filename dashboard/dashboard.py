import streamlit as st
import pandas as pd
import numpy as np
import time
import requests # NEW LINE: Import the requests library

# --- Page Setup ---
st.set_page_config(
    page_title="MediLearn Agent Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- Title & Description ---
st.title("🧠 MediLearn Agent Dashboard")
st.write("Live dashboard showing the federated learning agent's progress.")

# --- Backend URL ---
# IMPORTANT: Ask your Backend Teammate (Role #3) for this URL!
# It will probably be 'http://127.0.0.1:8000' or similar.
BACKEND_URL = "http://127.0.0.1:8000" # NEW LINE: A placeholder for your backend server address

# --- Start Button ---
if st.button("🚀 Start New Training Round"):
    try:
        # NEW LINE: This sends a POST request to the backend's /start endpoint
        response = requests.post(f"{BACKEND_URL}/start")
        
        if response.status_code == 200:
            st.success("Started new training round! Fetching results...")
        else:
            st.error("Error starting round. Is the backend server running?")
            
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to backend. Is the server running at " + BACKEND_URL)

st.divider() # A visual separator

# --- Main Metrics (with FAKE data for now) ---
st.subheader("Current Model Accuracy")

col1, col2, col3, col4 = st.columns(4)

col1.metric(label="🔁 Cycle", value="3")
col2.metric(label="🧠 Global Accuracy", value="92.3%", delta="1.2%")
col3.metric(label="🏥 Hospital A", value="87.4%")
col4.metric(label="🏥 Hospital B", value="89.1%")
# TODO: Add Hospital C

# --- Agent Status ---
st.subheader("Agent Status")
st.info("STATUS: Waiting for instructions...")
# TODO: This text will be updated from the backend's /status endpoint

# --- Accuracy Chart (with FAKE data for now) ---
st.subheader("Global Accuracy Over Time")

fake_data = pd.DataFrame(
    {
        "Cycle": [1, 2, 3],
        "Global Accuracy": [0.85, 0.91, 0.923]
    }
)

st.line_chart(fake_data, x="Cycle", y="Global Accuracy")
# TODO: This chart will be updated with REAL data from the /status endpoint