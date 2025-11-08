# dashboard/dashboard.py
import streamlit as st
import pandas as pd
import requests
import time

# --- Page Setup ---
st.set_page_config(
    page_title="🧠 MediLearn Dashboard",
    page_icon="💊",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])

# --- Sidebar Controls ---
st.sidebar.title("💊 MediLearn Control Panel")

if st.sidebar.button("🚀 Start Training"):
    try:
        res = requests.post(f"{BACKEND_URL}/start")
        if res.status_code == 200:
            st.success("Training simulation started!")
            st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])
        else:
            st.error(res.text)
    except Exception as e:
        st.error(f"Error: {e}")

if st.sidebar.button("🧹 Reset Simulation"):
    try:
        res = requests.post(f"{BACKEND_URL}/reset")
        st.success(res.json().get("message", "Reset complete."))
    except:
        st.error("Backend not reachable.")

# --- Privacy Controls ---
st.sidebar.markdown("### 🔒 Privacy Controls")
sigma = st.sidebar.slider("Differential Privacy (σ)", 0.0, 1.0, 0.02, step=0.01)
mask_std = st.sidebar.slider("Secure Aggregation Mask Std", 0.0, 1.0, 0.05, step=0.01)
if st.sidebar.button("Apply Privacy Settings"):
    st.sidebar.success("✅ Privacy settings applied (simulation only).")

# --- Main Title ---
st.title("🏥 MediLearn Federated Health AI Dashboard")

# -------------------
# 🩺 Test Global Model
# -------------------
st.subheader("🧠 Test the Global Model")
st.caption("Enter patient indicators below to predict **heart disease risk** using the globally trained model:")

cols = st.columns(2)
input_values = []

# Utility for color-coded warnings
def highlight_text(text, color="red"):
    st.markdown(f"<span style='color:{color};font-weight:600'>{text}</span>", unsafe_allow_html=True)

# 1️⃣ Age
with cols[0]:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=52, step=1, help="Patient age in years.")
    input_values.append(age)
    if age > 60:
        highlight_text("⚠️ Senior Age: Increased cardiovascular risk.")

# 2️⃣ Sex
with cols[1]:
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    input_values.append(sex[1])

# 3️⃣ Chest Pain Type
with cols[0]:
    cp = st.selectbox(
        "Chest Pain Type",
        options=[
            ("Typical Angina (0)", 0),
            ("Atypical Angina (1)", 1),
            ("Non-anginal Pain (2)", 2),
            ("Asymptomatic (3)", 3),
        ],
        format_func=lambda x: x[0],
        help="Different types of chest pain used in diagnosis."
    )
    input_values.append(cp[1])

# 4️⃣ Resting Blood Pressure
with cols[1]:
    bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130, step=1)
    input_values.append(bp)
    if bp > 140:
        highlight_text("⚠️ Hypertension detected (BP > 140 mmHg).")
    elif bp < 90:
        highlight_text("⚠️ Low BP: May indicate poor perfusion.")

# 5️⃣ Serum Cholesterol
with cols[0]:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
    input_values.append(chol)
    if chol > 300:
        highlight_text("🔴 High Cholesterol: Elevated cardiac risk.")
    elif chol < 150:
        highlight_text("⚠️ Low cholesterol (rare).")

# 6️⃣ Fasting Blood Sugar
with cols[1]:
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[("No (0)", 0), ("Yes (1)", 1)],
        format_func=lambda x: x[0],
    )
    input_values.append(fbs[1])
    if fbs[1] == 1:
        highlight_text("⚠️ Diabetic condition detected.")

# 7️⃣ Resting ECG Results
with cols[0]:
    ecg = st.selectbox(
        "Resting ECG Results",
        options=[
            ("Normal (0)", 0),
            ("ST-T Abnormality (1)", 1),
            ("Left Ventricular Hypertrophy (2)", 2),
        ],
        format_func=lambda x: x[0],
        help="Interpretation of resting ECG readings."
    )
    input_values.append(ecg[1])

# 8️⃣ Max Heart Rate
with cols[1]:
    maxhr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    input_values.append(maxhr)
    if maxhr < 100:
        highlight_text("⚠️ Low heart rate response during exercise.")

# 9️⃣ Exercise Induced Angina
with cols[0]:
    angina = st.selectbox("Exercise Induced Angina", options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])
    input_values.append(angina[1])
    if angina[1] == 1:
        highlight_text("⚠️ Angina detected: Monitor carefully.")

# 🔟 ST Depression
with cols[1]:
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.5, step=0.1)
    input_values.append(oldpeak)
    if oldpeak > 2:
        highlight_text("⚠️ Significant ST depression → Ischemia risk.")

# --- Prediction Button ---
if st.button("🧮 Run Prediction"):
    try:
        res = requests.post(f"{BACKEND_URL}/predict", json={"features": input_values})
        if res.status_code == 200:
            data = res.json()
            st.success(f"**Prediction:** {data.get('prediction', 'No response')}")
            st.caption(f"⏱ {data.get('timestamp', '')}")
        else:
            st.error(f"Backend error: {res.text}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# -------------------
# 💡 Explainable AI
# -------------------
st.subheader("💡 Explainable AI (Feature Importance)")
if st.button("🧩 Generate Explanation"):
    try:
        res = requests.get(f"{BACKEND_URL}/explain")
        if res.status_code == 200 and "image/png" in res.headers.get("content-type", ""):
            with open("explanation.png", "wb") as f:
                f.write(res.content)
            st.image("explanation.png", caption="Global Model Feature Importance", use_container_width=True)
        else:
            st.error(f"Backend error: {res.text}")
    except Exception as e:
        st.error(f"Failed to fetch explanation: {e}")

st.divider()

# -------------------
# 🔐 Privacy Summary
# -------------------
st.subheader("🔐 Privacy & Security Overview")
try:
    resp = requests.get(f"{BACKEND_URL}/privacy_stats")
    if resp.status_code == 200:
        stats = resp.json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Differential Privacy (σ)", sigma)
        col2.metric("Secure Aggregation Mask Std", mask_std)
        col3.metric("Privacy Utility", "98.0%")
        st.progress(0.98)
    else:
        st.warning("No privacy data yet.")
except:
    st.warning("Backend not reachable for privacy stats.")

st.divider()

# -------------------
# 📊 Training Progress
# -------------------
st.subheader("📈 Global Model Training Progress")
try:
    status_res = requests.get(f"{BACKEND_URL}/status")
    if status_res.status_code == 200:
        status = status_res.json()
        global_acc = status.get("global_accuracy", 0)
        cycle = status.get("cycle", 0)
        hospitals = status.get("hospitals", [])

        if cycle > 0 and (st.session_state.history.empty or st.session_state.history.iloc[-1]["Cycle"] != cycle):
            new_row = pd.DataFrame({"Cycle": [cycle], "Global Accuracy": [global_acc]})
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🔁 Cycle", cycle)
        col2.metric("🧠 Global Accuracy", f"{global_acc * 100:.1f}%")
        if hospitals:
            for i, hosp in enumerate(hospitals[:3]):
                col = [col3, col4, col5][i]
                col.metric(f"🏥 {hosp.get('hospital', f'Hospital_{chr(65+i)}')}", f"{hosp.get('accuracy', 0) * 100:.1f}%")

        if not st.session_state.history.empty:
            st.line_chart(st.session_state.history, x="Cycle", y="Global Accuracy")
        else:
            st.info("Waiting for training data...")
except Exception as e:
    st.error(f"Failed to fetch training data: {e}")

time.sleep(3)
st.rerun()
