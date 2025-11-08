import streamlit as st
import pandas as pd
import requests
import time
import io

# -------------------------------
# 🎨 PAGE SETUP
# -------------------------------
st.set_page_config(
    page_title="🧠 MediLearn Dashboard",
    page_icon="💊",
    layout="wide"
)

# -------------------------------
# 🌈 ENHANCED VISUAL STYLES (HUMAN-DESIGNED)
# -------------------------------
st.markdown("""
<style>
:root {
    --main-bg: linear-gradient(135deg, #eef4ff, #f9fbff);
    --sidebar-bg: #111827;
    --text-dark: #1e2a47;
    --accent: #4e8cff;
    --accent-light: #60a5fa;
    --card-bg: #ffffffd9;
    --shadow-soft: 0 4px 12px rgba(0, 0, 0, 0.05);
}

body, .stApp {
    background: var(--main-bg);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: var(--text-dark);
}

/* Header */
h1 {
    text-align: center;
    font-weight: 700;
    color: var(--text-dark);
    letter-spacing: -0.3px;
    margin-bottom: 0.2em;
}
p {
    text-align: center;
    color: #476C9B;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    color: white;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
    color: white;
}

/* Sidebar buttons */
.stButton>button {
    background: linear-gradient(to right, var(--accent), var(--accent-light));
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.55em 1.1em;
    font-weight: 600;
    transition: all 0.25s ease-in-out;
    box-shadow: 0 2px 8px rgba(78, 140, 255, 0.2);
}
.stButton>button:hover {
    background: linear-gradient(to right, var(--accent-light), var(--accent));
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(78, 140, 255, 0.3);
}

/* Dividers */
.divider {
    height: 1px;
    background: linear-gradient(to right, #cfdaf5, #f0f4f8);
    margin: 25px 0;
}

/* Metric Cards */
.metric-card {
    background-color: var(--card-bg);
    border-radius: 15px;
    box-shadow: var(--shadow-soft);
    padding: 15px;
    transition: 0.3s ease;
    border: 1px solid rgba(220, 230, 255, 0.7);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: var(--text-dark) !important;
    background: rgba(255,255,255,0.7) !important;
    border-radius: 10px !important;
}

/* Toast */
.stToast {
    background-color: var(--accent) !important;
    color: white !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
}

/* Section Title */
h2, h3, .section-title {
    color: var(--text-dark);
    font-weight: 600;
}

/* Metric numbers */
[data-testid="stMetricValue"] {
    color: var(--accent);
    font-weight: 600;
}

/* Cards in hospital list */
.block-container div[data-testid="stHorizontalBlock"] > div {
    transition: transform 0.2s ease-in-out;
}
.block-container div[data-testid="stHorizontalBlock"] > div:hover {
    transform: scale(1.03);
}

/* Animated Header Strip */
header:before {
    content: "";
    display: block;
    height: 4px;
    background: linear-gradient(to right, #4e8cff, #60a5fa, #7bd4ff);
    animation: gradientShift 4s ease-in-out infinite alternate;
    border-radius: 3px;
    margin-bottom: 10px;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# CONFIG
# -------------------------------
BACKEND_URL = "http://127.0.0.1:8000"
MANAGER_URL = "http://127.0.0.1:8600"

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2947/2947979.png", width=100)
st.sidebar.title("💊 MediLearn Control Panel")

if st.sidebar.button("🚀 Start Training"):
    try:
        res = requests.post(f"{BACKEND_URL}/start")
        if res.status_code == 200:
            st.toast("🧠 Training simulation started!", icon="🚀")
            st.session_state.history = pd.DataFrame(columns=["Cycle", "Global Accuracy"])
        else:
            st.error(res.text)
    except Exception as e:
        st.error(f"Error: {e}")

if st.sidebar.button("🧹 Reset Simulation"):
    try:
        res = requests.post(f"{BACKEND_URL}/reset")
        st.toast("Simulation reset complete.", icon="🧹")
    except:
        st.error("Backend not reachable.")

st.sidebar.markdown("### 🔒 Privacy Controls")
sigma = st.sidebar.slider("Differential Privacy (σ)", 0.0, 1.0, 0.02, step=0.01)
mask_std = st.sidebar.slider("Secure Aggregation Mask Std", 0.0, 1.0, 0.05, step=0.01)
if st.sidebar.button("Apply Settings"):
    st.sidebar.success("✅ Privacy settings applied (simulation only).")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1>🏥 MediLearn Federated Health AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p>Empowering hospitals to collaborate securely — without sharing sensitive data.</p>", unsafe_allow_html=True)

# ==============================================================
# 🏥 ADD NEW HOSPITAL NODE
# ==============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🏥 Add New Hospital Node")

with st.expander("➕ Create or Upload New Hospital", expanded=False):
    st.write("Register a new hospital node and automatically integrate it into the federation.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        hospital_name = st.text_input("Hospital Name", placeholder="Hospital_6")
    with c2:
        dataset_name = st.selectbox("Select Dataset", ["heart_disease.csv", "diabetes.csv", "stroke.csv"])
    with c3:
        port = st.number_input("Port", min_value=8001, max_value=9000, value=8006, step=1)

    custom_dataset = st.file_uploader("or Upload Custom Dataset (CSV)", type=["csv"])
    autostart = st.checkbox("Auto-start hospital", value=True)

    if st.button("🚀 Add Hospital Node"):
        if not hospital_name.strip():
            st.error("Please provide a hospital name.")
        else:
            try:
                data = {
                    "hospital_name": hospital_name,
                    "dataset_name": dataset_name,
                    "port": str(port),
                    "autostart": str(autostart).lower(),
                }

                files = None
                if custom_dataset:
                    dataset_bytes = custom_dataset.read()
                    files = {"file": (custom_dataset.name, io.BytesIO(dataset_bytes), "text/csv")}
                    data["dataset_name"] = custom_dataset.name

                res = requests.post(f"{MANAGER_URL}/add_hospital", data=data, files=files)
                if res.status_code == 200:
                    resp = res.json()
                    st.success(f"✅ Hospital created successfully: {resp.get('hospital_name', hospital_name)}")
                    st.json(resp)
                    st.toast("New hospital node launched!", icon="🏥")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to Hospital Manager (port 8600).")

# ==============================================================
# 🏥 REGISTERED HOSPITALS
# ==============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🏥 Registered Hospitals")

try:
    res = requests.get(f"{MANAGER_URL}/list_hospitals")
    if res.status_code == 200:
        hospitals = res.json().get("registered_hospitals", [])
        if hospitals:
            cols = st.columns(3)
            for i, h in enumerate(hospitals):
                with cols[i % 3]:
                    st.markdown(f'<div class="metric-card">🏥 <b>{h}</b></div>', unsafe_allow_html=True)
        else:
            st.info("No hospitals registered yet.")
    else:
        st.error("Failed to fetch hospital list.")
except requests.exceptions.ConnectionError:
    st.warning("⚠️ Hospital Manager not reachable (port 8600).")

# ==============================================================
# 🧠 GLOBAL MODEL TESTING
# ==============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("🧠 Test the Global Model")
st.caption("Enter patient indicators to predict **heart disease risk** using the globally trained model:")

cols = st.columns(2)
input_values = []

def highlight(text, color="red"):
    st.markdown(f"<span style='color:{color}; font-weight:600'>{text}</span>", unsafe_allow_html=True)

with cols[0]:
    age = st.number_input("Age (years)", 1, 120, 52)
    input_values.append(age)
    if age > 60:
        highlight("⚠️ Senior Age: Higher risk.")

with cols[1]:
    sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
    input_values.append(sex[1])

with cols[0]:
    cp = st.selectbox("Chest Pain Type", [("Typical (0)", 0), ("Atypical (1)", 1), ("Non-anginal (2)", 2), ("Asymptomatic (3)", 3)], format_func=lambda x: x[0])
    input_values.append(cp[1])

with cols[1]:
    bp = st.number_input("Resting BP (mmHg)", 80, 200, 130)
    input_values.append(bp)
    if bp > 140:
        highlight("⚠️ Hypertension detected (BP > 140).")

with cols[0]:
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    input_values.append(chol)
    if chol > 300:
        highlight("🔴 High Cholesterol: Risk factor.")

with cols[1]:
    fbs = st.selectbox("Fasting Sugar > 120 mg/dl", [("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])
    input_values.append(fbs[1])

with cols[0]:
    ecg = st.selectbox("Resting ECG", [("Normal (0)", 0), ("ST-T Abnormal (1)", 1), ("LV Hypertrophy (2)", 2)], format_func=lambda x: x[0])
    input_values.append(ecg[1])

with cols[1]:
    maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
    input_values.append(maxhr)

with cols[0]:
    angina = st.selectbox("Exercise Induced Angina", [("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])
    input_values.append(angina[1])

with cols[1]:
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.5, step=0.1)
    input_values.append(oldpeak)

if st.button("🧮 Run Prediction"):
    try:
        res = requests.post(f"{BACKEND_URL}/predict", json={"features": input_values})
        if res.status_code == 200:
            data = res.json()
            st.success(f"🩺 Prediction: {data['prediction_label']} — Confidence: {data['confidence_percent']}%")
        else:
            st.error(f"Backend Error: {res.text}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==============================================================
# 📊 GLOBAL MODEL PROGRESS
# ==============================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.subheader("📈 Global Model Training Progress")

try:
    status_res = requests.get(f"{BACKEND_URL}/status")
    if status_res.status_code == 200:
        status = status_res.json()
        global_acc = status.get("global_accuracy", 0)
        cycle = status.get("cycle", 0)
        hospitals = status.get("hospitals", [])

        col1, col2, *cols = st.columns(7)
        col1.metric("🔁 Current Cycle", cycle)
        col2.metric("🧠 Global Accuracy", f"{global_acc * 100:.1f}%")

        for i, hosp in enumerate(hospitals[:5]):
            name = hosp.get("hospital", f"Hospital_{chr(65+i)}")
            acc = hosp.get("accuracy", 0)
            cols[i].metric(f"🏥 {name}", f"{acc * 100:.1f}%")

        if hospitals:
            df = pd.DataFrame([
                {"Hospital": h.get("hospital"), "Accuracy": h.get("accuracy", 0)}
                for h in hospitals if "hospital" in h
            ])
            st.line_chart(df.set_index("Hospital"), use_container_width=True)
    else:
        st.warning("⚠️ No training data yet.")
except Exception as e:
    st.error(f"Failed to fetch training data: {e}")

time.sleep(3)
st.rerun()
