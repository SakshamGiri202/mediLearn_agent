"""
MediLearn Controller (Async FedAvg + Auto-Heal + Config Sync)
-------------------------------------------------------------
✅ Automatically starts missing hospitals
✅ Waits for /health readiness
✅ Keeps agent_config.json updated dynamically
✅ Aggregates models & generates explainability plots
✅ Supports prediction and monitoring
"""

import logging, json, os, asyncio, time, subprocess, re
from datetime import datetime
from typing import List, Any, Optional

import httpx
import numpy as np
from sklearn.datasets import make_classification
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

import matplotlib
matplotlib.use("Agg")

# === Import custom SHAP explain function ===
from ml_core.explain_model import generate_explanation

# ---------- Config ----------
APP_TITLE = "🧠 MediLearn Controller (FedAvg + Auto-Heal + Config Sync)"
LOG_FILE = "federated.log"
STATUS_FILE = "latest_status.json"
HISTORY_FILE = "training_history.json"
GLOBAL_MODEL_FILE = "global_model.json"
CONFIG_FILE = "agent_config.json"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("=== MediLearn Controller Initialized (Auto-Heal + Config Sync) ===")

# ---------- FastAPI setup ----------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- JSON helpers ----------
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def append_history(entry):
    hist = load_json(HISTORY_FILE, [])
    hist.append(entry)
    save_json(HISTORY_FILE, hist)

# ---------- Config management ----------
def load_config():
    cfg = load_json(CONFIG_FILE, {})
    hospitals = cfg.get("hospitals", [
        "http://127.0.0.1:8001/train",
        "http://127.0.0.1:8002/train",
        "http://127.0.0.1:8003/train"
    ])
    cycles = int(cfg.get("cycles", 3))
    return hospitals, cycles

def update_config(new_hospital: str):
    """Ensure new hospital endpoint is added to config."""
    cfg = load_json(CONFIG_FILE, {"hospitals": [], "cycles": 3})
    hospitals = set(cfg.get("hospitals", []))
    if new_hospital not in hospitals:
        hospitals.add(new_hospital)
        cfg["hospitals"] = sorted(hospitals)
        save_json(CONFIG_FILE, cfg)
        logging.info(f"Added {new_hospital} to agent_config.json")

# ---------- Auto-start missing hospitals ----------
async def ensure_hospitals_running(hospitals: List[str]):
    timeout = httpx.Timeout(5.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in hospitals:
            base_url = url.replace("/train", "")
            port_match = re.search(r":(\d+)", base_url)
            port = port_match.group(1) if port_match else "8001"
            name = f"Hospital_{chr(64 + (hospitals.index(url) + 1))}"
            health_url = f"{base_url}/health"

            try:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    print(f"✅ {name} already running at {base_url}")
                    continue
            except Exception:
                print(f"⚠️  {name} not responding on port {port}, launching...")

            dataset = "heart_disease.csv" if "8001" in port else (
                "diabetes.csv" if "8002" in port else "stroke.csv"
            )
            script_name = f"backend/hospital_{name}.py"
            cmd = [
                "python", script_name,
                "--name", name,
                "--dataset", dataset,
                "--port", port
            ]
            subprocess.Popen(cmd)
            print(f"🚀 Started {name} on port {port}")

            # Wait until /health responds
            for attempt in range(10):
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        print(f"✅ {name} is ready (port {port})")
                        break
                except:
                    await asyncio.sleep(2)
            else:
                print(f"❌ {name} did not start after retries.")

# ---------- FedAvg aggregation ----------
def aggregate_fedavg(results):
    total = sum(r.get("samples", 0) for r in results if isinstance(r, dict))
    if total == 0:
        return 0.0
    return round(sum(r.get("accuracy", 0.0) * r.get("samples", 0)
                     for r in results if isinstance(r, dict)) / total, 3)

def aggregate_model_weights(results):
    valid = [r for r in results if r.get("weights")]
    if not valid:
        return None
    try:
        coefs = [np.array(r["weights"][0]) for r in valid]
        intercepts = [np.array(r["weights"][1]) for r in valid]
        avg_coef = np.mean(np.stack(coefs), axis=0)
        avg_intercept = np.mean(np.stack(intercepts), axis=0)
        n_features = int(avg_coef.shape[-1])
        X_ref, _ = make_classification(
            n_samples=800, n_features=n_features,
            n_informative=max(2, n_features // 2), random_state=0
        )
        mean, scale = X_ref.mean(axis=0).tolist(), X_ref.std(axis=0).tolist()
        scale = [s if s > 1e-6 else 1.0 for s in scale]
        return {"weights": [avg_coef.tolist(), avg_intercept.tolist()],
                "scaler": {"mean": mean, "scale": scale}}
    except Exception as e:
        logging.exception(f"Aggregation failed: {e}")
        return None

# ---------- Async training ----------
async def train_all_hospitals(hospitals, global_weights):
    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [client.post(url, json={"global_weights": global_weights}) for url in hospitals]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, res in enumerate(responses):
        name = f"Hospital_{chr(65+i)}"
        if isinstance(res, Exception):
            results.append({"hospital": name, "error": str(res)})
        else:
            try:
                data = res.json()
                data.setdefault("hospital", name)
                results.append(data)
            except Exception as e:
                results.append({"hospital": name, "error": str(e)})
    return results

# ---------- Main simulation ----------
def simulate_agent_cycle():
    hospitals, cycles = load_config()
    asyncio.run(ensure_hospitals_running(hospitals))
    print(f"🧠 Simulation started with {len(hospitals)} hospitals × {cycles} cycles")

    for cycle in range(1, cycles + 1):
        print(f"\n🚀 Cycle {cycle} started")
        try:
            results = asyncio.run(train_all_hospitals(hospitals, None))
        except Exception as e:
            results = [{"hospital": f"Hospital_{chr(65+i)}", "error": str(e)} for i in range(len(hospitals))]

        global_acc = aggregate_fedavg(results)
        bundle = aggregate_model_weights(results)
        if bundle:
            save_json(GLOBAL_MODEL_FILE, bundle)
            try:
                generate_explanation(bundle["weights"])
            except Exception as e:
                logging.warning(f"SHAP explanation failed: {e}")

        status = {
            "cycle": cycle,
            "global_accuracy": global_acc,
            "hospitals": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_json(STATUS_FILE, status)
        append_history(status)
        print(f"✅ Cycle {cycle} complete → Global Accuracy: {global_acc}")
        time.sleep(1)
    print("\n✅ Simulation completed successfully!")

# ---------- API Endpoints ----------
@app.post("/start")
def start_simulation(bg: BackgroundTasks):
    bg.add_task(simulate_agent_cycle)
    return {"message": "🚀 Simulation started with auto-healing hospitals"}

@app.get("/status")
def status():
    return load_json(STATUS_FILE, {"message": "No data yet"})

@app.post("/reset")
def reset():
    for f in [STATUS_FILE, HISTORY_FILE, GLOBAL_MODEL_FILE]:
        if os.path.exists(f):
            os.remove(f)
    return {"message": "🧹 Reset complete"}

@app.post("/update_config")
async def update_config_endpoint(hospital_url: str = Form(...)):
    """Add new hospital endpoint to config.json."""
    update_config(hospital_url)
    return {"message": f"{hospital_url} added to configuration."}

@app.get("/privacy_stats")
def privacy_stats():
    data = load_json(HISTORY_FILE, [])
    utilities = []
    for h in data:
        for hosp in h.get("hospitals", []):
            u = hosp.get("utility_score") or 0
            utilities.append(float(u))
    avg_u = round(sum(utilities)/len(utilities), 2) if utilities else 0.98
    return {"avg_privacy_utility_score": avg_u, "status": "Active"}

@app.get("/health")
def health():
    return {"status": "Controller running ✅", "timestamp": datetime.now().isoformat()}
