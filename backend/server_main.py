# backend/server_main.py
import logging, json, os, asyncio, time, httpx
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from ml_core.explain_model import generate_explanation
from fastapi.responses import FileResponse

app = FastAPI(title="🧠 MediLearn Controller (FedAvg + Async Upgrade)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
logging.info("=== MediLearn Controller Initialized (FedAvg + Async Upgrade) ===")

# -----------------------------
# ⚙️ Helper Functions
# -----------------------------
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def append_history(entry):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    history.append(entry)
    save_json(HISTORY_FILE, history)

def load_config():
    """Load dynamic configuration from JSON."""
    default_hospitals = [
        "http://127.0.0.1:8001/train",
        "http://127.0.0.1:8002/train",
        "http://127.0.0.1:8003/train"
    ]
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            hospitals = cfg.get("hospitals", default_hospitals)
            cycles = cfg.get("cycles", 3)
            return hospitals, cycles
        except:
            pass
    return default_hospitals, 3

def aggregate_fedavg(results):
    total = sum(r.get("samples", 0) for r in results if r.get("samples"))
    if total == 0:
        return 0.0
    weighted = sum(r["accuracy"] * r["samples"] for r in results if r.get("samples"))
    return round(weighted / total, 3)

def aggregate_model_weights(results):
    valid = [r for r in results if r.get("weights")]
    if not valid:
        return None
    coefs = [np.array(r["weights"][0]) for r in valid]
    intercepts = [np.array(r["weights"][1]) for r in valid]
    avg_coef = np.mean(coefs, axis=0).tolist()
    avg_intercept = np.mean(intercepts, axis=0).tolist()
    return [avg_coef, avg_intercept]

def load_global_model():
    if os.path.exists(GLOBAL_MODEL_FILE):
        with open(GLOBAL_MODEL_FILE, "r") as f:
            return json.load(f)
    return None

# -----------------------------
# ⚙️ Async Training Logic
# -----------------------------
async def train_all_hospitals(hospitals, global_weights):
    async with httpx.AsyncClient() as client:
        tasks = [client.post(url, json={"global_weights": global_weights}, timeout=15) for url in hospitals]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, res in enumerate(responses):
        if isinstance(res, Exception):
            results.append({"hospital": f"Hospital_{chr(65+i)}", "error": str(res)})
        else:
            try:
                results.append(res.json())
            except:
                results.append({"hospital": f"Hospital_{chr(65+i)}", "error": "Invalid response"})
    return results

# -----------------------------
# 🔁 Main Simulation
# -----------------------------
def simulate_agent_cycle():
    hospitals, cycles = load_config()
    global_weights = load_global_model()
    print(f"🧠 MediLearn Simulation Started → Hospitals: {len(hospitals)} | Cycles: {cycles}")
    logging.info("Simulation started.")

    global_data = {"hospitals": [], "global_accuracy": 0.0, "cycle": 0}

    for cycle in range(1, cycles + 1):
        print(f"\n🚀 Cycle {cycle} started...")
        try:
            results = asyncio.run(train_all_hospitals(hospitals, global_weights))
        except Exception as e:
            logging.error(f"Cycle {cycle} failed: {e}")
            continue

        global_accuracy = aggregate_fedavg(results)
        new_global_weights = aggregate_model_weights(results)
        if new_global_weights:
            save_json(GLOBAL_MODEL_FILE, new_global_weights)

            try:
                generate_explanation(new_global_weights)   # 👈 Add this line
            except Exception as e:
                logging.warning(f"SHAP explanation skipped: {e}")

        global_data.update({
            "cycle": cycle,
            "global_accuracy": global_accuracy,
            "hospitals": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        save_json(STATUS_FILE, global_data)
        append_history(global_data)
        logging.info(f"Cycle {cycle} complete → Global Accuracy: {global_accuracy}")

        global_weights = new_global_weights
        time.sleep(1)

    print("\n✅ Simulation completed successfully!")
    logging.info("Simulation completed successfully.")

# -----------------------------
# 🌐 API ROUTES
# -----------------------------
@app.post("/aggregate")
async def aggregate_endpoint(request: Request):
    payload = await request.json()
    results = payload.get("results", [])
    if not results:
        raise HTTPException(status_code=400, detail="No results provided.")

    global_accuracy = aggregate_fedavg(results)
    new_global_weights = aggregate_model_weights(results)
    if new_global_weights:
        save_json(GLOBAL_MODEL_FILE, new_global_weights)

    status = {
        "cycle": len(results),
        "global_accuracy": global_accuracy,
        "hospitals": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(STATUS_FILE, status)
    append_history(status)
    logging.info(f"Aggregation complete → Global accuracy: {global_accuracy}")
    return JSONResponse({"message": "Aggregation complete", "global_accuracy": global_accuracy})

@app.post("/start")
def start_simulation(background_tasks: BackgroundTasks):
    background_tasks.add_task(simulate_agent_cycle)
    return {"message": "🚀 Simulation started (Async Aggregator)"}

@app.get("/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {"message": "No training data yet. Run agent to start."}

@app.get("/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

@app.get("/global_model")
def get_global_model():
    if os.path.exists(GLOBAL_MODEL_FILE):
        with open(GLOBAL_MODEL_FILE, "r") as f:
            return json.load(f)
    return {"message": "No global model yet."}

@app.get("/explain")
def explain_global_model():
    """Generate and return global feature importance plot."""
    if not os.path.exists(GLOBAL_MODEL_FILE):
        return {"message": "Error: No global model trained yet."}
    with open(GLOBAL_MODEL_FILE, "r") as f:
        weights = json.load(f)
    path = generate_explanation(weights)
    return FileResponse(path, media_type="image/png", filename="feature_importance.png")

@app.get("/privacy_stats")
def privacy_stats():
    """Return system privacy–utility performance summary."""
    if not os.path.exists(HISTORY_FILE):
        return {"message": "No training history yet."}
    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)
    utilities = [h.get("hospitals", [{}])[0].get("utility_score", 0) for h in data if h.get("hospitals")]
    avg_utility = round(sum(utilities) / len(utilities), 2) if utilities else 0
    return {"avg_privacy_utility_score": avg_utility, "noise_sigma": 0.02, "status": "Active"}

@app.get("/stream")
async def stream_status():
    async def event_stream():
        while True:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, "r") as f:
                    yield f"data: {f.read()}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "MediLearn Controller Active ✅", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/reset")
def reset():
    for p in [STATUS_FILE, HISTORY_FILE, GLOBAL_MODEL_FILE]:
        if os.path.exists(p):
            os.remove(p)
    logging.info("Controller reset requested.")
    return {"message": "🧹 Reset complete"}

@app.get("/")
def home():
    return {"status": "MediLearn Aggregator 🩺", "version": "4.0 (Async Upgrade FedAvg)"}
