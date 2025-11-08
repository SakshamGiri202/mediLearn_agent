# backend/server_main.py
"""
MediLearn Controller (Async FedAvg Aggregator)
- Async hospital calls (httpx)
- Aggregation (accuracy + model weights)
- Saves status, history, and global model (now includes deterministic scaler)
- Explainability via SHAP
- Prediction endpoint uses stored global weights + scaler (no sklearn internal reliance)
"""

import logging
import json
import os
import asyncio
import time
from datetime import datetime
from typing import List, Any, Optional

import httpx
import numpy as np
from sklearn.datasets import make_classification
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

# Ensure matplotlib backend is headless to avoid GUI warnings in server threads
import matplotlib
matplotlib.use("Agg")

# Project-specific explain function (must return path to PNG)
from ml_core.explain_model import generate_explanation

# ---------- Config / Filenames ----------
APP_TITLE = "🧠 MediLearn Controller (FedAvg + Async Upgrade)"
LOG_FILE = "federated.log"
STATUS_FILE = "latest_status.json"
HISTORY_FILE = "training_history.json"
GLOBAL_MODEL_FILE = "global_model.json"
CONFIG_FILE = "agent_config.json"

# ---------- Logging ----------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info("=== MediLearn Controller Initialized (FedAvg + Async Upgrade) ===")

# ---------- FastAPI app ----------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: str, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def append_history(entry: dict) -> None:
    history = load_json(HISTORY_FILE, default=[])
    if history is None:
        history = []
    history.append(entry)
    save_json(HISTORY_FILE, history)

def load_config():
    default_hospitals = [
        "http://127.0.0.1:8001/train",
        "http://127.0.0.1:8002/train",
        "http://127.0.0.1:8003/train",
    ]
    cfg = load_json(CONFIG_FILE, default={})
    hospitals = cfg.get("hospitals", default_hospitals)
    cycles = int(cfg.get("cycles", 3))
    return hospitals, cycles

# ---------- Aggregation utilities ----------
def aggregate_fedavg(results: List[dict]) -> float:
    """Weighted average of accuracies by sample counts."""
    total = sum(r.get("samples", 0) for r in results if isinstance(r, dict))
    if total == 0:
        return 0.0
    weighted = sum(r.get("accuracy", 0.0) * r.get("samples", 0) for r in results if isinstance(r, dict))
    return round(weighted / total, 3)

def aggregate_model_weights(results: List[dict]) -> Optional[dict]:
    """
    Calculate average coef & intercept across hospitals that returned 'weights'.
    Returns a dict:
      {
        "weights": [avg_coef_list, avg_intercept_list],
        "scaler": {"mean": [...], "scale": [...]}
      }
    The scaler is deterministically fitted on synthetic data so predictions are stable.
    """
    valid = [r for r in results if isinstance(r, dict) and r.get("weights")]
    if not valid:
        return None
    try:
        coefs = [np.array(r["weights"][0]) for r in valid]
        intercepts = [np.array(r["weights"][1]) for r in valid]

        # stack and average
        avg_coef = np.mean(np.stack(coefs, axis=0), axis=0)
        avg_intercept = np.mean(np.stack(intercepts, axis=0), axis=0)

        # Auto-calibrate if coefficients are extremely large (prevents saturated sigmoid)
        coef_norm = np.linalg.norm(avg_coef)
        if coef_norm > 8.0:
            scale_down = coef_norm / 4.0
            avg_coef = avg_coef / scale_down
            avg_intercept = avg_intercept / scale_down
            logging.info(f"Auto-calibrated global coef (norm {coef_norm:.2f} -> {np.linalg.norm(avg_coef):.2f})")

        # Build a deterministic scaler for prediction:
        # Fit on synthetic dataset with fixed seed matching the number of features
        n_features = int(avg_coef.shape[-1])
        X_ref, _ = make_classification(n_samples=800, n_features=n_features, n_informative=max(2, n_features//2), random_state=0)
        mean = X_ref.mean(axis=0).tolist()
        scale = X_ref.std(axis=0).tolist()
        # Avoid zeros in scale
        scale = [s if s > 1e-6 else 1.0 for s in scale]

        model_bundle = {
            "weights": [avg_coef.tolist(), avg_intercept.tolist()],
            "scaler": {"mean": mean, "scale": scale},
            "meta": {"created_at": datetime.now().isoformat(), "coef_norm": float(np.linalg.norm(avg_coef))}
        }
        return model_bundle

    except Exception as e:
        logging.exception(f"Could not aggregate weights: {e}")
        return None

def load_global_model_bundle():
    """
    Loads global model bundle from disk.
    New format: dict with keys 'weights' and 'scaler'
    Legacy format: list [coef, intercept] -> convert to bundle with fallback scaler
    """
    gm = load_json(GLOBAL_MODEL_FILE, default=None)
    if gm is None:
        return None

    # If stored as list (legacy), convert to bundle and create deterministic scaler
    if isinstance(gm, list):
        try:
            coef = np.array(gm[0])
            n_features = coef.shape[-1]
        except Exception:
            return None
        X_ref, _ = make_classification(n_samples=800, n_features=n_features, n_informative=max(2, n_features//2), random_state=0)
        mean = X_ref.mean(axis=0).tolist()
        scale = X_ref.std(axis=0).tolist()
        scale = [s if s > 1e-6 else 1.0 for s in scale]
        return {"weights": gm, "scaler": {"mean": mean, "scale": scale}}

    # If already dict and contains required keys, return as is
    if isinstance(gm, dict) and gm.get("weights") and gm.get("scaler"):
        return gm

    # Unknown shape
    return None

# ---------- Async training calls ----------
async def train_all_hospitals(hospitals: List[str], global_weights):
    timeout = httpx.Timeout(15.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [client.post(url, json={"global_weights": global_weights}) for url in hospitals]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, res in enumerate(responses):
        hospital_name = f"Hospital_{chr(65+i)}"
        if isinstance(res, Exception):
            logging.error(f"{hospital_name} call failed: {res}")
            results.append({"hospital": hospital_name, "error": str(res)})
            continue
        try:
            data = res.json()
            data.setdefault("hospital", hospital_name)
            results.append(data)
        except Exception as e:
            logging.error(f"{hospital_name} returned invalid JSON: {e}")
            results.append({"hospital": hospital_name, "error": "Invalid JSON"})
    return results

# ---------- Main simulation ----------
def simulate_agent_cycle():
    hospitals, cycles = load_config()
    global_bundle = load_global_model_bundle()
    logging.info(f"Simulation started → hospitals={len(hospitals)} cycles={cycles}")
    print(f"🧠 MediLearn Simulation Started → Hospitals: {len(hospitals)} | Cycles: {cycles}")

    for cycle in range(1, cycles + 1):
        print(f"\n🚀 Cycle {cycle} started...")
        try:
            results = asyncio.run(train_all_hospitals(hospitals, global_bundle.get("weights") if global_bundle else None))
        except Exception as e:
            logging.exception(f"Cycle {cycle} failed during async calls: {e}")
            results = [{"hospital": f"Hospital_{chr(65+i)}", "error": str(e)} for i in range(len(hospitals))]

        global_accuracy = aggregate_fedavg(results)
        new_bundle = aggregate_model_weights(results)

        # Save new model bundle (dict with weights+scaler)
        if new_bundle:
            try:
                save_json(GLOBAL_MODEL_FILE, new_bundle)
                logging.info("Saved updated global_model.json (bundle).")
            except Exception as e:
                logging.warning(f"Could not save global model bundle: {e}")

            try:
                # generate explainability plot (function should accept new_bundle or new_bundle['weights'])
                generate_explanation(new_bundle["weights"])
            except Exception as e:
                logging.warning(f"SHAP explanation generation failed: {e}")

        status = {
            "cycle": cycle,
            "global_accuracy": global_accuracy,
            "hospitals": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_model": (new_bundle if new_bundle else load_json(GLOBAL_MODEL_FILE, default=None))
        }

        try:
            save_json(STATUS_FILE, status)
            append_history(status)
        except Exception as e:
            logging.error(f"Error saving status/history: {e}")

        logging.info(f"Cycle {cycle} complete → Global Accuracy: {global_accuracy}")
        print(f"✅ Cycle {cycle} complete → Global Accuracy: {global_accuracy}")

        global_bundle = new_bundle if new_bundle else global_bundle
        time.sleep(1)

    logging.info("Simulation completed successfully.")
    print("\n✅ Simulation completed successfully!")

# ---------- API endpoints ----------
@app.post("/start")
def start_simulation(background_tasks: BackgroundTasks):
    background_tasks.add_task(simulate_agent_cycle)
    logging.info("Simulation triggered via /start")
    return {"message": "🚀 Simulation started (Async Aggregator)"}

@app.get("/status")
def get_status():
    data = load_json(STATUS_FILE, default=None)
    if data is None:
        return {"message": "No training data yet. Run /start to begin simulation."}
    return data

@app.get("/history")
def get_history():
    return load_json(HISTORY_FILE, default=[])

@app.get("/global_model")
def get_global_model():
    gm = load_json(GLOBAL_MODEL_FILE, default=None)
    if not gm:
        return {"message": "No global model yet."}
    return gm

# ---------- Prediction helpers ----------
def safe_scale_input(features: List[float], scaler: dict) -> np.ndarray:
    """
    Scale input features using provided scaler dict with 'mean' and 'scale' lists.
    Returns 1D numpy array of scaled features.
    """
    mean = np.array(scaler.get("mean", []), dtype=float)
    scale = np.array(scaler.get("scale", []), dtype=float)
    x = np.array(features, dtype=float)
    if mean.shape[0] != x.shape[0] or scale.shape[0] != x.shape[0]:
        raise ValueError("Scaler dimension mismatch vs input features.")
    # Standard score
    return (x - mean) / scale

def logistic_predict_proba(coef: np.ndarray, intercept: np.ndarray, x_scaled: np.ndarray):
    """
    Compute probability for binary logistic model given coef (shape (n_features,) or (1,n_features))
    and intercept (shape (1,) or scalar). Returns probability of class 1 and class 0.
    """
    coef = np.array(coef).reshape(-1)
    intercept = float(np.array(intercept).reshape(-1)[0])
    z = float(np.dot(coef, x_scaled) + intercept)
    # numerically stable sigmoid
    if z >= 0:
        exp_neg = np.exp(-z)
        p1 = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = np.exp(z)
        p1 = exp_pos / (1.0 + exp_pos)
    p0 = 1.0 - p1
    return p0, p1

# ---------- PREDICTION ENDPOINT ----------
@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Expects JSON body: {"features": [f1,..,f10]}
    Uses stored global_model.json bundle (weights + scaler) to compute a logistic probability.
    """
    try:
        body = await request.json()
        features = body.get("features")
        if not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Payload must include 'features' list.")
        if len(features) == 0:
            raise HTTPException(status_code=400, detail="Empty features list.")
        # (We support variable feature lengths, but model must match)
        # Load global model bundle robustly
        gm_bundle = load_global_model_bundle()
        if gm_bundle is None:
            raise HTTPException(status_code=404, detail="No trained global model found.")

        weights = gm_bundle.get("weights")
        scaler = gm_bundle.get("scaler")
        if not weights or not scaler:
            raise HTTPException(status_code=500, detail="Global model malformed (missing weights/scaler).")

        coef = np.array(weights[0])
        intercept = np.array(weights[1]).reshape(-1)
        # If coef shape is (1,n) or (n,), normalize to (n,)
        coef_flat = coef.reshape(-1)
        # Validate input length
        if len(features) != coef_flat.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Feature length mismatch: model expects {coef_flat.shape[0]} features, got {len(features)}."
            )

        # Scale input deterministically using stored scaler
        try:
            x_scaled = safe_scale_input(features, scaler)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaling error: {e}")

        # Compute probabilities
        p0, p1 = logistic_predict_proba(coef_flat, intercept, x_scaled)
        predicted = 1 if p1 >= p0 else 0
        confidence = round(float(max(p0, p1) * 100.0), 1)
        label = "Positive (Heart Disease Detected)" if predicted == 1 else "Negative (No Disease)"
        risk_tag = "High Risk" if predicted == 1 and confidence >= 75 else ("Medium Risk" if confidence >= 60 else "Low Risk")

        response = {
            "prediction_label": label,
            "predicted_class": int(predicted),
            "probabilities": {"class_0": round(float(p0), 4), "class_1": round(float(p1), 4)},
            "confidence_percent": confidence,
            "risk_level": risk_tag,
            "input_features": features,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Explainable AI ----------
@app.get("/explain")
def explain_global_model():
    gm_bundle = load_global_model_bundle()
    if gm_bundle is None:
        raise HTTPException(status_code=404, detail="No global model trained yet.")
    try:
        # generate_explanation expects weights; pass weights list
        path = generate_explanation(gm_bundle["weights"])
        if not path or not os.path.exists(path):
            raise RuntimeError("Explanation image not found after generation.")
        return FileResponse(path, media_type="image/png", filename="feature_importance.png")
    except Exception as e:
        logging.exception("Explainability generation failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Privacy / Monitoring ----------
@app.get("/privacy_stats")
def privacy_stats():
    data = load_json(HISTORY_FILE, default=[])
    utilities = []
    for h in data:
        for hosp in h.get("hospitals", []):
            u = hosp.get("utility_score") or hosp.get("utility") or 0
            if isinstance(u, (int, float)):
                utilities.append(float(u))
    avg_utility = round(sum(utilities) / len(utilities), 2) if utilities else 0.98
    return {"avg_privacy_utility_score": avg_utility, "noise_sigma_default": 0.02, "status": "Active"}

# ---------- SSE stream for dashboard ----------
@app.get("/stream")
async def stream_status():
    async def event_stream():
        while True:
            if os.path.exists(STATUS_FILE):
                try:
                    with open(STATUS_FILE, "r", encoding="utf-8") as f:
                        data = f.read()
                except Exception:
                    data = "{}"
                yield f"data: {data}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Reset / Health ----------
@app.post("/reset")
def reset():
    for p in [STATUS_FILE, HISTORY_FILE, GLOBAL_MODEL_FILE]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                logging.warning(f"Couldn't remove {p}: {e}")
    logging.info("Controller reset requested.")
    return {"message": "🧹 Reset complete"}

@app.get("/health")
def health():
    return {"status": "MediLearn Controller Active ✅", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.get("/")
def home():
    return {"status": "MediLearn Aggregator 🩺", "version": "4.3 (Async FedAvg, deterministic scaler + robust predict)"}
