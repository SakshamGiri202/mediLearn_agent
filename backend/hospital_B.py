# backend/hospital_B.py
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ml_core.train_local import train_on_local_data
from datetime import datetime

app = FastAPI(title="🏥 Hospital B - Diabetes Training Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
async def train_model(request: Request):
    """Train model on Hospital B’s local diabetes dataset."""
    payload = await request.json()
    global_weights = payload.get("global_weights", None)

    local_weights, accuracy, samples = train_on_local_data("diabetes.csv", global_weights)
    result = {
        "hospital": "Hospital_B",
        "dataset": "diabetes.csv",
        "accuracy": accuracy,
        "samples": samples,
        "weights": local_weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result

@app.get("/health")
def health_check():
    return {"status": "Hospital B active ✅"}
