# backend/hospital_A.py
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ml_core.train_local import train_on_local_data
from datetime import datetime

app = FastAPI(title="🏥 Hospital A - Heart Disease Training Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
async def train_model(request: Request):
    """Train model on Hospital A’s local heart disease dataset."""
    payload = await request.json()
    global_weights = payload.get("global_weights", None)

    local_weights, accuracy, samples = train_on_local_data("heart_disease.csv", global_weights)
    result = {
        "hospital": "Hospital_A",
        "dataset": "heart_disease.csv",
        "accuracy": accuracy,
        "samples": samples,
        "weights": local_weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result

@app.get("/health")
def health_check():
    return {"status": "Hospital A active ✅"}
