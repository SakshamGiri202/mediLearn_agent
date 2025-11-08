# backend/hospital_C.py
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ml_core.train_local import train_on_local_data
from datetime import datetime

app = FastAPI(title="🏥 Hospital C - Stroke Training Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/train")
async def train_model(request: Request):
    """Train model on Hospital C’s local stroke dataset."""
    payload = await request.json()
    global_weights = payload.get("global_weights", None)

    local_weights, accuracy, samples = train_on_local_data("stroke.csv", global_weights)
    result = {
        "hospital": "Hospital_C",
        "dataset": "stroke.csv",
        "accuracy": accuracy,
        "samples": samples,
        "weights": local_weights,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return result

@app.get("/health")
def health_check():
    return {"status": "Hospital C active ✅"}
