import sys, os, json, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Fix imports so we can access ml_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data

# ------------------ FastAPI Setup ------------------
app = FastAPI(title="Hospital A Local Training Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Logging ------------------
logging.basicConfig(
    filename="hospital_A.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATASET = "heart_disease.csv"

# ------------------ Routes ------------------
@app.post("/train")
async def train(request: Request):
    """Receive global weights and train on local dataset."""
    try:
        payload = await request.json()
        global_weights = payload.get("global_weights")

        weights, accuracy, samples = train_on_local_data(DATASET, global_weights)

        # ✅ Build clean JSON-safe response
        response = {
            "hospital": "Hospital_A",
            "accuracy": accuracy,
            "samples": samples,
            "weights": weights,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logging.info(f"Hospital A → Training done. Accuracy={accuracy}, Samples={samples}")
        return response

    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        return {"error": f"Training failed: {str(e)}"}


@app.get("/health")
def health():
    """Simple check for hospital status."""
    return {"status": "Hospital A active ✅", "dataset": DATASET}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
