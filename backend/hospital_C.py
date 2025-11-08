import sys, os, json, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# ------------------ Path Fix ------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data  # ✅ DP + SHAP version

# ------------------ Hospital Identity ------------------
os.environ["HOSPITAL_NAME"] = "Hospital_C"
HOSPITAL_NAME = os.getenv("HOSPITAL_NAME")
DATASET = "stroke.csv"

# ------------------ FastAPI Setup ------------------
app = FastAPI(title=f"{HOSPITAL_NAME} Local Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Logging ------------------
LOG_FILE = f"{HOSPITAL_NAME}.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.info(f"🏥 {HOSPITAL_NAME} initialized on dataset {DATASET}")

# ------------------ Routes ------------------
@app.post("/train")
async def train(request: Request):
    """Receive global weights, perform local training with DP + SHAP."""
    try:
        payload = await request.json()
        global_weights = payload.get("global_weights")

        # ✅ Local training
        weights, accuracy, samples, feature_names = train_on_local_data(DATASET, global_weights)

        response = {
            "hospital": HOSPITAL_NAME,
            "dataset": DATASET,
            "accuracy": accuracy,
            "samples": samples,
            "weights": weights,
            "feature_names": feature_names,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logging.info(f"{HOSPITAL_NAME}: Training complete. Acc={accuracy}, Samples={samples}")
        return response

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return {"error": f"{HOSPITAL_NAME} training failed: {str(e)}"}

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": f"{HOSPITAL_NAME} active ✅",
        "dataset": DATASET,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ------------------ Main Run ------------------
if __name__ == "__main__":
    import uvicorn
    port = 8003
    logging.info(f"🚀 Starting {HOSPITAL_NAME} server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
