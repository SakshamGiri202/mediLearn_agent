# backend/hospital_template.py
import sys, os, json, logging, argparse
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ------------------ Path Fix ------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data  # ✅ uses 4-value return

# ------------------ Argparse Setup ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Hospital name")
parser.add_argument("--dataset", required=True, help="Dataset file name")
parser.add_argument("--port", required=True, type=int, help="Port number")
args = parser.parse_args()

HOSPITAL_NAME = args.name
DATASET = args.dataset
PORT = args.port

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
    """Perform local training with DP + SHAP and return results."""
    try:
        payload = await request.json()
        global_weights = payload.get("global_weights")

        # ✅ Correct unpack for 4-value return
        local_weights, accuracy, samples, feature_names = train_on_local_data(DATASET, global_weights)

        response = {
            "hospital": HOSPITAL_NAME,
            "dataset": DATASET,
            "accuracy": accuracy,
            "samples": samples,
            "weights": local_weights,
            "feature_names": feature_names,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logging.info(f"{HOSPITAL_NAME}: Training complete (Acc={accuracy}, Samples={samples})")
        return response

    except Exception as e:
        logging.error(f"{HOSPITAL_NAME} training failed: {e}", exc_info=True)
        return {"error": f"{HOSPITAL_NAME} training failed: {str(e)}"}

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": f"{HOSPITAL_NAME} active ✅",
        "dataset": DATASET,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ------------------ Main ------------------
if __name__ == "__main__":
    import uvicorn
    logging.info(f"🚀 Starting {HOSPITAL_NAME} on port {PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
