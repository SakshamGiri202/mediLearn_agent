# backend/hospital_D.py
import sys, os, json, logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# ------------------ Path Fix ------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data

# ------------------ Hospital Config ------------------
HOSPITAL_NAME = "Hospital_D"
DATASET = "heart_disease.csv"
PORT = 8004

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
logging.info(f"🏥 {HOSPITAL_NAME} initialized with {DATASET}")

# ------------------ Routes ------------------
@app.post("/train")
async def train(request: Request):
    try:
        payload = await request.json()
        global_weights = payload.get("global_weights")

        # ✅ Fixed unpack to 4 values
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
    return {"status": f"{HOSPITAL_NAME} active ✅", "dataset": DATASET}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
