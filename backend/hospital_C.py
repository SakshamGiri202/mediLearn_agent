import sys, os, json, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_core.train_local import train_on_local_data

app = FastAPI(title="Hospital C Local Training Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    filename="hospital_C.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DATASET = "stroke.csv"

@app.post("/train")
async def train(request: Request):
    try:
        payload = await request.json()
        global_weights = payload.get("global_weights")

        weights, accuracy, samples = train_on_local_data(DATASET, global_weights)
        response = {
            "hospital": "Hospital_C",
            "accuracy": accuracy,
            "samples": samples,
            "weights": weights,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logging.info(f"Hospital C → Training done. Accuracy={accuracy}")
        return response

    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        return {"error": f"Training failed: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "Hospital C active ✅", "dataset": DATASET}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
