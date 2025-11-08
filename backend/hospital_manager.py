# backend/hospital_manager.py
"""
🏥 MediLearn Hospital Manager (Auto Dynamic Version)
- Automatically detects free port (8004–9000)
- Uses latest hospital_template.py for new nodes
- Auto-registers new hospitals in agent_config.json
- Starts each hospital automatically in a new console
"""

import os
import sys
import json
import shutil
import subprocess
import logging
import socket
from datetime import datetime
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# 1. Path setup
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

BACKEND_DIR = os.path.join(ROOT, "backend")
CONFIG_FILE = os.path.join(ROOT, "agent_config.json")
TEMPLATE_FILE = os.path.join(BACKEND_DIR, "hospital_template.py")
DATASET_DIR = os.path.join(ROOT, "ml_core", "dataset")

# Ensure backend/ and ml_core/ are real packages
for pkg in ("backend", "ml_core"):
    pkg_init = os.path.join(ROOT, pkg, "__init__.py")
    os.makedirs(os.path.dirname(pkg_init), exist_ok=True)
    if not os.path.exists(pkg_init):
        open(pkg_init, "a", encoding="utf-8").close()

# ------------------------------------------------------------------
# 2. FastAPI app + logging
# ------------------------------------------------------------------
app = FastAPI(title="🏥 MediLearn Hospital Manager")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = os.path.join(ROOT, "hospital_manager.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("Hospital Manager started. root=%s", ROOT)


# ------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------
def find_free_port(start=8004, end=9000):
    """Find first available port in given range."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("⚠️ No available port found in range 8004–9000.")


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def python_exec() -> str:
    """Ensure consistent Python path."""
    return sys.executable or "python"


def start_hospital(script_path, name, dataset, port):
    """Start hospital node in a new console."""
    cmd = [python_exec(), script_path, "--name", name, "--dataset", dataset, "--port", str(port)]
    try:
        if os.name == "nt":
            subprocess.Popen(cmd, cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd, cwd=ROOT)
        logging.info(f"🚀 Started {name} on port {port}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to start {name}: {e}")
        return False


# ------------------------------------------------------------------
# 4. API: Add Hospital (Dynamic)
# ------------------------------------------------------------------
@app.post("/add_hospital")
async def add_hospital(
    hospital_name: str = Form(...),
    dataset_name: str = Form(...),
    port: int | None = Form(None),
    autostart: bool = Form(True),
    file: UploadFile = None
):
    """Dynamically adds a new hospital node and launches it."""
    try:
        hn = hospital_name.strip().replace(" ", "_")
        script_name = f"hospital_{hn}.py"
        new_script_path = os.path.join(BACKEND_DIR, script_name)

        # Upload dataset if provided
        if file:
            dataset_path = os.path.join(DATASET_DIR, dataset_name)
            os.makedirs(DATASET_DIR, exist_ok=True)
            with open(dataset_path, "wb") as f:
                f.write(await file.read())
            logging.info(f"📁 Custom dataset uploaded: {dataset_path}")

        # Use existing dataset if no file uploaded
        elif not os.path.exists(os.path.join(DATASET_DIR, dataset_name)):
            raise HTTPException(status_code=400, detail=f"Dataset '{dataset_name}' not found or not uploaded.")

        # Auto-assign free port if not provided
        port = port or find_free_port()

        # Copy latest hospital template
        if not os.path.exists(TEMPLATE_FILE):
            raise HTTPException(status_code=500, detail="Template file missing.")
        shutil.copy(TEMPLATE_FILE, new_script_path)
        logging.info(f"🧩 New hospital script created: {new_script_path}")

        # Update config
        cfg = load_config()
        hospitals = cfg.get("hospitals", [])
        endpoint = f"http://127.0.0.1:{port}/train"
        if endpoint not in hospitals:
            hospitals.append(endpoint)
        cfg["hospitals"] = hospitals
        save_config(cfg)
        logging.info(f"🔗 Added hospital endpoint to config: {endpoint}")

        # Auto-start
        started = False
        if autostart:
            started = start_hospital(new_script_path, hn, dataset_name, port)

        return JSONResponse({
            "message": f"✅ Hospital '{hn}' created successfully",
            "endpoint": endpoint,
            "autostarted": started,
            "dataset": dataset_name,
            "port": port,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        logging.error(f"Error adding hospital: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------------
# 5. List & Remove
# ------------------------------------------------------------------
@app.get("/list_hospitals")
def list_hospitals():
    cfg = load_config()
    return {"registered_hospitals": cfg.get("hospitals", []), "count": len(cfg.get("hospitals", []))}


@app.post("/remove_hospital")
async def remove_hospital(hospital_name: str = Form(...)):
    hn = hospital_name.strip().replace(" ", "_")
    script_path = os.path.join(BACKEND_DIR, f"hospital_{hn}.py")

    if os.path.exists(script_path):
        os.remove(script_path)

    cfg = load_config()
    hospitals = [h for h in cfg.get("hospitals", []) if hn.lower() not in h.lower()]
    cfg["hospitals"] = hospitals
    save_config(cfg)

    return {"message": f"🗑️ {hn} removed successfully", "script_deleted": not os.path.exists(script_path)}


# ------------------------------------------------------------------
# 6. Root / Health
# ------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "🏥 Hospital Manager Active",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "example_add": "/add_hospital (Form: hospital_name, dataset_name, autostart)",
        "example_list": "/list_hospitals"
    }


# ------------------------------------------------------------------
# 7. Run (no reload)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Hospital Manager on http://127.0.0.1:8600 (auto dynamic mode)")
    uvicorn.run("backend.hospital_manager:app", host="127.0.0.1", port=8600, reload=False)
