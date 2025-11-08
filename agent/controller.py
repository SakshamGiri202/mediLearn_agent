# agent/controller.py
import os
import json
import requests
from rich.console import Console

console = Console()

# Path to store & reuse global model weights locally for agent
GLOBAL_MODEL_PATH = "agent/latest_global_model.json"


# ---------------------------------------------------------
# ⚙️ UTILITIES FOR GLOBAL MODEL MANAGEMENT
# ---------------------------------------------------------
def load_global_weights():
    """Load saved global model weights from previous round."""
    if os.path.exists(GLOBAL_MODEL_PATH):
        with open(GLOBAL_MODEL_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                console.print("[red]⚠️ Corrupted global model file. Starting fresh.[/red]")
                return None
    return None


def save_global_weights(weights):
    """Save global model weights locally for the next round."""
    if weights:
        with open(GLOBAL_MODEL_PATH, "w") as f:
            json.dump(weights, f, indent=2)
        console.print("[magenta]💾 Saved updated global model for next round.[/magenta]")


# ---------------------------------------------------------
# 🏥 HOSPITAL TRAINING
# ---------------------------------------------------------
def train_at_hospital(hospital_url: str, timeout: int = 25):
    """
    Sends current global weights to a hospital for local training.
    Returns hospital results (accuracy, weights, etc.).
    """
    try:
        # Load previously aggregated global model (if any)
        global_weights = load_global_weights()
        payload = {"global_weights": global_weights}

        console.print(f"[cyan]🚀 Sending training request to {hospital_url}...[/cyan]")
        response = requests.post(hospital_url, json=payload, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            hospital_name = data.get("hospital", hospital_url)
            console.print(f"[green]✅ {hospital_name} | Accuracy: {data.get('accuracy')}[/green]")
            return data
        else:
            console.print(f"[red]❌ Error from {hospital_url}: {response.status_code} | {response.text}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]⚠️ Failed to reach {hospital_url}: {e}[/red]")
        return None


# ---------------------------------------------------------
# 🌍 AGGREGATION
# ---------------------------------------------------------
def aggregate_results(results: list, aggregation_endpoint: str, timeout: int = 30):
    """
    Sends all hospital training results to controller for FedAvg aggregation.
    Saves and returns updated global model for next round.
    """
    try:
        payload = {"results": results}
        response = requests.post(aggregation_endpoint, json=payload, timeout=timeout)

        if response.status_code == 200:
            data = response.json()

            # Log global accuracy
            global_acc = data.get("global_accuracy", None)
            console.print(f"[yellow]🌍 Aggregation complete | Global Accuracy: {global_acc}[/yellow]")

            # If controller returned updated model, save it
            if "global_model" in data and data["global_model"]:
                save_global_weights(data["global_model"])

            return data

        else:
            console.print(f"[red]❌ Aggregation failed: {response.status_code} | {response.text}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]⚠️ Aggregation error: {e}[/red]")
        return None
