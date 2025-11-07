# controller.py
import requests
from rich.console import Console

console = Console()

def train_at_hospital(hospital_url: str):
    """Send a POST request to hospital to start local training."""
    try:
        console.print(f"[cyan]🚀 Sending training request to {hospital_url}...[/cyan]")
        response = requests.post(hospital_url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]✅ Training complete at {hospital_url} | Accuracy: {data.get('accuracy')}[/green]")
            return data
        else:
            console.print(f"[red]❌ Error from {hospital_url}: {response.status_code}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]⚠️ Failed to reach {hospital_url}: {e}[/red]")
        return None


def aggregate_results(results: list, aggregation_endpoint: str):
    """Send all model results to aggregator."""
    try:
        payload = {"results": results}
        response = requests.post(aggregation_endpoint, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            console.print(f"[yellow]🌍 Aggregation complete | Global Accuracy: {data.get('global_accuracy')}[/yellow]")
            return data
        else:
            console.print(f"[red]❌ Aggregation failed: {response.status_code}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]⚠️ Aggregation error: {e}[/red]")
        return None
