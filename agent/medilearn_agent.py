# agent/medilearn_agent.py
import json
import time
from rich.console import Console
from agent.controller import train_at_hospital, aggregate_results

console = Console()

# ---------------------------------------------------------
# ⚙️ CONFIGURATION LOADER
# ---------------------------------------------------------
def load_config(path="agent/config.json"):
    """Loads configuration for agent: hospitals, aggregator, and cycle count."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# 🤖 MEDILEARN FEDERATED AGENT
# ---------------------------------------------------------
def run_agent():
    cfg = load_config()
    hospitals = cfg.get("hospitals", [])
    aggregation_endpoint = cfg.get("aggregation_endpoint", "http://127.0.0.1:8000/aggregate")
    cycles = cfg.get("cycles", 3)

    console.rule("[bold magenta]🤖 MediLearn Agent Started[/bold magenta]")
    console.print(f"[yellow]Hospitals:[/yellow] {len(hospitals)} | [yellow]Rounds:[/yellow] {cycles}\n")

    last_global_accuracy = None

    for cycle in range(1, cycles + 1):
        console.rule(f"[bold blue]🌀 Round {cycle} / {cycles}[/bold blue]")
        round_results = []

        # 1️⃣ Send global model → hospitals train locally
        for hospital in hospitals:
            result = train_at_hospital(hospital)
            if result:
                round_results.append(result)
            else:
                console.print(f"[red]⚠️ Skipping {hospital} (no result).[/red]")
            time.sleep(1)  # pacing delay (for realism)

        # 2️⃣ Aggregate models into global model
        if round_results:
            console.print("[cyan]🔄 Aggregating updated hospital models...[/cyan]")
            global_result = aggregate_results(round_results, aggregation_endpoint)

            if global_result:
                global_acc = global_result.get("global_accuracy", 0)
                console.print(f"[green]🌟 Global accuracy after Round {cycle}: {global_acc}[/green]")

                if last_global_accuracy is not None:
                    delta = round(global_acc - last_global_accuracy, 3)
                    trend = "📈 Improvement" if delta > 0 else ("⚖️ Stable" if delta == 0 else "📉 Drop")
                    console.print(f"[white]Change vs last round:[/white] {delta:+.3f} → {trend}")

                last_global_accuracy = global_acc
            else:
                console.print("[red]❌ Aggregation failed this round.[/red]")
        else:
            console.print("[red]🚫 No valid hospital results received this round.[/red]")

        # 3️⃣ Wait before next federated round
        console.print("[dim]🕐 Waiting before next round...[/dim]")
        time.sleep(2)

    console.rule("[bold green]✅ All training cycles complete![/bold green]")
    console.print(f"[bold cyan]Final Global Accuracy:[/bold cyan] {last_global_accuracy}")
    console.print("[dim]All global models and logs saved locally for next run.[/dim]")


# ---------------------------------------------------------
# 🧩 MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    run_agent()
