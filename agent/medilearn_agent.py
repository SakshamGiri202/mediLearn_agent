# medilearn_agent.py
import json
import time
from rich.console import Console
from controller import train_at_hospital, aggregate_results

console = Console()

def load_config(path="agent/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def run_agent():
    config = load_config()
    hospitals = config["hospitals"]
    aggregation_endpoint = config["aggregation_endpoint"]
    cycles = config.get("cycles", 3)

    console.print("[bold magenta]🤖 MediLearn Agent Started[/bold magenta]")
    console.print(f"[yellow]Total Hospitals:[/yellow] {len(hospitals)} | [yellow]Cycles:[/yellow] {cycles}\n")

    for cycle in range(1, cycles + 1):
        console.rule(f"[bold blue]Round {cycle} / {cycles}[/bold blue]")
        round_results = []

        for hospital in hospitals:
            result = train_at_hospital(hospital)
            if result:
                round_results.append(result)
            time.sleep(1)  # Small delay to simulate travel time

        # Once all hospitals are trained, aggregate results
        if round_results:
            console.print("[cyan]🔄 Aggregating models from hospitals...[/cyan]")
            global_result = aggregate_results(round_results, aggregation_endpoint)
            if global_result:
                console.print(f"[green]🌟 Global accuracy after round {cycle}: {global_result.get('global_accuracy')}[/green]")
        else:
            console.print("[red]No valid results received this round.[/red]")

        time.sleep(2)  # Small pause before next cycle

    console.print("\n[bold green]✅ All training cycles complete![/bold green]")

if __name__ == "__main__":
    run_agent()
