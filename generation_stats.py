import json
import os
from collections import defaultdict
import requests

TASK_ERROR_LOG_PATH = "D:\\ai_agent\\server\\task_error_log.json"

def call_ollama(prompt: str, model_name: str) -> str:
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=600,
    )
    res.raise_for_status()
    data = res.json()
    return data["message"]["content"]


# --- core_logic ---
def load_error_history() -> list:
    """Load error history from JSON file."""
    try:
        with open(TASK_ERROR_LOG_PATH, 'r', encoding='utf-8-sig') as f:
            logs = json.load(f)
        return logs.get("errors", [])
    except Exception as e:
        print(f"[ERROR_STATS] Failed to load error history: {e}")
        return []

def aggregate_stats(errors: list) -> dict:
    """Aggregate statistics on attempt count, section-wise failure counts, and success rate."""
    stats = {
        "total_attempts": 0,
        "section_failures": defaultdict(int),
        "success_count": 0
    }
    
    for entry in errors:
        stats["total_attempts"] += 1
        if "successful" in entry and entry["successful"]:
            stats["success_count"] += 1
        else:
            section = entry.get("section", "Unknown")
            stats["section_failures"][section] += 1
    
    stats["failure_count"] = stats["total_attempts"] - stats["success_count"]
    if stats["total_attempts"] > 0:
        stats["success_rate"] = (stats["success_count"] / stats["total_attempts"]) * 100
    else:
        stats["success_rate"] = 0
    
    return stats

def run_generation_stats():
    """Controller function to load error history, aggregate statistics, and print the results."""
    errors = load_error_history()
    aggregated_stats = aggregate_stats(errors)
    
    print("Generation Statistics Report:")
    print(f"Total Attempts: {aggregated_stats['total_attempts']}")
    print(f"Successful Attempts: {aggregated_stats['success_count']}")
    print(f"Failed Attempts: {aggregated_stats['failure_count']}")
    print(f"Success Rate: {aggregated_stats['success_rate']:.2f}%")
    
    if aggregated_stats["section_failures"]:
        print("Section-wise Failures:")
        for section, failure_count in aggregated_stats["section_failures"].items():
            print(f"  - Section: {section}, Failures: {failure_count}")


# --- logging_and_integration ---
def log_load_error_history():
    """Log the process of loading error history."""
    print("[LOG] Loading error history from file.")

def log_aggregate_stats(errors: list):
    """Log the process of aggregating statistics."""
    print(f"[LOG] Aggregating {len(errors)} errors for statistics.")

def log_run_generation_stats():
    """Log the start of running the generation statistics report."""
    print("[LOG] Running generation statistics report.")

# Example usage
if __name__ == "__main__":
    log_load_error_history()
    errors = load_error_history()
    log_aggregate_stats(errors)
    aggregated_stats = aggregate_stats(errors)
    log_run_generation_stats()
    run_generation_stats()
