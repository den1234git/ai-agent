import requests
from typing import List, Dict
import time
import json

# Constants
MODELS = ["qwen2.5-coder", "lfm2.5", "Llama-3.1-Swallow-8B"]
TEST_PROMPT = "Past error patterns"

# Data class for storing latency results
class LatencyResult:
    def __init__(self, model_name: str, latency: float):
        self.model_name = model_name
        self.latency = latency

# Function to call Ollama API
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


def measure_latency(model_name: str, prompt: str) -> float:
    start_time = time.time()
    call_ollama(prompt, model_name)
    end_time = time.time()
    return end_time - start_time

def save_results(results: List[LatencyResult], filename: str):
    with open(filename, 'w') as f:
        json.dump([result.__dict__ for result in results], f, indent=4)

def run_model_latency_log():
    results = []
    for model_name in MODELS:
        latency = measure_latency(model_name, TEST_PROMPT)
        result = LatencyResult(model_name, latency)
        results.append(result)
    
    save_results(results, 'model_latencies.json')
    
    for result in results:
        print(f"Model: {result.model_name}, Latency: {result.latency:.2f} seconds")


# --- logging_and_integration ---
def log_start():
    print("[START] Starting latency measurement for models.")

def log_end():
    print("[END] Latency measurement completed and results saved.")

def log_model_start(model_name: str):
    print(f"[PROCESSING] Starting latency test for model: {model_name}")

def log_model_complete(model_name: str, latency: float):
    print(f"[COMPLETED] Model: {model_name}, Latency: {latency:.2f} seconds")

def log_save_results(filename: str):
    print(f"[SAVING] Saving results to file: {filename}")

# Usage example
if __name__ == "__main__":
    log_start()
    run_model_latency_log()
    log_end()
