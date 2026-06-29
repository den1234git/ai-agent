import requests
from typing import List, Dict
import time
import json

# Constants
MODELS = ["qwen2.5-coder", "lfm2.5", "Llama-3.1-Swallow-8B"]
TEST_PROMPT = "Past error patterns"
OUTPUT_FILE = r"D:\ai_agent\server\stress_test_result.json"

# Data class for storing stress test results
class StressTestResult:
    def __init__(self, model_name: str, average_latency: float, max_latency: float, min_latency: float, failure_rate: float):
        self.model_name = model_name
        self.average_latency = average_latency
        self.max_latency = max_latency
        self.min_latency = min_latency
        self.failure_rate = failure_rate

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


# --- core_logic ---

def run_model_stress_test():
    results = []
    for model_name in MODELS:
        latencies = []
        failures = 0
        log_model_start(model_name)
        for _ in range(10):
            try:
                start_time = time.time()
                call_ollama(TEST_PROMPT, model_name)
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
            except Exception as e:
                failures += 1
                print(f"Error calling {model_name}: {e}")
        
        if len(latencies) > 0:
            average_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            failure_rate = failures / 10.0
        else:
            average_latency = 0.0
            max_latency = 0.0
            min_latency = 0.0
            failure_rate = 1.0
        
        result = StressTestResult(model_name, average_latency, max_latency, min_latency, failure_rate)
        results.append(result)
        
        log_model_complete(model_name, average_latency)
    
    save_results(results, OUTPUT_FILE)
    log_save_results(OUTPUT_FILE)

def save_results(results: List[StressTestResult], filename: str):
    with open(filename, 'w') as f:
        json.dump([result.__dict__ for result in results], f, indent=4)


# --- logging_and_integration ---
def log_start():
    print("[START] Starting stress test for models.")

def log_end():
    print("[END] Stress test completed and results saved.")

def log_model_start(model_name: str):
    print(f"[PROCESSING] Starting stress test for model: {model_name}")

def log_model_complete(model_name: str, average_latency: float):
    print(f"[COMPLETED] Model: {model_name}, Average Latency: {average_latency:.2f} seconds")

def log_save_results(filename: str):
    print(f"[SAVING] Saving results to file: {filename}")

# Usage example
if __name__ == "__main__":
    log_start()
    run_model_stress_test()
    log_end()
