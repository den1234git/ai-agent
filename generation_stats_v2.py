import json
import requests
from typing import List, Dict
import matplotlib.pyplot as plt

# Constants for model names and API endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder"

# Data class to hold generation statistics
class GenerationStat:
    def __init__(self, timestamp: str, success_rate: float):
        self.timestamp = timestamp
        self.success_rate = success_rate

# Type alias for list of GenerationStat objects
GenerationStatsList = List[GenerationStat]

# Function to call Ollama model
def call_ollama(prompt: str) -> str:
    res = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=600,
    )
    res.raise_for_status()
    data = res.json()
    return data["message"]["content"]


def fetch_generation_stats() -> GenerationStatsList:
    # Placeholder for fetching generation stats from a data source
    # This should be replaced with actual logic to retrieve data
    return [
        GenerationStat("2023-10-01T12:00:00", 0.85),
        GenerationStat("2023-10-02T12:00:00", 0.90),
        GenerationStat("2023-10-03T12:00:00", 0.78),
        # Add more data as needed
    ]

def plot_success_rate(stats: GenerationStatsList):
    timestamps = [stat.timestamp for stat in stats]
    success_rates = [stat.success_rate for stat in stats]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, success_rates, marker='o', linestyle='-')
    plt.title('Success Rate Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_generation_stats_v2():
    stats = fetch_generation_stats()
    plot_success_rate(stats)

# Example usage
if __name__ == "__main__":
    run_generation_stats_v2()


import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_fetching_data():
    logging.info("Fetching generation statistics data...")

def log_plotting_data():
    logging.info("Plotting success rate over time...")

def log_error(message: str):
    logging.error(f"Error: {message}")

# Example usage of the logging functions
if __name__ == "__main__":
    log_fetching_data()
    log_plotting_data()
    log_error("This is an example error message.")
