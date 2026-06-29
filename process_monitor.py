import os
import json
from datetime import datetime
import requests

# Constants
TASK_ERROR_LOG_PATH = "D:\\ai_agent\\server\\task_error_log.json"
MODEL_NAME = "qwen2.5-coder"

# Function to call Ollama model
def call_ollama(prompt: str) -> str:
    res = requests.post(
        "http://localhost:11434/api/chat",
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


# --- core_logic ---

def run_process_monitor():
    """Monitor the process and log any errors."""
    start_task_log("Process Monitoring", "core_logic")
    
    try:
        # Simulate monitoring logic (for demonstration purposes)
        monitor_pid = 18112
        if not is_process_running(monitor_pid):
            add_error(1, "process_monitor", f"PID {monitor_pid} is not running.")
            finish_task_log(False)
            return
        
        # Additional monitoring logic can be added here
        print(f"[PROCESS_MONITOR] PID {monitor_pid} is running.")

    except Exception as ex:
        add_error(1, "process_monitor", str(ex))
        finish_task_log(False)

def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception as ex:
        add_error(1, "process_check", str(ex))
        return False

# This section does not require additional functions for this specific task.


# --- logging_and_integration ---

def log_stage(stage: str):
    """Log the current processing stage."""
    print(f"[LOG_STAGE] {stage}")

def log_error(section: str, error: str):
    """Log an error encountered during processing."""
    print(f"[LOG_ERROR] Section: {section}, Error: {error}")

# Example usage
log_stage("Starting process monitoring")
try:
    # Simulated processing logic
    raise ValueError("Simulated error for demonstration")
except Exception as e:
    log_error("process_monitor", str(e))
