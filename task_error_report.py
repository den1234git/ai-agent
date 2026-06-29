import requests
from typing import List, Dict

# Constants
ERROR_LOG_FILE = "task_errors.json"

# Data class for storing error statistics
class ErrorStat:
    def __init__(self, section: str, error_type: str, count: int):
        self.section = section
        self.error_type = error_type
        self.count = count


import json

def read_error_log(filename: str) -> List[Dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def aggregate_errors(errors: List[Dict]) -> Dict[str, Dict[str, int]]:
    error_stats = {}
    for error in errors:
        section = error.get('section', 'Unknown')
        error_type = error.get('error_type', 'Unknown')
        
        if section not in error_stats:
            error_stats[section] = {}
        
        if error_type not in error_stats[section]:
            error_stats[section][error_type] = 0
        
        error_stats[section][error_type] += 1
    
    return error_stats

def print_error_report(error_stats: Dict[str, Dict[str, int]]):
    for section, errors in error_stats.items():
        print(f"Section: {section}")
        for error_type, count in errors.items():
            print(f"  Error Type: {error_type}, Count: {count}")
        print()

def run_task_error_report():
    errors = read_error_log(ERROR_LOG_FILE)
    error_stats = aggregate_errors(errors)
    print_error_report(error_stats)

# Example usage
if __name__ == "__main__":
    run_task_error_report()


# Logging and integration section for task_error_report.py

def log_start():
    print("[START] Starting error report aggregation.")

def log_end():
    print("[END] Error report aggregation completed.")

def log_read_error_log(filename: str):
    print(f"[PROCESSING] Reading error log from file: {filename}")

def log_aggregate_errors():
    print("[PROCESSING] Aggregating errors by section and type.")

def log_print_report():
    print("[PROCESSING] Printing aggregated error report.")

# Usage example
if __name__ == "__main__":
    log_start()
    run_task_error_report()
    log_end()
