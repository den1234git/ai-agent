import json
import os
import shutil

# Constants
ERROR_PATTERNS_FILE = r"D:\ai_agent\server\error_patterns.json"
BACKUP_DIR = r"D:\ai_agent\server\_backups"


class ErrorPattern:
    def __init__(self, error_type: str, detail: str, count: int):
        self.error_type = error_type
        self.detail = detail
        self.count = count


def load_patterns() -> list:
    """Load error patterns from the JSON file."""
    if os.path.exists(ERROR_PATTERNS_FILE):
        with open(ERROR_PATTERNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_patterns(patterns: list):
    """Save error patterns to the JSON file."""
    with open(ERROR_PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=4)


def backup_patterns():
    """Create a backup of the current error patterns file."""
    if not os.path.exists(ERROR_PATTERNS_FILE):
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup_path = os.path.join(BACKUP_DIR, "error_patterns.json.bak")
    shutil.copy2(ERROR_PATTERNS_FILE, backup_path)
    print(f"[CLEANER] backup created: {backup_path}")


def clean_old_patterns(patterns: list) -> list:
    """Remove patterns with count <= 1."""
    return [pattern for pattern in patterns if pattern["count"] > 1]


def run_error_pattern_cleaner():
    """Controller function to clean old error patterns."""
    log_loading_patterns()
    patterns = load_patterns()
    log_backing_up_patterns()
    backup_patterns()
    log_cleaning_old_patterns()
    cleaned_patterns = clean_old_patterns(patterns)
    save_patterns(cleaned_patterns)
    log_saving_patterns()
    print(f"[CLEANER] {len(patterns) - len(cleaned_patterns)} patterns removed, {len(cleaned_patterns)} remaining.")


def log_loading_patterns():
    print("[LOG] Loading error patterns from the JSON file.")


def log_saving_patterns():
    print("[LOG] Saving error patterns to the JSON file.")


def log_backing_up_patterns():
    print("[LOG] Creating backup of the current error patterns file.")


def log_cleaning_old_patterns():
    print("[LOG] Cleaning old patterns with count <= 1.")


if __name__ == "__main__":
    run_error_pattern_cleaner()