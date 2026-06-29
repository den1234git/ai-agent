import json
import os
from datetime import datetime

TASK_ERROR_LOG_PATH = "D:\\ai_agent\\server\\task_error_log.json"

# Current task log (in-memory)
_current_log = None


def start_task_log(task: str, module_name: str):
    """Start a new task log. Overwrites previous log."""
    global _current_log
    _current_log = {
        "task": task,
        "module": module_name,
        "timestamp": datetime.now().isoformat(),
        "errors": [],
        "success": None,
    }
    _save_log()
    print(f"[TASK_LOG] Started: {task, module_name}")


def add_error(attempt: int, section: str, error: str):
    """Add an error entry to the current task log."""
    global _current_log
    if _current_log is None:
        return
    entry = {
        "attempt": attempt,
        "section": section,
        "error": error,
    }
    _current_log["errors"].append(entry)
    _save_log()
    print(f"[TASK_LOG] Error recorded (attempt {attempt}, {section}): {error[:80]}")


def finish_task_log(success: bool):
    """Mark the current task as finished."""
    global _current_log
    if _current_log is None:
        return
    _current_log["success"] = success
    _save_log()
    print(f"[TASK_LOG] Finished: success={success}, errors={len(_current_log['errors'])}")


def get_errors_for_prompt() -> str:
    """Return current task errors as a prompt string for LLM context."""
    if _current_log is None or not _current_log["errors"]:
        return ""

    lines = ["Errors encountered in this task so far:"]
    for e in _current_log["errors"]:
        lines.append(f"  - Attempt {e['attempt']} ({e['section']}): {e['error']}")
    return "\n".join(lines)


def flush_to_error_memory():
    """
    Write repeated error patterns to error_memory.
    Only errors that appear 2+ times are written (avoids learning one-off mistakes).
    """
    if _current_log is None or not _current_log["errors"]:
        return

    # Count occurrences of each error message
    counts = {}
    for e in _current_log["errors"]:
        key = e["error"][:120]
        counts[key] = counts.get(key, 0) + 1

    # Write repeated errors to error_memory
    try:
        import error_memory
        written = 0
        for error_text, count in counts.items():
            if count >= 2:
                # Derive error_type from section or content
                error_type = _infer_error_type(error_text)
                error_memory.add_pattern(error_type, error_text)
                written += 1
                print(f"[TASK_LOG] Flushed to error_memory: {error_type}: {error_text[:60]}")
        if written == 0:
            print("[TASK_LOG] No repeated errors to flush.")
    except Exception as ex:
        print(f"[TASK_LOG] flush_to_error_memory failed: {ex}")


def _infer_error_type(error_text: str) -> str:
    """Infer error type label from error message content."""
    text = error_text.lower()
    if "python" in text and "orphan" in text:
        return "python_orphan"
    if "json" in text or "structure" in text:
        return "json_structure"
    if "torch" in text or "transformers" in text or "automodel" in text:
        return "no_torch"
    if "syntax" in text:
        return "SyntaxError"
    if "import" in text:
        return "ImportError"
    if "controller" in text or "missing" in text:
        return "missing_controller"
    return "GeneralError"


def _save_log():
    """Save current log to JSON file."""
    if _current_log is None:
        return
    try:
        with open(TASK_ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(_current_log, f, ensure_ascii=False, indent=4)
    except Exception as ex:
        print(f"[TASK_LOG] Save failed: {ex}")


def load_log() -> dict:
    """Load the last task log from file."""
    if not os.path.exists(TASK_ERROR_LOG_PATH):
        return {}
    try:
        with open(TASK_ERROR_LOG_PATH, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return {}