import json
import os

# Constants
ERROR_PATTERNS_FILE = "D:\\ai_agent\\server\\error_patterns.json"
SUCCESS_PATTERNS_FILE = "D:\\ai_agent\\server\\success_patterns.json"
MAX_TOP_PATTERNS = 3
MAX_TOP_SUCCESS_PATTERNS = 3


class ErrorPattern:
    def __init__(self, error_type: str, detail: str, count: int):
        self.error_type = error_type
        self.detail = detail
        self.count = count


# ─── エラーパターン ───────────────────────────────────────

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


def add_pattern(error_type: str, detail: str):
    """Add a new error pattern to the list."""
    patterns = load_patterns()
    new_pattern = {"type": error_type, "detail": detail, "count": 1}
    patterns.append(new_pattern)
    save_patterns(patterns)


def increment_count(error_type: str):
    """Increment the count of an existing error pattern."""
    patterns = load_patterns()
    for pattern in patterns:
        if pattern["type"] == error_type:
            pattern["count"] += 1
            break
    save_patterns(patterns)


def get_top_patterns_prompt() -> str:
    """Generate a prompt for top error patterns."""
    patterns = load_patterns()
    sorted_patterns = sorted(patterns, key=lambda x: x["count"], reverse=True)
    top_patterns = sorted_patterns[:MAX_TOP_PATTERNS]

    prompt = "Top error patterns:\n"
    for i, pattern in enumerate(top_patterns, start=1):
        prompt += f"{i}. {pattern['type']}: {pattern['detail']} - Count: {pattern['count']}\n"

    return prompt


# ─── 成功パターン ─────────────────────────────────────────

def load_success_patterns() -> list:
    """Load success patterns from the JSON file."""
    if os.path.exists(SUCCESS_PATTERNS_FILE):
        with open(SUCCESS_PATTERNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_success_patterns(patterns: list):
    """Save success patterns to the JSON file."""
    with open(SUCCESS_PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=4, ensure_ascii=False)


def add_success_pattern(module_name: str, section: str, summary: str):
    """
    成功した生成のパターンを記録する。
    同じmodule_name+sectionの組み合わせがあればcountを増やす。
    なければ新規追加。

    module_name: 生成したモジュール名（例: error_pattern_cleaner）
    section: セクション名（imports_and_models / core_logic / logging_and_integration）
    summary: 成功した内容の要約（例: "Used defaultdict for aggregation"）
    """
    patterns = load_success_patterns()

    for pattern in patterns:
        if pattern["module"] == module_name and pattern["section"] == section:
            pattern["count"] += 1
            save_success_patterns(patterns)
            print(f"[SUCCESS_MEMORY] count++ {module_name}/{section}: {summary[:60]}")
            return

    new_pattern = {
        "module": module_name,
        "section": section,
        "summary": summary,
        "count": 1,
    }
    patterns.append(new_pattern)
    save_success_patterns(patterns)
    print(f"[SUCCESS_MEMORY] new pattern {module_name}/{section}: {summary[:60]}")


def record_success(module_name: str, full_code: str):
    """
    生成成功時に呼ぶ。full_codeを解析して各セクションの特徴を
    簡易的に抽出し、success_patternsに記録する。

    詳細なAI解析は行わず、コードの構造的特徴（使用ライブラリ、
    関数数など）を機械的に抽出する。
    """
    import re

    # importsセクション：使用ライブラリを抽出
    imports = re.findall(r"^import (\w+)|^from (\w+)", full_code, re.MULTILINE)
    libs = [a or b for a, b in imports if a or b]
    if libs:
        add_success_pattern(module_name, "imports_and_models", f"libs: {', '.join(set(libs))}")

    # core_logicセクション：関数数を抽出
    func_count = len(re.findall(r"^def \w+\(", full_code, re.MULTILINE))
    if func_count > 0:
        add_success_pattern(module_name, "core_logic", f"{func_count} functions defined")

    # logging_and_integrationセクション：printログの有無
    log_count = len(re.findall(r'print\(f"\[', full_code))
    if log_count > 0:
        add_success_pattern(module_name, "logging_and_integration", f"{log_count} log print statements")


def get_top_success_patterns_prompt() -> str:
    """
    成功パターンの上位をプロンプト文字列として返す。
    build_python_section_promptから呼んで生成プロンプトに注入する。
    """
    patterns = load_success_patterns()
    if not patterns:
        return ""

    sorted_patterns = sorted(patterns, key=lambda x: x["count"], reverse=True)
    top_patterns = sorted_patterns[:MAX_TOP_SUCCESS_PATTERNS]

    lines = ["Past successful patterns (follow these approaches):"]
    for i, p in enumerate(top_patterns, start=1):
        lines.append(f"{i}. [{p['section']}] {p['summary']} (used {p['count']} times)")

    return "\n".join(lines)


def run_error_memory():
    """Controller function to manage error patterns."""
    add_pattern("SyntaxError", "Missing colon")
    increment_count("SyntaxError")
    print(get_top_patterns_prompt())
    print(get_top_success_patterns_prompt())


# ─── ログ関数 ─────────────────────────────────────────────

def log_loading_patterns():
    print("[LOG] Loading error patterns from the JSON file.")


def log_saving_patterns():
    print("[LOG] Saving error patterns to the JSON file.")


def log_adding_pattern(error_type: str, detail: str):
    print(f"[LOG] Adding new pattern for {error_type}: {detail}")


def log_incrementing_count(error_type: str):
    print(f"[LOG] Incrementing count for {error_type}")


def log_generating_prompt():
    print("[LOG] Generating prompt for top error patterns.")


if __name__ == "__main__":
    add_pattern("SyntaxError", "Missing colon")
    increment_count("SyntaxError")
    print(get_top_patterns_prompt())
    print(get_top_success_patterns_prompt())