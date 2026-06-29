"""
Python向け分割生成パイプライン
"""

import os
import re
import shutil
import time

from ai import ask_ai
import progress
import error_memory
import task_error_log

error_memory.load_patterns()  # patterns loaded on demand

BASE_SERVER_DIR = r"D:\ai_agent\server"
BACKUP_DIR = os.path.join(BASE_SERVER_DIR, "_backups")

PYTHON_SECTIONS = ["imports_and_models", "core_logic", "logging_and_integration"]

MAX_RETRIES_PER_SECTION = 2
MAX_SEMANTIC_RETRIES = 2

def _build_reference_section(reference_files: list) -> str:
    if not reference_files:
        return "(none)"
    parts = []
    for f in reference_files:
        parts.append(f"// --- {f['name']} ---\n{f['content']}")
    return "\n\n".join(parts)

def build_python_split_task_prompt(task: str, module_name: str = None) -> str:
    if module_name:
        return f'{{"module_name": "{module_name}"}}'
    return f"""You are an AI that designs Python backend code.
Decide one module name to implement the following task.

# Task
{task}

# Output Rules (strictly follow)
- Output ONLY the following JSON format. No explanations.
- module_name must be snake_case with lowercase letters and underscores only (e.g. self_improve, data_validator)
- Do NOT use names that conflict with existing Python files (router, progress, ai, executor, memory, etc.)
- Do NOT write Japanese characters inside string literals in generated code.
  Use English or Unicode escapes instead.
  Bad:  prompt = "過去のエラー傾向"
  Good: prompt = "Past error patterns"

Output format:
{{"module_name": "..."}}
"""


def build_python_section_prompt(step: str, task: str, context: str,
                                  controller_function: str,
                                  reference_files: list = []) -> str:
    common_header = f"""You are an AI that generates Python backend code.
You are implementing the following task split into 3 sections.
Output ONLY the code for the current section.

# Task
{task}

# Code generated so far (for reference only - do NOT repeat or rewrite it)
{context if context else "(none - this is the first section)"}

[IMPORTANT] The above is reference only. Never:
- Rewrite, duplicate, or create similarly-named functions/classes from above
- Summarize or repeat the above logic
Write ONLY the next new part.

# Output Rules (strictly follow)
- No explanations, preambles, or code block markers (no ``` or ```python lines)
- Output only the code for the specified section
- Comments are allowed but keep them brief
- Do NOT write raw newlines inside string literals. Use \\n or split into multiple print/logging calls.
- Do NOT write Japanese characters directly inside string literals.
  Use English or Unicode escapes instead.
  Bad:  prompt = "過去のエラー傾向"
  Good: prompt = "Past error patterns"

# Absolute prohibitions
- Do NOT output JSON structures like {{"imports": [...], "constants": {{...}}}}
  Your output will be written directly to a Python file. Output pure Python source code.
- Do NOT wrap the entire output in \"\"\" or ''' string literals.
- The first line must start with import, #, def, or class. Never start with {{ or ".

# How to call models (IMPORTANT - always follow this pattern)
If the task mentions model names like qwen2.5-coder or lfm2.5, call them via local Ollama.
Do NOT use torch, transformers, AutoModelForCausalLM, or openai. Never import these.

Use exactly this pattern:

    import requests

    def call_ollama(prompt: str, model_name: str) -> str:
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={{
                "model": model_name,
                "messages": [{{"role": "user", "content": prompt}}],
                "stream": False,
            }},
            timeout=600,
        )
        res.raise_for_status()
        data = res.json()
        return data["message"]["content"]
        

# Reference files (follow these patterns)
{_build_reference_section(reference_files)}
# Past error patterns (must avoid)
{error_memory.get_top_patterns_prompt() or "(none)"}
# Past success patterns (follow these approaches)
{error_memory.get_top_success_patterns_prompt() or "(none)"}
"""

    if step == "imports_and_models":
        return common_header + f"""
# Current section: imports_and_models
Write ONLY:
- import statements
- constant definitions (if any)
- data classes / type definitions used in this module

Do NOT write function logic here. That goes in core_logic.
"""

    elif step == "core_logic":
        return common_header + f"""
# Current section: core_logic
Write ALL the main logic of this module.

[CRITICAL] Rules:
1. This section MUST include ALL functions described in the task.
   Do NOT skip any function. Do NOT write placeholder/stub functions.
   Every function must have a complete, working implementation.

2. This section MUST also include this controller function:

    def {controller_function}(...):

3. Do NOT write import statements here (already in imports_and_models).

Example: if the task requires load_patterns, save_patterns, add_pattern,
increment_count, get_top_patterns_prompt — write ALL 5 functions here,
plus the controller function. Missing even one function is a failure.
"""

    elif step == "logging_and_integration":
        return common_header + f"""
# Current section: logging_and_integration
Write:
- Log output for each processing stage (print statements are fine, e.g. print(f"[..] ..."))
  Write these as helper log functions, not inside core_logic directly.
- A brief usage example if helpful

Do NOT add new primary logic here. That belongs in core_logic.
"""

    else:
        raise ValueError(f"unknown step: {step}")


def _escape_japanese_in_strings(code: str) -> str:
    # DEBUG
    jp_chars_in = [c for c in code if ord(c) >= 0x3000]
    print(f"[ESCAPE_JP] input has {len(jp_chars_in)} japanese chars: {jp_chars_in[:5]}")

    result = []
    i = 0
    in_string = False
    string_char = ""
    is_triple = False

    while i < len(code):
        ch = code[i]

        if not in_string:
            if ch == "#":
                end = code.find("\n", i)
                if end == -1:
                    result.append(code[i:])
                    break
                result.append(code[i:end])
                i = end
                continue

            if code[i:i+3] in ('"""', "'''"):
                in_string = True
                is_triple = True
                string_char = code[i:i+3]
                result.append(string_char)
                i += 3
                continue

            if ch in ('"', "'"):
                in_string = True
                is_triple = False
                string_char = ch
                result.append(ch)
                i += 1
                continue

            result.append(ch)
            i += 1

        else:
            if is_triple:
                if code[i:i+3] == string_char:
                    result.append(string_char)
                    i += 3
                    in_string = False
                    continue
            else:
                if ch == "\\" and i + 1 < len(code):
                    result.append(ch)
                    result.append(code[i+1])
                    i += 2
                    continue
                if ch == string_char:
                    result.append(ch)
                    i += 1
                    in_string = False
                    continue

            if ord(ch) >= 0x3000:
                result.append(f"\\u{ord(ch):04x}")
            else:
                result.append(ch)
            i += 1

    output = "".join(result)

    # DEBUG
    jp_chars_out = [c for c in output if ord(c) >= 0x3000]
    print(f"[ESCAPE_JP] output has {len(jp_chars_out)} japanese chars remaining")

    return output


def _validate_python_section(step: str, code: str, controller_function: str) -> list:
    problems = []

    if not code or not code.strip():
        problems.append(f"{step}: output is empty.")
        return problems

    # Check for orphan 'python' lines (Markdown code fence remnants)
    orphan_lines = [i+1 for i, line in enumerate(code.split("\n")) if line.strip() == "python"]
    if orphan_lines:
        error_memory.add_pattern("python_orphan", "bare python line in output - do not use Markdown code block syntax")
        problems.append(
            f"Output contains bare 'python' line(s) at line(s) {orphan_lines}. "
            "Do NOT use Markdown code block syntax (no ```python or standalone 'python' lines)."
        )
        return problems

    stripped = code.strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        problems.append(
            "Output is still in JSON object format. "
            "Output raw Python source code only, not wrapped in JSON."
        )
        error_memory.increment_count("json_structure")
        return problems

    if stripped.startswith('"""') or stripped.startswith("'''"):
        problems.append(
            "Output is wrapped in triple-quoted string literal. "
            "Output executable Python code directly, not as a string."
        )
        return problems

    try:
        import ast
        ast.parse(stripped)
    except SyntaxError as e:
        problems.append(
            f"Invalid Python syntax ({e.msg}, line {e.lineno}). "
            f"Output must be valid Python code only."
        )
        return problems

    if step == "core_logic":
        pattern = f"def {controller_function}("
        if pattern not in code:
            problems.append(
                f"Controller function '{controller_function}' is missing. "
                f"Must include 'def {controller_function}(...)'."
            )

    return problems


def _auto_fix_raw_newlines_in_strings(code: str) -> str:
    lines = code.split("\n")
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        triple_count = line.count('"""') + line.count("'''")
        if triple_count % 2 == 1:
            result_lines.append(line)
            i += 1
            continue

        # Skip lines containing triple quotes (e.g. docstring open/close lines)
        # to avoid misdetecting apostrophes or quotes inside docstrings
        if triple_count > 0:
            result_lines.append(line)
            i += 1
            continue

        unescaped_double = len([c for idx, c in enumerate(line) if c == '"' and (idx == 0 or line[idx-1] != "\\")])
        unescaped_single = len([c for idx, c in enumerate(line) if c == "'" and (idx == 0 or line[idx-1] != "\\")])

        if unescaped_double % 2 == 1 or unescaped_single % 2 == 1:
            merged = line
            j = i + 1
            while j < len(lines):
                merged += "\\n" + lines[j]
                d = len([c for idx, c in enumerate(merged) if c == '"' and (idx == 0 or merged[idx-1] != "\\")])
                s = len([c for idx, c in enumerate(merged) if c == "'" and (idx == 0 or merged[idx-1] != "\\")])
                if d % 2 == 0 and s % 2 == 0:
                    break
                j += 1
            result_lines.append(merged)
            i = j + 1
            continue

        result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def _generate_python_section_with_retry(step: str, task: str, context: str,
                                          controller_function: str = "",
                                          reference_files: list = []) -> str:
    feedback = ""
    code = ""

    for attempt in range(MAX_RETRIES_PER_SECTION + 1):
        prompt = build_python_section_prompt(step, task, context, controller_function, reference_files)
        if feedback:
            prompt += f"\n\nThe previous output had the following issues. Fix them:\n{feedback}"

        print(f"[SPLIT_PY] generating: {step} (attempt {attempt + 1}/{MAX_RETRIES_PER_SECTION + 1})")
        section_result = ask_ai(prompt, model="codellama")
        print(f"[SPLIT_PY DEBUG] section_result={section_result}")

        code = section_result.get("code", "")
        if not code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    code = a.get("content")
                    break

        code = _auto_fix_raw_newlines_in_strings(code)
        code = _escape_japanese_in_strings(code)

        problems = _validate_python_section(step, code, controller_function)
        if not problems:
            return code

        print(f"[SPLIT_PY] {step} has problems: {problems}")
        for problem in problems:
            task_error_log.add_error(attempt + 1, step, problem)
        feedback = "\n".join(f"- {p}" for p in problems)

    print(f"[WARN] {step} could not be fixed after {MAX_RETRIES_PER_SECTION + 1} attempts. Continuing with current code.")
    return code


def _to_snake_case(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def _controller_function_name(module_name: str) -> str:
    return f"run_{module_name}"


def _backup_if_exists(file_path: str):
    if not os.path.exists(file_path):
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    backup_path = os.path.join(BACKUP_DIR, f"{filename}.bak")

    shutil.copy2(file_path, backup_path)
    print(f"[SPLIT_PY] backup created: {backup_path}")


def _check_existing_file(task: str, file_path: str) -> bool:
    """
    既存ファイルがタスクの要件を満たしているかlfm2.5でチェックする。
    満たしていればTrueを返し、生成をスキップする。
    """
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            existing_code = f.read()
    except Exception:
        return False

    prompt = f"""You are a code reviewer. Check if the existing file already satisfies the task requirements.

Task:
{task}

Existing file content:
{existing_code}

Evaluation criteria:
- Does the file implement the functionality described in the task?
- Ignore implementation language differences (Python vs TypeScript vs HTML are all valid).
- Focus only on whether the required features/functions/UI elements are present.
- Minor differences in variable names or style are NOT a reason to fail.

Answer ONLY with this JSON:
{{
  "satisfies": true or false,
  "reason": "one-line reason"
}}
"""
    try:
        result = ask_ai(prompt, model="router")
        satisfies = bool(result.get("satisfies", False))
        reason = result.get("reason", "")
        print(f"[SPLIT_PY][EXISTING CHECK] satisfies={satisfies} reason={reason}")
        return satisfies
    except Exception as e:
        print(f"[SPLIT_PY][EXISTING CHECK] check failed: {e}")
        return False


def run_split_generation_python(task: str, generation: int,
                                  percent_start: int = 0, percent_end: int = 100,
                                  progress_label: str = "", module_name_override: str = None,
                                  reference_files: list = []):
    start_time = time.monotonic()

    progress.update(generation, percent_start, f"{progress_label}モジュール名を決定しています")
    plan_result = ask_ai(build_python_split_task_prompt(task, module_name=module_name_override), model="codellama")
    module_name_raw = plan_result.get("module_name", "generated_module")

    module_name = module_name_override if module_name_override else _to_snake_case(module_name_raw)
    controller_function = _controller_function_name(module_name)

    task_error_log.start_task_log(task, module_name)  # ★ 追加

    file_path = os.path.join(BASE_SERVER_DIR, f"{module_name}.py")

    print(f"[SPLIT_PY] module_name={module_name}")
    print(f"[SPLIT_PY] controller_function={controller_function}")
    print(f"[SPLIT_PY] file_path={file_path}")

    # 既存ファイルがタスクを満たしているかチェック → 満たしていれば生成スキップ
    progress.update(generation, percent_start + 2, f"{progress_label}既存ファイルを確認しています")
    if _check_existing_file(task, file_path):
        print(f"[SPLIT_PY] Existing file satisfies requirements. Skipping generation.")
        elapsed_seconds = round(time.monotonic() - start_time, 1)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            existing_code = f.read()
        return {
            "module_name": module_name,
            "controller_function": controller_function,
            "file_path": file_path,
            "full_code": existing_code,
            "elapsed_seconds": elapsed_seconds,
        }

    context = ""
    sections = {}

    total_steps = len(PYTHON_SECTIONS)
    section_zone_start = percent_start + max(int((percent_end - percent_start) * 0.1), 1)
    section_zone_span = max(percent_end - section_zone_start - 10, 10)

    for i, step in enumerate(PYTHON_SECTIONS):
        current_percent = section_zone_start + int(section_zone_span * (i / total_steps))
        progress.update(generation, current_percent, f"{progress_label}生成中: {step}")

        code = _generate_python_section_with_retry(step, task, context, controller_function, reference_files)

        sections[step] = code
        context += f"\n\n# --- {step} ---\n{code}"
        print(f"[SPLIT_PY] {step} done ({len(code)} chars)")

    progress.update(generation, percent_end - 8, f"{progress_label}ファイルを統合しています")

    full_code = _assemble_python(sections)

    progress.update(generation, percent_end - 5, f"{progress_label}生成内容を検証しています")

    for semantic_attempt in range(MAX_SEMANTIC_RETRIES + 1):
        check = _check_semantic_match_python(task, full_code)
        print(f"[SPLIT_PY][SEMANTIC CHECK] matches={check['matches']} "
              f"reason={check['reason']} problem_section={check['problem_section']}")

        if check["matches"]:
            break

        reason_lower = check["reason"].lower()
        if any(kw in reason_lower for kw in ["torch", "transformers", "automodel", "openai"]):
            error_memory.increment_count("no_torch")
        elif any(kw in reason_lower for kw in ["placeholder", "dummy"]):
            error_memory.increment_count("json_structure")

        problem_section = check["problem_section"]
        if not problem_section or problem_section not in sections:
            print("[SPLIT_PY][SEMANTIC CHECK] Could not identify problem section. Skipping regeneration.")
            break

        if semantic_attempt >= MAX_SEMANTIC_RETRIES:
            print("[SPLIT_PY][SEMANTIC CHECK] Max retries reached. Continuing with current code.")
            break

        task_error_log.add_error(semantic_attempt + 1, problem_section or "unknown", check["reason"])
        print(f"[SPLIT_PY][SEMANTIC CHECK] Regenerating {problem_section}. Reason: {check['reason']}")
        progress.update(generation, percent_end - 5, f"{progress_label}{problem_section}を見直しています")

        retry_prompt = build_python_section_prompt(problem_section, task, context, controller_function)
        retry_prompt += f"\n\nAn objective review identified the following issue. Fix it:\n- {check['reason']}"
        section_result = ask_ai(retry_prompt, model="codellama")
        new_code = section_result.get("code", "")
        if not new_code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    new_code = a.get("content")
                    break

        new_code = _auto_fix_raw_newlines_in_strings(new_code)
        new_code = _escape_japanese_in_strings(new_code)

        sections[problem_section] = new_code
        full_code = _assemble_python(sections)

    _backup_if_exists(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    print(f"[SPLIT_PY] written: {file_path}")

    # 実行チェック：保存したファイルをimportして動くか確認
    runtime_result = _runtime_check_python(module_name, file_path)
    if not runtime_result["ok"]:
        print(f"[SPLIT_PY] Runtime error detected: {runtime_result['error_type']}: {runtime_result['detail']}")

    task_error_log.finish_task_log(runtime_result["ok"])
    if runtime_result["ok"]:
        error_memory.record_success(module_name, full_code)  # ★ 成功パターン記録
    task_error_log.flush_to_error_memory()

    elapsed_seconds = round(time.monotonic() - start_time, 1)
    print(f"[SPLIT_PY] elapsed: {elapsed_seconds}s")

    return {
        "module_name": module_name,
        "controller_function": controller_function,
        "file_path": file_path,
        "full_code": full_code,
        "elapsed_seconds": elapsed_seconds,
    }


def _remove_orphan_python_lines(code: str) -> str:
    """Remove bare 'python' lines left over from Markdown code fence syntax."""
    lines = [line for line in code.split("\n") if line.strip() != "python"]
    return "\n".join(lines)


def _assemble_python(secs: dict) -> str:
    raw = f"""{secs.get("imports_and_models", "")}


{secs.get("core_logic", "")}


{secs.get("logging_and_integration", "")}
"""
    return _remove_orphan_python_lines(raw)


def _runtime_check_python(module_name: str, file_path: str) -> dict:
    """
    生成されたファイルを一時的にimportして実行エラーを検知する。
    エラーがあれば error_memory と task_error_log に記録して返す。
    戻り値: {"ok": True} or {"ok": False, "error_type": str, "detail": str}
    """
    if not os.path.exists(file_path):
        return {"ok": False, "error_type": "FileNotFoundError", "detail": f"{file_path} not found"}

    import importlib.util
    import sys

    # 既にimport済みの場合はキャッシュを削除して再ロード
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"[RUNTIME CHECK] {module_name}: OK")
        return {"ok": True}
    except Exception as e:
        error_type = type(e).__name__
        detail = str(e)[:120]
        print(f"[RUNTIME CHECK] {module_name}: {error_type}: {detail}")
        error_memory.add_pattern(error_type, detail)
        task_error_log.add_error(0, "runtime", f"{error_type}: {detail}")
        return {"ok": False, "error_type": error_type, "detail": detail}


def _check_semantic_match_python(task: str, full_code: str) -> dict:
    prompt = f"""You are a code reviewer. Compare the task description and the generated Python code.
Determine whether the required functionality is actually implemented.

Task:
{task}

Generated code:
{full_code}

Criteria:
- Are the models/services named in the task (e.g. qwen2.5-coder, lfm2.5, Ollama) actually called?
  Just having the name in a variable or comment is NOT enough - there must be actual call code.
- IMPORTANT: Model calls MUST use requests.post to http://localhost:11434 (Ollama API).
  Using torch, transformers, AutoModelForCausalLM, AutoTokenizer, or openai.Completion is WRONG.
  This project uses Ollama only.
- Are function bodies empty/placeholder (return 'dummy', pass) when the task requires real logic?
- Does the control flow (loops, conditionals) match the task's required processing flow?
- Minor variable name differences or comment wording are NOT mismatches.

Return ONLY this JSON:
{{
  "matches": true or false,
  "reason": "one-line reason",
  "problem_section": "imports_and_models" or "core_logic" or "logging_and_integration" or null
}}
"""
    try:
        result = ask_ai(prompt, model="router")
        return {
            "matches": bool(result.get("matches", True)),
            "reason": result.get("reason", ""),
            "problem_section": result.get("problem_section"),
        }
    except Exception as e:
        print(f"[WARN] Semantic check failed (skipping): {e}")
        return {"matches": True, "reason": "check skipped", "problem_section": None}