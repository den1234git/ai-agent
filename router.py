import re

from build_prompt import (
    build_prompt,
    build_split_task_prompt,
    build_section_prompt,
    build_html_mockup_prompt,
    build_multi_file_plan_prompt,
    build_connection_plan_prompt,
    build_connect_components_prompt,
    build_vision_prompt,
)
from run_split_generation_python import run_split_generation_python

from ai import ask_ai
import progress

print("ROUTER FILE LOADED:", __file__)

MAX_RETRIES_PER_SECTION = 2
MAX_SEMANTIC_RETRIES = 1
MAX_LOOP_ITERATIONS = 10  # 絶対上限（UIの999も内部でこれにクランプ）

PYTHON_MODULE_KEYWORDS = [
    "自己改善ループ",
    "qwen2.5-coder",
    "qwen2.5",
    "lfm2.5",
    "イテレーション",
    "ルーティングロジック",
    "router.py",
    "チェッカー",
]


def _looks_like_python_module_request(user: str) -> bool:
    return any(kw in user for kw in PYTHON_MODULE_KEYWORDS)


def _component_decl_pattern(component_name: str) -> re.Pattern:
    name_pattern = re.escape(component_name)
    return re.compile(
        rf"^(export\s+default\s+)?(function\s+{name_pattern}\s*\(|const\s+{name_pattern}\b.*=>\s*\{{)\s*$"
    )


def _sanitize_state_section(code: str, component_name: str = "") -> str:
    if not component_name:
        return code.strip()

    component_decl_re = _component_decl_pattern(component_name)
    lines = code.split("\n")
    cleaned = []
    removed_open = False
    for line in lines:
        stripped = line.strip()
        if component_decl_re.match(stripped):
            removed_open = True
            continue
        cleaned.append(line)

    if removed_open:
        while cleaned and cleaned[-1].strip() == "}":
            cleaned.pop()
            break

    return "\n".join(cleaned).strip()


def _sanitize_jsx_section(code: str) -> str:
    has_jsx_tag = re.search(r"<[a-zA-Z]", code)
    if not has_jsx_tag:
        print(f"[WARN] jsx_return セクションにJSXタグが見つからない。フォールバックを使用します。元の内容: {code!r}")
        return (
            "<div>\n"
            "  {/* 生成に失敗したため仮の表示です。元の指示を変えて再生成してください */}\n"
            "  <p>プレビュー生成エラー：JSXが正しく生成されませんでした</p>\n"
            "</div>"
        )
    return code


def fix_react_imports(code: str) -> str:
    hooks = [
        "useState", "useEffect", "useCallback", "useMemo",
        "useRef", "useContext", "useReducer", "useLayoutEffect",
    ]
    used_hooks = sorted(set(h for h in hooks if re.search(rf'\b{h}\s*\(', code)))

    if not used_hooks:
        return code

    m = re.search(r"import\s+React\s*,\s*\{([^}]*)\}\s*from\s*['\"]react['\"];?", code)
    if m:
        existing = [x.strip() for x in m.group(1).split(",") if x.strip()]
        missing = [h for h in used_hooks if h not in existing]
        if missing:
            new_line = f"import React, {{ {', '.join(existing + missing)} }} from 'react';"
            code = code[:m.start()] + new_line + code[m.end():]
        return code

    m = re.search(r"import\s*\{([^}]*)\}\s*from\s*['\"]react['\"];?", code)
    if m:
        existing = [x.strip() for x in m.group(1).split(",") if x.strip()]
        missing = [h for h in used_hooks if h not in existing]
        if missing:
            new_line = f"import {{ {', '.join(existing + missing)} }} from 'react';"
            code = code[:m.start()] + new_line + code[m.end():]
        return code

    m = re.search(r"import\s+React\s+from\s*['\"]react['\"];?", code)
    if m:
        new_line = f"import React, {{ {', '.join(used_hooks)} }} from 'react';"
        code = code[:m.start()] + new_line + code[m.end():]
        return code

    new_line = f"import React, {{ {', '.join(used_hooks)} }} from 'react';\n"
    return new_line + code


def _find_props_interface_name(imports_code: str, component_name: str):
    preferred_name = f"{component_name}Props"
    if re.search(rf"\b(interface|type)\s+{re.escape(preferred_name)}\b", imports_code):
        return preferred_name
    m = re.search(r"\b(?:interface|type)\s+(\w*Props)\b", imports_code)
    if m:
        return m.group(1)
    return None


def fix_component_props(code: str, imports_code: str, component_name: str) -> str:
    props_name = _find_props_interface_name(imports_code, component_name)
    if not props_name:
        return code

    already_has_param = re.search(
        rf"export\s+default\s+function\s+{re.escape(component_name)}\s*\([^)]*\w[^)]*\)",
        code,
    )
    if already_has_param:
        return code

    decl_pattern = re.compile(
        rf"(export\s+default\s+function\s+{re.escape(component_name)}\s*)\(\s*\)"
    )
    new_code, n = decl_pattern.subn(rf"\1(props: {props_name})", code, count=1)
    if n == 0:
        return code

    iface_match = re.search(
        rf"(?:interface|type)\s+{re.escape(props_name)}\s*(?:=\s*)?\{{([^}}]*)\}}",
        imports_code,
        re.DOTALL,
    )
    if iface_match:
        body = iface_match.group(1)
        prop_names = re.findall(r"^\s*(\w+)\s*\??\s*:", body, re.MULTILINE)
        if prop_names:
            destructure_line = f"  const {{ {', '.join(prop_names)} }} = props;\n"
            insert_pos = new_code.find("{", new_code.find(f"function {component_name}"))
            if insert_pos != -1:
                insert_pos += 1
                new_code = (
                    new_code[:insert_pos]
                    + "\n"
                    + destructure_line
                    + new_code[insert_pos:]
                )

    return new_code


def _validate_section(section_name: str, code: str, component_name: str = "") -> list:
    problems = []

    if not code or not code.strip():
        problems.append("コードが空です。")
        return problems

    if re.search(r'"\s*code\s*"\s*:\s*"', code):
        problems.append('出力にJSON記法の断片（"code": "...）が紛れ込んでいます。コードのみを出力してください。')

    if section_name == "imports_and_types":
        if re.search(r"^(export\s+default\s+)?function\s+\w+\s*\(", code, re.MULTILINE):
            problems.append("imports_and_types に関数定義が含まれています。importと型定義のみにしてください。")

    elif section_name == "state_and_handlers":
        if component_name:
            component_decl_re = re.compile(
                _component_decl_pattern(component_name).pattern, re.MULTILINE
            )
            if component_decl_re.search(code):
                problems.append(
                    f"state_and_handlers にコンポーネント自体の宣言行（{component_name} の関数宣言）が"
                    "含まれています。コンポーネントの宣言は書かず、中身の処理のみを書いてください。"
                    "なお、handleClickのようなハンドラー関数を const handleClick = () => {...}; の形で"
                    "宣言するのは問題ありません。それは消さずに残してください。"
                )
        if re.search(r"^\s*return\s*\(", code, re.MULTILINE):
            problems.append("state_and_handlers にJSXのreturn文が含まれています。JSXはここに書かないでください。")

    elif section_name == "jsx_return":
        if not re.search(r"<[a-zA-Z]", code):
            problems.append("jsx_return に実際のJSXタグ（<div>など）が含まれていません。表示するJSXを書いてください。")
        if re.search(r"^\s*//.*不要", code, re.MULTILINE):
            problems.append("「不要」などのコメントのみで実体のコードがありません。実際に動くJSXを書いてください。")

    return problems


def _generate_section_with_retry(section_name, task, context, component_name="", reference_files=None, knowledge_context=""):
    feedback = ""
    code = ""

    for attempt in range(MAX_RETRIES_PER_SECTION + 1):
        prompt = build_section_prompt(section_name, task, context, reference_files=reference_files, knowledge_context=knowledge_context)
        if feedback:
            prompt += f"\n\n前回の出力には以下の問題がありました。必ず修正してください：\n{feedback}"

        print(f"[SPLIT] generating: {section_name} (attempt {attempt + 1}/{MAX_RETRIES_PER_SECTION + 1})")
        section_result = ask_ai(prompt, model="codellama")
        print(f"[SPLIT DEBUG] section_result={section_result}")

        code = section_result.get("code", "")
        if not code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    code = a.get("content")
                    break

        problems = _validate_section(section_name, code, component_name)
        if not problems:
            return code

        print(f"[SPLIT] {section_name} に問題あり: {problems}")
        feedback = "\n".join(f"- {p}" for p in problems)

    print(f"[WARN] {section_name} は{MAX_RETRIES_PER_SECTION + 1}回試行しても問題が解決しませんでした。サニタイズ処理にフォールバックします。")
    return code


def _check_semantic_match(task: str, full_code: str) -> dict:
    prompt = f"""
あなたはコードレビューの担当者です。以下のタスク指示と、実際に生成されたReactコンポーネントのコードを比較し、
指示された機能がコードに実装されているかを客観的に判定してください。

タスク指示：
{task}

生成されたコード：
{full_code}

判定基準：
- タスクで要求された機能（ボタン、表示内容、状態管理など）が実際にコードに存在するか
- 明らかに指示と異なる、または機能が欠けている場合は不一致と判定する
- 些細な見た目の違い（色、文言など）は不一致としない

以下のJSONのみ返してください：
{{
  "matches": true または false,
  "reason": "判定理由を一言で",
  "problem_section": "imports_and_types" または "state_and_handlers" または "jsx_return" または null
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
        print(f"[WARN] 意味チェックに失敗しました（無視して進めます）: {e}")
        return {"matches": True, "reason": "検証スキップ", "problem_section": None}


def _generate_one_component(
    task,
    generation,
    component_name_hint=None,
    progress_label="",
    percent_start=10,
    percent_end=90,
    reference_files=None,
):
    if component_name_hint:
        component_name = component_name_hint
        steps = ["imports_and_types", "state_and_handlers", "jsx_return"]
    else:
        progress.update(generation, percent_start, f"{progress_label}生成計画を立てています")
        plan_result = ask_ai(build_split_task_prompt(task), model="codellama")
        component_name = plan_result.get("component_name", "Generated")
        steps = plan_result.get("steps", ["imports_and_types", "state_and_handlers", "jsx_return"])

    file_path = f"src/pages/{component_name}.tsx"

    print(f"[SPLIT] file_path={file_path}")
    print(f"[SPLIT] steps={steps}")

    context = ""
    sections = {}
    knowledge_context = ""  # ★ 消防知識を別変数で保持

    # ★ ChromaDBから類似経験・消防知識を検索
    try:
        from memory_store import (
            search_experience, search_shobo_knowledge,
            format_experience_context, format_knowledge_context,
        )
        exp_context = format_experience_context(search_experience(task))
        knowledge_context = format_knowledge_context(search_shobo_knowledge(task))
        if exp_context:
            context += f"\n\n{exp_context}"
        if exp_context or knowledge_context:
            print(f"[MEMORY] contextに追加: 経験={bool(exp_context)} 知識={bool(knowledge_context)}")
    except Exception as e:
        print(f"[MEMORY] 検索をスキップしました（無視して続行）: {e}")
    total_steps = max(len(steps), 1)
    section_zone_start = percent_start + max(int((percent_end - percent_start) * 0.1), 1)
    section_zone_span = max(percent_end - section_zone_start - 10, 10)

    for i, step in enumerate(steps):
        current_percent = section_zone_start + int(section_zone_span * (i / total_steps))
        progress.update(generation, current_percent, f"{progress_label}生成中: {step}")

        code = _generate_section_with_retry(step, task, context, component_name, reference_files=reference_files, knowledge_context=knowledge_context)

        if step == "state_and_handlers":
            code = _sanitize_state_section(code, component_name)
        elif step == "jsx_return":
            code = _sanitize_jsx_section(code)

        sections[step] = code
        context += f"\n\n// --- {step} ---\n{code}"
        print(f"[SPLIT] {step} done ({len(code)} chars)")

    progress.update(generation, percent_end - 8, f"{progress_label}ファイルを統合しています")

    def assemble(secs):
        return f"""{secs.get("imports_and_types", "")}

export default function {component_name}() {{
{secs.get("state_and_handlers", "")}

  return (
{secs.get("jsx_return", "")}
  );
}}
"""

    full_code = assemble(sections)

    progress.update(generation, percent_end - 4, f"{progress_label}生成内容を検証しています")

    for semantic_attempt in range(MAX_SEMANTIC_RETRIES + 1):
        check = _check_semantic_match(task, full_code)
        print(f"[SEMANTIC CHECK] matches={check['matches']} reason={check['reason']} problem_section={check['problem_section']}")

        if check["matches"]:
            break

        problem_section = check["problem_section"]
        if not problem_section or problem_section not in sections:
            print("[SEMANTIC CHECK] 問題箇所が特定できなかったため、再生成をスキップします。")
            break

        if semantic_attempt >= MAX_SEMANTIC_RETRIES:
            print("[SEMANTIC CHECK] 最大再試行回数に達したため、現状のコードのまま進めます。")
            break

        print(f"[SEMANTIC CHECK] {problem_section} を意味的な問題（{check['reason']}）を踏まえて再生成します。")
        progress.update(generation, percent_end - 4, f"{progress_label}{problem_section} を見直しています")

        retry_prompt = build_section_prompt(problem_section, task, context, reference_files=reference_files, knowledge_context=knowledge_context)
        retry_prompt += f"\n\n客観的なレビューで以下の問題が指摘されました。必ず修正してください：\n- {check['reason']}"
        section_result = ask_ai(retry_prompt, model="codellama")
        new_code = section_result.get("code", "")

        if problem_section == "state_and_handlers":
            new_code = _sanitize_state_section(new_code, component_name)
        elif problem_section == "jsx_return":
            new_code = _sanitize_jsx_section(new_code)

        sections[problem_section] = new_code
        full_code = assemble(sections)

    full_code = fix_react_imports(full_code)
    full_code = fix_component_props(full_code, sections.get("imports_and_types", ""), component_name)

    return {
        "component_name": component_name,
        "file_path": file_path,
        "full_code": full_code,
    }


def _split_code_to_sections(full_code: str, component_name: str) -> dict:
    """
    assemble()で組み立てたコードを3セクションに分解して返す。
    ループモードで特定セクションだけ差し替える際に使う。
    """
    sections = {
        "imports_and_types": "",
        "state_and_handlers": "",
        "jsx_return": "",
    }

    fn_decl = re.search(
        rf"export\s+default\s+function\s+{re.escape(component_name)}\s*\(",
        full_code
    )
    if fn_decl:
        sections["imports_and_types"] = full_code[:fn_decl.start()].strip()
        body = full_code[fn_decl.end():]
        brace_pos = body.find("{")
        if brace_pos != -1:
            inner = body[brace_pos + 1:]
            return_match = re.search(r"\n\s*return\s*\(", inner)
            if return_match:
                sections["state_and_handlers"] = inner[:return_match.start()].strip()
                jsx_part = inner[return_match.end():]
                last_paren = jsx_part.rfind(");")
                if last_paren != -1:
                    jsx_part = jsx_part[:last_paren]
                jsx_part = jsx_part.rstrip()
                if jsx_part.endswith("}"):
                    jsx_part = jsx_part[:-1]
                sections["jsx_return"] = jsx_part.strip()

    return sections


def _assemble_sections(sections: dict, component_name: str) -> str:
    return f"""{sections.get("imports_and_types", "")}

export default function {component_name}() {{
{sections.get("state_and_handlers", "")}

  return (
{sections.get("jsx_return", "")}
  );
}}
"""


def run_loop_generation(task, generation, max_iterations=3, reference_files=None):
    """
    生成→自己評価→修正をループするモード。
    _generate_one_component()で初回生成し、_check_semantic_match()で評価、
    NGなら問題セクションのみ再生成してもう一度評価する。
    """
    iterations = min(max_iterations, MAX_LOOP_ITERATIONS)
    print(f"[LOOP] 開始 max_iterations={iterations}")

    # 初回生成
    progress.update(generation, 5, "初回生成を開始しています")
    result = _generate_one_component(
        task, generation,
        percent_start=5, percent_end=60,
        reference_files=reference_files,
    )
    component_name = result["component_name"]
    file_path = result["file_path"]
    full_code = result["full_code"]

    # ループ評価・修正
    for i in range(1, iterations):
        pct = 60 + int(35 * (i / iterations))
        progress.update(generation, pct, f"自己評価中（{i}/{iterations}回目）")

        check = _check_semantic_match(task, full_code)
        print(f"[LOOP {i}] matches={check['matches']} reason={check['reason']}")

        if check["matches"]:
            print(f"[LOOP] {i}回目でOK。終了します。")
            break

        problem_section = check.get("problem_section")
        if not problem_section:
            print("[LOOP] 問題箇所が特定できないため終了します。")
            break

        progress.update(generation, pct + 2, f"修正中: {problem_section}（{i}/{iterations}回目）")
        print(f"[LOOP {i}] {problem_section} を修正します。理由: {check['reason']}")

        retry_prompt = build_section_prompt(problem_section, task, full_code, reference_files=reference_files)
        retry_prompt += f"\n\n以下の問題が指摘されました。必ず修正してください：\n- {check['reason']}"
        section_result = ask_ai(retry_prompt, model="codellama")
        new_code = section_result.get("code", "")

        if not new_code:
            print(f"[LOOP] {problem_section} の再生成コードが空のためスキップします。")
            continue

        if problem_section == "state_and_handlers":
            new_code = _sanitize_state_section(new_code, component_name)
        elif problem_section == "jsx_return":
            new_code = _sanitize_jsx_section(new_code)

        sections = _split_code_to_sections(full_code, component_name)
        sections[problem_section] = new_code
        full_code = _assemble_sections(sections, component_name)
        full_code = fix_react_imports(full_code)
        full_code = fix_component_props(full_code, sections.get("imports_and_types", ""), component_name)
    else:
        print(f"[LOOP] {iterations}回すべて完了。")

    suggestions = _generate_suggestions(task, f"loop generated {file_path}")
    return {
        "plan": [f"loop generation ({iterations} iterations): {task}"],
        "actions": [{"type": "write_file", "path": file_path, "content": full_code, "task": task}],
        "suggestions": suggestions,
    }


def run_split_generation(task, generation):
    result = _generate_one_component(task, generation, percent_start=10, percent_end=90)
    suggestions = _generate_suggestions(task, f"generated {result['file_path']}")
    return {
        "plan": [f"split generation: {task}"],
        "actions": [
            {
                "type": "write_file",
                "path": result["file_path"],
                "content": result["full_code"]
            }
        ],
        "suggestions": suggestions,
    }


MAX_MULTI_FILES = 3


def _connect_generated_components(generated_files, task, generation):
    if len(generated_files) < 2:
        return generated_files

    components_summary = "\n".join(
        f"- {name}: {result.get('description', '')}".rstrip(": ")
        for name, result in generated_files.items()
    )
    if not any(generated_files[name].get("description") for name in generated_files):
        components_summary = "\n".join(f"- {name}" for name in generated_files)

    progress.update(generation, 96, "コンポーネント同士の連携を検討しています")

    try:
        plan_result = ask_ai(
            build_connection_plan_prompt(components_summary, task),
            model="codellama",
        )
    except Exception as e:
        print(f"[CONNECT] 連携計画の取得に失敗しました（連携をスキップします）: {e}")
        return generated_files

    connections = plan_result.get("connections", [])
    if not connections:
        print("[CONNECT] 連携が必要なペアはないと判断されました。")
        return generated_files

    print(f"[CONNECT] connections={connections}")

    for conn in connections:
        parent_name = conn.get("parent")
        child_name = conn.get("child")
        relationship = conn.get("relationship", "")

        if not parent_name or not child_name:
            continue
        if parent_name not in generated_files or child_name not in generated_files:
            print(f"[CONNECT] {parent_name} または {child_name} が生成済みファイルに見つからないためスキップします。")
            continue
        if parent_name == child_name:
            continue

        progress.update(generation, 97, f"{parent_name} に {child_name} を組み込んでいます")

        parent_code = generated_files[parent_name]["full_code"]
        child_code = generated_files[child_name]["full_code"]

        try:
            connect_result = ask_ai(
                build_connect_components_prompt(
                    parent_name, parent_code, child_name, child_code, relationship
                ),
                model="codellama",
            )
            new_parent_code = connect_result.get("code", "")

            if not new_parent_code or not new_parent_code.strip():
                print(f"[CONNECT] {parent_name} の連携後コードが空だったため、元のコードのまま残します。")
                continue

            new_parent_code = fix_react_imports(new_parent_code)
            new_parent_code = fix_component_props(new_parent_code, new_parent_code, parent_name)

            generated_files[parent_name]["full_code"] = new_parent_code
            print(f"[CONNECT] {parent_name} に {child_name} を組み込みました。")
        except Exception as e:
            print(f"[CONNECT] {parent_name} と {child_name} の連携に失敗しました（元のコードのまま残します）: {e}")
            continue

    return generated_files


def run_multi_file_generation(task, generation, reference_files=None):
    progress.update(generation, 5, "(計画中) 必要なファイルを検討しています")
    plan_result = ask_ai(build_multi_file_plan_prompt(task, MAX_MULTI_FILES), model="codellama")

    files_plan = plan_result.get("files", [])
    if not files_plan:
        print("[MULTI] 計画が空だったため、単一ファイル生成にフォールバックします。")
        return run_split_generation(task, generation)

    files_plan = files_plan[:MAX_MULTI_FILES]
    total_files = len(files_plan)
    print(f"[MULTI] files_plan={files_plan}")

    generated_files = {}
    failed_files = []

    zone_start_overall = 5
    zone_end_overall = 95
    zone_span = (zone_end_overall - zone_start_overall) / total_files

    for i, file_plan in enumerate(files_plan):
        component_name = file_plan.get("component_name") or f"GeneratedComponent{i + 1}"
        description = file_plan.get("description", task)
        label = f"({i + 1}/{total_files}) {component_name}: "

        percent_start = int(zone_start_overall + zone_span * i)
        percent_end = int(zone_start_overall + zone_span * (i + 1))

        try:
            import os as _os
            _base = _os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
            _existing_path = _os.path.join(_base, "src", "pages", f"{component_name}.tsx")
            from run_split_generation_python import _check_existing_file
            print(f"[MULTI DEBUG] checking: {_existing_path}, exists={_os.path.exists(_existing_path)}")
            if _os.path.exists(_existing_path) and _check_existing_file(description, _existing_path):
                print(f"[MULTI] {component_name}: Existing file satisfies. Skipping.")
                with open(_existing_path, 'r', encoding='utf-8-sig') as _f:
                    _existing_code = _f.read()
                result = {
                    "full_code": _existing_code,
                    "file_path": f"src/pages/{component_name}.tsx",
                    "description": description,
                }
                generated_files[component_name] = result
                continue

            result = _generate_one_component(
                description,
                generation,
                component_name_hint=component_name,
                progress_label=label,
                percent_start=percent_start,
                percent_end=percent_end,
                reference_files=reference_files,
            )
            result["description"] = description
            generated_files[component_name] = result
            print(f"[MULTI] {component_name} 生成完了 ({len(result['full_code'])} chars)")
        except Exception as e:
            print(f"[MULTI] {component_name} の生成に失敗しました: {e}")
            failed_files.append({"component_name": component_name, "error": str(e)})

    generated_files = _connect_generated_components(generated_files, task, generation)

    progress.update(generation, 99, "生成したファイルをまとめています")

    actions = [
    {
        "type": "write_file",
        "path": result["file_path"],
        "content": result["full_code"],
        "task": task,
    }
    for result in generated_files.values()
    ]

    plan_summary = [f"multi-file generation: {task}"]
    plan_summary += [f"  - {name}" for name in generated_files.keys()]

    if failed_files:
        failed_names = "、".join(f["component_name"] for f in failed_files)
        actions.append({
            "type": "answer",
            "content": f"以下のファイルは生成に失敗しました：{failed_names}。お手数ですが個別に再度お試しください。",
        })

    if not actions:
        actions = [{"type": "answer", "content": "ファイルの生成にすべて失敗しました。もう一度お試しください。"}]

    suggestions = _generate_suggestions(task, f"generated {len(generated_files)} files: {', '.join(generated_files.keys())}")
    return {
        "plan": plan_summary,
        "actions": actions,
        "suggestions": suggestions,
    }


def run_html_mockup_generation(task, generation):
    import re as _re
    _m = _re.search(r'[\w\-]+\.html', task)
    if _m:
        from run_split_generation_python import _check_existing_file
        import os as _os
        _base = _os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
        _existing_path = _os.path.join(_base, "mockups", _m.group(0))
        if _check_existing_file(task, _existing_path):
            print(f"[HTML_MOCKUP] Existing file satisfies requirements. Skipping.")
            with open(_existing_path, 'r', encoding='utf-8-sig') as f:
                existing_code = f.read()
            return {
                "plan": [f"html mockup (existing): {task}"],
                "actions": [{"type": "write_file", "path": f"mockups/{_m.group(0)}", "content": existing_code}],
            }

    progress.update(generation, 30, "UIモックアップを生成しています")
    result = ask_ai(build_html_mockup_prompt(task), model="codellama")

    actions = result.get("actions", [])

    for action in actions:
        if action.get("type") == "write_file":
            path = action.get("path") or "mockup.html"
            filename = path.split("/")[-1]
            if not filename.lower().endswith(".html"):
                filename = re.sub(r"\.\w+$", "", filename) + ".html"
            action["path"] = f"mockups/{filename}"

    progress.update(generation, 90, "ファイルを保存しています")

    if not actions:
        actions = [{"type": "answer", "content": "モックアップの生成に失敗しました。もう一度試してください。"}]

    return {
        "plan": result.get("plan", [f"html mockup generation: {task}"]),
        "actions": actions,
    }


def _generate_suggestions(task: str, result_summary: str) -> list:
    prompt = f"""
以下のタスクが完了しました。次にやるべきことを3つ提案してください。

完了したタスク：
{task}

生成結果の概要：
{result_summary}

ルール：
- 具体的で実行可能な提案にすること
- 各提案は1行で、30文字以内にすること
- AI_001への指示文として使えるような形にすること

以下のJSONのみ返してください：
{{
  "suggestions": ["提案1", "提案2", "提案3"]
}}
"""
    try:
        result = ask_ai(prompt, model="router")
        suggestions = result.get("suggestions", [])
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:3]]
        return []
    except Exception as e:
        print(f"[SUGGESTIONS] 提案生成に失敗しました（無視して進めます）: {e}")
        return []


def _classify_intent(user: str) -> dict:
    intent_prompt = f"""
以下のユーザー入力の意図を判定してください。

ユーザー入力：
{user}

選択肢：
- "shobo_app": 消防点検管理システム(shobo-app)のReact画面・コンポーネントを実際に
  作ってほしいという依頼。「shobo-app」という言葉や、Customers.tsx/Staff.tsx/
  Settings.tsxのような具体的なファイル名が明示されている場合のみこれを選ぶこと。
- "html_mockup": shobo-app以外の、何らかのソフトウェアのUI画面・モックアップを
  作ってほしいという依頼（例：「会員登録画面を作って」「お問い合わせフォームを
  作って」「ログイン画面のHTMLを作って」）。
- "python_module": AI_001自身のバックエンド機能・ロジックをPythonコードとして
  実装してほしいという依頼（例：「自己改善ループを実装して」「新しいルーティング
  ロジックを追加して」「〜を判定する関数を作って」のような、画面ではなく
  サーバー側のPythonロジックの追加・変更を求める依頼）。
- "other": 上記のどちらにも当たらない依頼（通常の会話、質問、記憶、UI以外の
  簡単なコード生成など、まとまったPythonモジュールとしての実装が不要なもの）。

判定基準：
- 文章中に偶然それらしい単語（画面、コンポーネント、ファイル名など）が含まれて
  いるだけで、実際の作業依頼でない場合（例：資料の要約や引用）は "other" を選ぶこと。
- 「作って」「生成して」「実装して」のような明確な作業依頼があるかを重視すること。
- 依頼がReactの画面・UIに関するものなら shobo_app か html_mockup を優先して選ぶこと。
- 依頼がAI_001自体の振る舞い・処理ロジックに関するものなら python_module を選ぶこと。

以下のJSONのみ返してください：
{{
  "intent": "shobo_app" または "html_mockup" または "python_module" または "other",
  "reason": "判定理由を一言で"
}}
"""
    try:
        result = ask_ai(intent_prompt, model="router")
        intent = result.get("intent", "other")
        if intent not in ("shobo_app", "html_mockup", "python_module", "other"):
            intent = "other"
        return {"intent": intent, "reason": result.get("reason", "")}
    except Exception as e:
        print(f"[WARN] 意図判定に失敗しました（otherとして続行します）: {e}")
        return {"intent": "other", "reason": "判定失敗"}


def route(user, mem, generation, reference_files=None, image_base64=None, mode="normal", max_iterations=1):

    user = user.strip()
    name = mem.get("name")

    if "あなたの名前は" in user and ("?" in user or "？" in user):
        progress.update(generation, 50, "応答中")
        return {
            "plan": ["質問"],
            "actions": [{"type": "answer", "content": f"私は{name if name else '未設定'}です"}]
        }

    if "あなたの名前は" in user:
        progress.update(generation, 50, "記憶中")
        return {
            "plan": ["記憶"],
            "actions": [{"type": "remember", "key": "name", "value": user}]
        }

    # 画像が添付されている場合はvisionモデルで直接処理
    if image_base64:
        print("[DEBUG] → vision処理")
        progress.update(generation, 30, "画像を解析しています")
        vision_result = ask_ai(
            build_vision_prompt(user),
            model="vision",
            image_base64=image_base64,
        )
        content_text = ""
        if vision_result.get("actions"):
            content_text = vision_result["actions"][0].get("content", "")
        if not content_text:
            content_text = str(vision_result)
        return {
            "plan": ["vision: 画像解析"],
            "actions": [{"type": "answer", "content": content_text}],
        }

    # ループモードの場合はrun_loop_generationへ
    if mode == "loop":
        print(f"[DEBUG] → run_loop_generation (max_iterations={max_iterations})")
        return run_loop_generation(user, generation, max_iterations=max_iterations, reference_files=reference_files)

    progress.update(generation, 10, "依頼内容を判定しています")
    intent_check = _classify_intent(user)
    intent = intent_check["intent"]
    print(f"[INTENT] → {intent} ({intent_check['reason']})")

    if intent != "python_module" and _looks_like_python_module_request(user):
        print(f"[INTENT] キーワード判定により '{intent}' から 'python_module' に上書きします")
        intent = "python_module"

    if intent == "shobo_app":
        print("[DEBUG] → run_multi_file_generation")
        return run_multi_file_generation(user, generation, reference_files=reference_files)

    if intent == "html_mockup":
        print("[DEBUG] → run_html_mockup_generation")
        return run_html_mockup_generation(user, generation)

    if intent == "python_module":
        print("[DEBUG] → run_split_generation_python")
        import re as _re
        _m = _re.search(r'(?:ファイル名|module_name)\s*[:：]\s*(\w+)(?:\.py)?', user)
        _module_name_override = _m.group(1) if _m else None
        result = run_split_generation_python(user, generation, module_name_override=_module_name_override, reference_files=reference_files or [])
        suggestions = _generate_suggestions(user, f"generated {result['file_path']}")
        return {
            "plan": [f"Pythonモジュール {result['module_name']} を生成"],
            "actions": [{"type": "write_file", "path": result["file_path"], "content": result["full_code"], "task": user}],
            "answers": [],
            "generated_file_path": result["file_path"],
            "elapsed_seconds": result["elapsed_seconds"],
            "suggestions": suggestions,
        }

    progress.update(generation, 15, "振り分け先を判定しています")
    routing_prompt = f"""
以下のユーザー入力を分析して、どのモデルで処理すべきか判断してください。

ユーザー入力：
{user}

以下のJSONのみ返してください：
{{
  "model": "codellama" または "swallow",
  "reason": "理由を一言で"
}}

ルール：
- コード生成・プログラム作成・ファイル作成 → "codellama"
- 日本語の質問・会話・文章生成 → "swallow"
"""

    routing = ask_ai(routing_prompt, model="router")
    selected_model = routing.get("model", "swallow")
    print(f"\n[ROUTER] → {selected_model} ({routing.get('reason', '')})")

    progress.update(generation, 50, f"{selected_model} で生成しています")
    return ask_ai(build_prompt(user, mem), model=selected_model)