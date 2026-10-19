import re
import random   # ←追加

from build_prompt import (
    build_prompt,
    build_split_task_prompt,
    build_section_prompt,
    build_html_mockup_prompt,
    build_multi_file_plan_prompt,
    build_connection_plan_prompt,
    build_connect_components_prompt,
    build_vision_prompt,
    build_clarification_check_prompt,
)
from run_split_generation_python import run_split_generation_python
from bug_detection_handler import run_bug_detection_handler
from ai import ask_ai, ask_ai_with_rag, OllamaUnavailableError
import progress
import chat_history

# router役を任せられる「生徒」候補（明日から司令塔として頑張ってもらうメンバー）
ROUTER_STUDENT_POOL = ["router", "llama3_2", "gemma3", "granite", "lfm_jp"]


def pick_router_student() -> str:
    """router役の候補からランダムに1体選ぶ"""
    chosen = random.choice(ROUTER_STUDENT_POOL)
    print(f"[STUDENT] router役 → {chosen}")
    return chosen


def _extract_code_from_raw_text(text: str) -> str:
    """
    ask_ai() のJSONパースに失敗した際、{"plan": [], "actions": [{"type": "answer",
    "content": "..."}]} という形でフォールバックされることがある。この content が
    実は壊れたJSON文字列そのもの（例: '{\n  "code": "..."\n}'）である場合、それを
    そのままコードとして採用すると、生成ファイルにJSON記法の断片
    （"code": "..." など）がそのまま書き込まれてしまう事故が起きる
    （2026-07-17 InspectionResultList.tsx で実際に発生）。

    ここでは、textが {"code": "..."} 形式のJSONらしき文字列であれば、
    "code" フィールドの中身だけを正規表現で抜き出す。抜き出せなければ
    textをそのまま返す（今までの動作を壊さないため）。
    """
    if not text:
        return text

    stripped = text.strip()
    # 先頭が { で始まり、"code" というキーを含む場合のみJSONらしいとみなす
    if stripped.startswith("{") and '"code"' in stripped:
        # ★2026-07-30 追加: "code": "..." （二重引用符）だけでなく、
        # "code": `...` （バッククォート、テンプレートリテラル風）で
        # 出力されるケースもある（実例: TabSwitcher.tsx生成時）ため、
        # どちらの引用符でも抽出できるようにする。
        m = re.search(r'"code"\s*:\s*(["`])(.*)\1\s*\}?\s*$', stripped, re.DOTALL)
        if m:
            quote_char = m.group(1)
            extracted = m.group(2)
            if quote_char == '"':
                # JSON文字列としてエスケープされている改行・引用符を戻す
                extracted = extracted.replace("\\n", "\n").replace('\\"', '"')
            return extracted.strip()

    return text

print("ROUTER FILE LOADED:", __file__)

MAX_RETRIES_PER_SECTION = 2
MAX_SEMANTIC_RETRIES = 1
MAX_LOOP_ITERATIONS = 10
# ★2026-07-31 追加: 型チェック（_typecheck_code）専用の再試行回数。
# MAX_RETRIES_PER_SECTION（セクション単体の構文検証用）や
# MAX_SEMANTIC_RETRIES（意味チェック用）とは別枠で持つ。型チェックは
# tscの実行コストが高い（i7-3770環境で数秒〜数十秒）ため、他のチェックより
# 少ない回数に抑える。
MAX_TYPECHECK_RETRIES = 2

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


def _extract_declared_state_names(state_code: str) -> set:
    """
    state_and_handlersのコードから、useStateの変数名・handleXxxハンドラー名を集める。
    _validate_section（jsx_returnとの整合性チェック）と、
    _generate_one_component（意味チェック後の最終整合性チェック）の両方で使う。
    """
    declared_names = set()
    for m in re.finditer(r"const\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]\s*=\s*useState", state_code):
        declared_names.add(m.group(1))
        declared_names.add(m.group(2))
    for m in re.finditer(r"\b(?:const|function)\s+(handle\w+)\b", state_code):
        declared_names.add(m.group(1))
    return declared_names


def _find_component_declarations(code: str) -> set:
    """
    コード中に定義されているコンポーネントらしき宣言の名前をすべて集める。

    ★2026-07-30 追加: MainApp.tsx生成時にimports_and_typesへ無関係な
      Counterコンポーネントが丸ごと混入する事故が発生した。原因は
      function宣言のみを見ていた旧チェックがアロー関数形式
      （const Counter: React.FC<CounterProps> = ({...}) => {...}）を
      検出できなかったこと。関数コンポーネントは複数行にまたがる
      props分割代入（(\\{ \n  count,\n  ... \n\\}) => {）が非常に多く、
      「=>」まで正規表現で追いかけると改行を跨げず検出漏れするため、
      「const 大文字始まりの名前 ... = (」の開始部分だけで判定する
      （props分割代入がどう続くかは見なくてよい）。

    ★2026-07-30 追記: 上記を[A-Z]（英大文字）始まりに限定していたところ、
      「設定画面」「一般設定」のような日本語コンポーネント名（このアプリでは
      普通に使われる）が一切マッチせず、無関係コンポーネント混入チェックが
      日本語名のコンポーネントに対して丸ごと機能しない事故があった。
      「小文字英字(a-z)で始まらない識別子」に条件を緩め、PascalCase・
      日本語名の両方を拾いつつ、handleClickのようなcamelCaseハンドラーは
      引き続き除外する（(?![a-z])\\w+ ＝ 直後が小文字英字でないwordの並び）。
    """
    names = set()
    for m in re.finditer(r"^(?:export\s+(?:default\s+)?)?function\s+((?![a-z])\w+)\s*\(", code, re.MULTILINE):
        names.add(m.group(1))
    for m in re.finditer(r"^(?:export\s+(?:default\s+)?)?class\s+((?![a-z])\w+)\b", code, re.MULTILINE):
        names.add(m.group(1))
    for m in re.finditer(
        r"^(?:export\s+(?:default\s+)?)?const\s+((?![a-z])\w+)\s*(?::\s*(?:React\.)?FC\b[^=\n]*)?=\s*\(",
        code, re.MULTILINE
    ):
        names.add(m.group(1))
    return names


def _validate_connected_code(new_code: str, parent_name: str, child_name: str) -> list:
    """
    _connect_generated_components がLLMから受け取った「連携後の親コード」を
    採用してよいか検証する。

    ★2026-07-30 追加: 個別セクション生成（_validate_section）には検証・リトライが
      あるのに、連携（connect）ステップだけは結果を無条件で採用しており、
      壊れたコードでも上書きしてしまっていた。実例: TabBar.tsxに連携したところ、
      波括弧が2箇所閉じられておらず（interfaceブロック、if/elseブロック）、
      さらにchild（ErrorMessage）のコンポーネント定義がまるごと重複して
      混入し、親のhandleTabChange内で子スコープのsetErrorを参照する
      （実行不能な）コードになっていた。
    """
    problems = []
    if not new_code or not new_code.strip():
        problems.append("連携後のコードが空です。")
        return problems

    # 波括弧・丸括弧の対応が崩れていないか（簡易チェック。文字列・コメント内は
    # 区別しないが、閉じ忘れのような致命的な崩れを検出するには十分）
    if new_code.count("{") != new_code.count("}"):
        problems.append(
            f"波括弧の数が一致していません（{{: {new_code.count('{')}個, }}: {new_code.count('}')}個）。"
            "構文が壊れている可能性が高いです。"
        )
    if new_code.count("(") != new_code.count(")"):
        problems.append(
            f"丸括弧の数が一致していません（(: {new_code.count('(')}個, ): {new_code.count(')')}個）。"
            "構文が壊れている可能性が高いです。"
        )

    # parent自身の宣言が残っているか（_component_decl_patternは
    # "function Name(" の直後で改行される特定の書式しか想定しておらず、
    # 一般的な "function Name() {" 一行形式を誤って未検出とするため、
    # ここではより汎用的な _find_component_declarations を使う）
    declared = _find_component_declarations(new_code)
    if parent_name not in declared:
        problems.append(f"連携後のコードに {parent_name} 自体の宣言が見当たりません。")

    # childのコンポーネント定義がまるごと重複して埋め込まれていないか
    # （本来は import + <Child /> の参照だけであるべき）
    other_components = declared - {parent_name}
    if other_components:
        problems.append(
            f"連携後のコードに {parent_name} 以外のコンポーネント定義"
            f"（{', '.join(sorted(other_components))}）がそのまま埋め込まれています。"
            f"{child_name} は import して <{child_name} /> として使うだけにし、"
            "定義本体を重複させないでください。"
        )

    return problems


def _validate_section(section_name: str, code: str, component_name: str = "", task: str = "", context: str = "") -> list:
    """
    生成されたセクションのコードを検証し、問題点のリストを返す。

    ★2026-07-14 修正: 過去の編集で以下の事故が起きたため、この関数は
      「部分編集」ではなく「関数全体の丸ごと置き換え」でのみ更新すること。
      1. return文の直後に重複したelifブロックを追記 → SyntaxError
      2. 重複elifブロック（構文エラーなし版）→ 2つ目のjsx_return分岐が
         永久に到達しないデッドコード化
      3. bug_detection_handler呼び出し後の変数を計算するだけでproblemsに
         追加し忘れる → 検出処理が何もしない
      4. imports_and_typesのifブロックを誤って
         `if not task_mentions_counter:` の内側にネストしてしまい、
         state_and_handlers/jsx_returnのelifが永久に実行されなくなる
         （最も気づきにくい種類の事故だった）
    """
    problems = []

    if not code or not code.strip():
        problems.append("コードが空です。")
        return problems

    # ★2026-07-30 追加: 波括弧・丸括弧の対応が崩れていないかの簡易チェック。
    #   実例: GeneralSettingsTab.tsx生成時、if/elseブロックの閉じ忘れで
    #   handleReset がhandleSaveの内側にネストしてしまう構文崩れが
    #   検証を素通りしていた。文字列・コメント内は区別しないが、
    #   閉じ忘れのような致命的な崩れを検出するには十分。
    if code.count("{") != code.count("}"):
        problems.append(
            f"波括弧の数が一致していません（{{: {code.count('{')}個, }}: {code.count('}')}個）。"
            "if/elseブロックや関数の閉じ忘れがないか確認してください。"
        )
    if code.count("(") != code.count(")"):
        problems.append(
            f"丸括弧の数が一致していません（(: {code.count('(')}個, ): {code.count(')')}個）。"
        )

    # JSON記法の断片が紛れ込んでいないか
    if re.search(r'"\s*(code|jsx_return|imports_and_types|state_and_handlers)\s*"\s*:\s*', code):
        problems.append('出力にJSON記法の断片（"code": ... や "jsx_return": ... など）が紛れ込んでいます。コードのみを出力してください。')

    # プロンプト内の出力例（カウンターのサンプルコード）がそのままコピーされていないか
    example_leftover_terms = ["handleIncrement", "CounterState", "増やす", "setCount(count", "useState(0)"]
    task_lower = (task or "").lower()
    task_mentions_counter = any(t in task_lower for t in ["カウント", "count", "増やす", "カウンター"])
    if not task_mentions_counter:
        leaked = [t for t in example_leftover_terms if t in code]
        if leaked:
            problems.append(
                f"プロンプト内の出力例（カウンターのサンプルコード）がそのままコピーされています"
                f"（検出: {', '.join(leaked)}）。出力例は形式・粒度を示すためのサンプルに過ぎません。"
                "タスクの実際の内容に即した、独自の変数名・処理・表示内容に書き直してください。"
            )

    # ★2026-07-30 追加: component_name以外の無関係なコンポーネント定義が
    #   紛れ込んでいないか（セクションを問わず全体でチェックする）。
    if component_name:
        other_components = _find_component_declarations(code) - {component_name}
        if other_components:
            problems.append(
                f"{section_name} に、生成対象コンポーネント（{component_name}）とは無関係な"
                f"別のコンポーネント定義（{', '.join(sorted(other_components))}）が含まれています。"
                f"{component_name} 以外の新しいコンポーネントを定義しないでください。"
            )

    # ★2026-07-30 追加: このコードベースの組み立てテンプレート（assemble()参照）は
    #   常に `export default function {component_name}() {...}` という
    #   関数コンポーネント形式のみを前提にしている。クラスコンポーネントが
    #   混入すると、そのrender()内で使われていた `this.state.xxx` /
    #   `this.handleXxx` がそのままコピーされて関数コンポーネント側の
    #   state_and_handlers/jsx_returnに残ってしまうことがある（実例:
    #   class Tabs extends React.Component の混入から、SettingsTabsの
    #   JSXに this.state.activeTab / this.handleTabChange が紛れ込んだ）。
    #   関数コンポーネントに `this` は存在しないため、見つかったら即問題とする。
    if section_name in ("state_and_handlers", "jsx_return") and re.search(r"\bthis\.\w+", code):
        problems.append(
            "このセクションに `this.` を使った参照が含まれています。"
            "このアプリのコンポーネントは常に関数コンポーネント（Hooks形式、thisは使わない）"
            "なので、クラスコンポーネント風の this.state / this.ハンドラー名 という書き方をしないでください。"
        )

    # ★2026-07-30 追加: const X = (引数) => {...} 形式のハンドラー宣言で、
    #   引数に型注釈がない（暗黙のany型）場合を検出する。
    #   例: const handleTabChange = (tab) => {...} ← tabに型がない。
    #   TypeScriptの厳格設定では noImplicitAny によりコンパイルエラーになる。
    #   .map((item) => ...) のようなインラインコールバックはTS側の文脈型推論で
    #   実際にはエラーにならないため対象外とし、独立したconst宣言のみを見る。
    for m in re.finditer(r"(?:const|let)\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>", code):
        handler_name = m.group(1)
        params_str = m.group(2).strip()
        if not params_str or params_str.startswith("{") or params_str.startswith("["):
            continue
        untyped = []
        for p in params_str.split(","):
            p = p.strip()
            if not p or ":" in p:
                continue
            untyped.append(p.split("=")[0].strip())
        if untyped:
            problems.append(
                f"'{handler_name}' の引数（{', '.join(untyped)}）に型注釈がありません（暗黙のany型）。"
                f"例: const {handler_name} = ({untyped[0]}: string) => {{...}} のように型を明記してください。"
            )

    # ★ ここから下は、上の「カウンターコピペ検出」とは独立した、
    #   section_name 別のif/elifチェーン。インデントは必ずこの関数の
    #   本体（4スペース）に揃えること。上の if not task_mentions_counter:
    #   ブロックの内側に絶対にネストしないこと。
    if section_name == "imports_and_types":
            has_function_def = re.search(r"^(export\s+default\s+)?function\s+\w+\s*\(", code, re.MULTILINE)
            has_class_def = re.search(r"^(export\s+default\s+)?class\s+\w+", code, re.MULTILINE)
            # ★2026-07-30 修正: 旧正規表現は「=>」が同じ行にあることを前提にしていたため、
            #   props分割代入が複数行にまたがるアロー関数コンポーネント
            #   （const Counter: React.FC<CounterProps> = ({\n  count,\n}) => {...}）を
            #   検出できずスルーしていた。「const 大文字名 ... = (」の開始だけで判定する
            #   _find_component_declarations と、明示的な FC 型注釈の直接検出で置き換える。
            has_arrow_component = bool(_find_component_declarations(code))
            has_fc_annotation = re.search(r":\s*(React\.)?FC\b", code)
            if has_function_def or has_class_def or has_arrow_component or has_fc_annotation:
                problems.append(
                    "imports_and_types に関数コンポーネントまたはクラスコンポーネントの定義が含まれています"
                    "（function/class/アロー関数によるコンポーネント宣言）。"
                    "importと型定義（interface/type）のみにしてください。コンポーネント本体は別のセクションで生成されます。"
                )

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
        if re.search(r'\bjsx_return\b', code):
            problems.append('コード中に "jsx_return" という単語（セクション名）がそのまま含まれています。これは指示のラベルであり、コードの一部ではありません。削除してください。')

        # .map() で使われている変数が state_and_handlers で宣言されているか
        used_vars = set(re.findall(r'\b([a-zA-Z_]\w*)\s*\.\s*map\(', code))
        for v in used_vars:
            if not re.search(rf'\b(const|let|var)\s+\[?{re.escape(v)}\b', context) and v not in context:
                problems.append(f"jsx_return で '{v}' を使っていますが、state_and_handlers で宣言されていません。")

        # onClick={handleXxx} のようなハンドラー参照が state_and_handlers で宣言されているか
        used_handlers = set(re.findall(r'=\{(\w+)\}', code))
        for h in used_handlers:
            if h.startswith('handle') and not re.search(rf'\b(const|let|var)\s+{re.escape(h)}\b', context):
                problems.append(f"jsx_return で '{h}' を使っていますが、state_and_handlers で宣言されていません。")

        # ★2026-07-30 追加: 逆方向のチェック。state_and_handlersで宣言した
        #   state/ハンドラーが、jsx_returnに1つも登場しない場合を検出する。
        #   （実例: isModalOpen用のhandlerを作ったのに、jsx側は全く別の
        #   「AI機能リスト」を出力していた＝両セクションの内容が完全に
        #   噛み合っていなかった事故）。個々の変数が未使用というだけでは
        #   誤検知が多いため、宣言した名前が「1つも」使われていない場合のみ
        #   問題として報告する。
        state_section_match = re.search(
            r"// --- state_and_handlers ---\n(.*?)(?=\n// --- |\Z)", context, re.DOTALL
        )
        if state_section_match:
            state_code = state_section_match.group(1)
            declared_names = _extract_declared_state_names(state_code)
            if declared_names and code.strip():
                used_in_jsx = any(re.search(rf"\b{re.escape(name)}\b", code) for name in declared_names)
                if not used_in_jsx:
                    problems.append(
                        "state_and_handlers で宣言した状態・ハンドラー（"
                        f"{', '.join(sorted(declared_names))}）が jsx_return で一切使われていません。"
                        "state_and_handlersとjsx_returnの内容が噛み合っていない可能性があります。"
                    )

        # bug_detection_handler による追加チェック（event未定義参照・FormEvent型なのに<form>未使用）
        has_form_event_handler = bool(re.search(r"React\.FormEvent<HTMLFormElement>", context))
        bug_check = run_bug_detection_handler(
            code=code,
            component_name=component_name or "Unknown",
            expected_name=component_name or "Unknown",
            required_imports=[],
            has_form_event_handler=has_form_event_handler,
        )
        problems.extend(bug_check.get("problems", []))

    return problems

def _generate_section_with_retry(section_name, task, context, component_name="", reference_files=None, knowledge_context="", mem=None):
    """
    1セクションを生成し、_validate_section で問題が見つかった場合は再生成する。

    ★ mem を追加：社訓（memory.json の "motto"）をプロンプトに反映させるため、
      build_section_prompt に mem を渡す。mem は呼び出し元（_generate_one_component）
      経由で route() から渡される。mem が None の場合は build_section_prompt 側の
      デフォルト社訓が使われるため、未対応の呼び出し元があっても壊れない。
    """
    feedback = ""
    code = ""

    for attempt in range(MAX_RETRIES_PER_SECTION + 1):
        prompt = build_section_prompt(
            section_name, task, context,
            reference_files=reference_files,
            knowledge_context=knowledge_context,
            mem=mem,
        )
        if feedback:
            prompt += f"\n\n前回の出力には以下の問題がありました。必ず修正してください：\n{feedback}"

        print(f"[SPLIT] generating: {section_name} (attempt {attempt + 1}/{MAX_RETRIES_PER_SECTION + 1})")
        section_result = ask_ai(prompt, model="codellama")
        print(f"[SPLIT DEBUG] section_result={section_result}")

        code = section_result.get("code", "")
        # ★2026-07-30 追加: ai.pyの_parse_responseは "code" が文字列でない
        #   （モデルがネストしたオブジェクトを返した等）場合、そのまま
        #   dict等を返すことがある。ここでガードしないと後続の
        #   code.strip()（_validate_section内）で
        #   AttributeError: 'dict' object has no attribute 'strip' が発生し、
        #   MULTI生成全体が失敗する（実例: Animations.tsx生成時に発生）。
        if not isinstance(code, str):
            print(f"[SPLIT] section_result['code'] が文字列ではありません（type={type(code).__name__}）。空として扱います。")
            code = ""
        if not code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    code = _extract_code_from_raw_text(a.get("content"))
                    break

        problems = _validate_section(section_name, code, component_name, task, context)
        if not problems:
            return code

        print(f"[SPLIT] {section_name} に問題あり: {problems}")
        feedback = "\n".join(f"- {p}" for p in problems)

    print(f"[WARN] {section_name} は{MAX_RETRIES_PER_SECTION + 1}回試行しても問題が解決しませんでした。")

    # ★2026-07-30 追加: 諦める前の最後の手段。問題が「JSON記法の断片が
    # 紛れ込んでいる」ことだった場合、_extract_code_from_raw_text による
    # 自動抽出を試みる。実例: TabSwitcher.tsxのstate_and_handlersが
    # 3回とも {"code": `...`} 形式（バッククォート版）で出力され続け、
    # 抽出されないまま採用されて壊れたコードになっていた。
    if code and any("JSON記法の断片" in p for p in problems):
        salvaged = _extract_code_from_raw_text(code)
        if salvaged and salvaged != code:
            salvage_problems = _validate_section(section_name, salvaged, component_name, task, context)
            if not any("JSON記法の断片" in p for p in salvage_problems):
                print(f"[SPLIT] {section_name} をJSON断片から自動抽出して救済しました。")
                code = salvaged
                problems = salvage_problems

    # ★2026-07-30 追加: 諦める前のもう一つの最後の手段。波括弧がちょうど1個
    # 分だけ閉じ忘れている場合（実例: MainLayout.tsxで
    # `interface TabState {\n  activeTab: string;` のまま3回とも
    # 閉じ忘れて確定していた）、末尾に `}` を1個補って再検証する。
    # 2個以上の不一致は補う位置の推測が信頼できないため対象外とする。
    if code and any("波括弧の数が一致していません" in p for p in problems):
        net_open = code.count("{") - code.count("}")
        if net_open == 1:
            salvaged = code.rstrip() + "\n}\n"
            salvage_problems = _validate_section(section_name, salvaged, component_name, task, context)
            if not any("波括弧の数が一致していません" in p for p in salvage_problems):
                print(f"[SPLIT] {section_name} の波括弧の閉じ忘れ（1個分）を末尾に補って救済しました。")
                code = salvaged
                problems = salvage_problems

    if section_name == "state_and_handlers" and (not code or not code.strip()):
        code = "  // 生成失敗：状態管理・ハンドラーが生成されませんでした（要手動修正）"
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
        result = ask_ai(prompt, model=pick_router_student())
        return {
            "matches": bool(result.get("matches", True)),
            "reason": result.get("reason", ""),
            "problem_section": result.get("problem_section"),
        }
    except OllamaUnavailableError:
        print("[WARN] セマンティックチェック中にOllama接続エラー。チェックをスキップします。")
        return {"matches": True, "reason": "チェックスキップ（Ollama未起動）", "problem_section": None}
    except Exception as e:
        print(f"[WARN] 意味チェックに失敗しました（無視して進めます）: {e}")
        return {"matches": True, "reason": "検証スキップ", "problem_section": None}


def _typecheck_code(full_code: str, file_path: str) -> dict:
    """
    生成されたコードを実際にshobo-appプロジェクト内に一時的に書き出し、
    `npx tsc --noEmit -p <tsconfig.json>` を実行して型・import解決を検証する。

    ★2026-07-31 追加: _validate_section（構文レベル）と _check_semantic_match
      （タスク内容との意味的な整合性）は既にあったが、どちらも「実際にTypeScript
      として型が通るか」は見ていなかった。存在しないファイル・関数名の微妙な
      誤りなど、構文的には正しく意味的にもタスクに沿って見えるが型としては
      壊れているコードがすり抜けていたため、実際のコンパイラで確認する。

      ファイル単体を tsc に渡すと import 解決（他コンポーネント・db.ts等の参照）
      に失敗して偽陽性だらけになるため、必ずプロジェクト全体を対象にする
      （-p オプションでtsconfig.jsonを指定）。

    引数:
      full_code: 検証したい統合済みコード（imports_and_types + state_and_handlers +
                 jsx_return を組み立てたもの）
      file_path: "src/pages/ComponentName.tsx" のような、プロジェクトルートからの
                 相対パス（_generate_one_component が組み立てる file_path をそのまま渡す想定）

    戻り値:
      {"has_errors": bool, "errors": [str, ...]}
      対象ファイルに関するエラー行（tscの出力形式:
      path/to/file.tsx(行,列): error TSxxxx: メッセージ）だけを抽出して返す。

    ★タイムアウト・tsc未検出などの環境要因で判定できなかった場合は、
      _check_semantic_match と同じフォールバック方針（判定不能なら生成を
      止めない）に合わせ、has_errors=False を返して処理を継続する。
      ただしログには何が起きたか必ず残す。
    """
    import os as _os
    import subprocess as _subprocess

    _base = _os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
    _tsconfig_path = _os.path.join(_base, "tsconfig.json")
    _target_path = _os.path.join(_base, file_path.replace("/", _os.sep))

    if not _os.path.exists(_tsconfig_path):
        print(f"[TYPECHECK] tsconfig.jsonが見つかりません（{_tsconfig_path}）。型チェックをスキップします。")
        return {"has_errors": False, "errors": []}

    try:
        _os.makedirs(_os.path.dirname(_target_path), exist_ok=True)
        with open(_target_path, "w", encoding="utf-8") as f:
            f.write(full_code)
    except Exception as e:
        print(f"[TYPECHECK] 一時書き出しに失敗しました（{_target_path}, {e}）。型チェックをスキップします。")
        return {"has_errors": False, "errors": []}

    # ★Windowsではnpxが npx.cmd（バッチファイル）のため、shell=True かつ
    #   コマンドを1本の文字列で渡す方式が一番事故が少ない
    #   （引数リスト+shell=Trueの組み合わせはWindowsで解釈がずれることがある）。
    _cmd = f'npx tsc --noEmit -p "{_tsconfig_path}"'
    try:
        # ★2026-08-05修正: text=True だけだと、Windows既定のロケール(cp932)で
        #   出力をデコードしようとする。tscの実際の出力にはcp932でデコードできない
        #   バイト列が混ざることがあり、その場合subprocess内部の読み取りスレッド
        #   (_readerthread)がUnicodeDecodeErrorで静かに落ちる。この例外は
        #   バックグラウンドスレッドで発生するため、ここのtry/exceptでは
        #   捕まえられず、proc.stdout/stderrが空のまま「型エラー0件」と
        #   誤判定されて型チェックが常にスルーされる不具合があった
        #   (実例: test_typecheck.pyで確認、わざと壊したコードでもhas_errors=False)。
        #   encoding/errorsを明示することで解消する。
        proc = _subprocess.run(
            _cmd,
            cwd=_base,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            shell=True,
        )
    except _subprocess.TimeoutExpired:
        print("[TYPECHECK] tscの実行が60秒でタイムアウトしました。型チェックタイムアウトのためスキップして進めます。")
        return {"has_errors": False, "errors": []}
    except FileNotFoundError:
        print("[TYPECHECK] npx/tscが見つかりません（node.js未インストール等）。型チェックをスキップします。")
        return {"has_errors": False, "errors": []}
    except Exception as e:
        print(f"[TYPECHECK] tsc実行中に予期しないエラーが発生しました（{e}）。型チェックをスキップします。")
        return {"has_errors": False, "errors": []}

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # 対象ファイルに関するエラー行だけを抽出する。basenameだけで照合するのは、
    # tscが相対パスをどう正規化するか（/ か \ か、先頭に./が付くか等）が
    # 環境依存でぶれるため、ファイル名一致の方が確実だから。
    _target_basename = _os.path.basename(file_path)
    errors = []
    for line in output.splitlines():
        line = line.strip()
        m = re.match(r"^(.*?\.tsx?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)$", line)
        if m and _target_basename in m.group(1):
            errors.append(line)

    if errors:
        print(f"[TYPECHECK] {file_path} に型エラーが{len(errors)}件見つかりました。")
        return {"has_errors": True, "errors": errors}

    print(f"[TYPECHECK] {file_path} 型チェックOK。")
    return {"has_errors": False, "errors": []}


def _generate_one_component(
    task,
    generation,
    component_name_hint=None,
    progress_label="",
    percent_start=10,
    percent_end=90,
    reference_files=None,
    mem=None,
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
    knowledge_context = ""

    # ★2026-08-02 追加：対象ファイルが既に存在する場合、その内容を読み込んで
    # contextに含める。これまでは既存ファイルの有無を一切見ずに、指示文だけを
    # 頼りに毎回ゼロから組み立て直していたため、「バグを直して」のような依頼で
    # 既存の機能（一覧表示など）ごと消えてしまう事故が起きていた
    # （実例：2026-08-02 MeetingRoomList.tsx。指摘文だけを見て、未定義エラーの
    # 原因になっていた一覧表示部分ごと丸ごと削除して「修正」してしまった）。
    # run_multi_file_generation側の_check_existing_file（満たしていれば
    # スキップ）とは別物：あちらは「作り直す必要があるか」の判定だけに使い、
    # 既存コードの中身自体はここで初めてAIに渡す。
    try:
        import os as _os
        _shobo_base = _os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
        _existing_full_path = _os.path.join(_shobo_base, file_path.replace("/", _os.sep))
        if _os.path.exists(_existing_full_path):
            with open(_existing_full_path, "r", encoding="utf-8-sig") as f:
                _existing_code = f.read()
            if _existing_code.strip():
                print(f"[SPLIT] 既存ファイルを検出しました（{file_path}, {len(_existing_code)}文字）。これを土台に生成します。")
                context += (
                    "\n\n【重要：これが現在のファイルの内容です】\n"
                    "このファイルは既に存在しています。以下がその現在の中身です。"
                    "タスクで明示的に削除・変更するよう指示されていない機能・部分は、"
                    "絶対に削除しないでください。バグ修正の依頼であっても、"
                    "該当箇所以外の既存の機能（一覧表示・状態管理など）はそのまま残し、"
                    "この内容を土台にして修正・拡張してください。\n"
                    f"{_existing_code}"
                )
    except Exception as e:
        print(f"[SPLIT] 既存ファイルの確認に失敗しました（無視して新規生成として続行）: {e}")

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

        code = _generate_section_with_retry(
            step, task, context, component_name,
            reference_files=reference_files,
            knowledge_context=knowledge_context,
            mem=mem,
        )

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
        new_code = _extract_code_from_raw_text(section_result.get("code", ""))
        if not new_code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    new_code = _extract_code_from_raw_text(a.get("content"))
                    break

        if problem_section == "state_and_handlers":
            new_code = _sanitize_state_section(new_code, component_name)
        elif problem_section == "jsx_return":
            new_code = _sanitize_jsx_section(new_code)

        sections[problem_section] = new_code
        full_code = assemble(sections)

    # ★2026-07-30 追加: 意味チェックのループが「片方のセクションだけ」を
    # 再生成することで、既に確定していたもう片方との整合性が崩れる事故への対処。
    # 実例: state_and_handlersがsearchTerm/handleSearchInputChangeに再生成
    # されたのに、jsx_returnは古いvalue/handleInputChangeのまま残り、
    # 未定義変数を参照する壊れたコードが確定してしまった
    # （意味チェックはこの不整合を検出していたが、MAX_SEMANTIC_RETRIESを
    # 使い切っていたため「現状のコードのまま進めます」で見逃されていた）。
    # ループが終わったあと、最終的なstate_and_handlersとjsx_returnが
    # 噛み合っているかをもう一度だけ確認し、ダメならjsx_returnを
    # 最終版のstate_and_handlersに合わせて1回だけ作り直す。
    final_state_code = sections.get("state_and_handlers", "")
    final_jsx_code = sections.get("jsx_return", "")
    declared_names = _extract_declared_state_names(final_state_code)
    if declared_names and final_jsx_code.strip():
        used_in_jsx = any(re.search(rf"\b{re.escape(name)}\b", final_jsx_code) for name in declared_names)
        if not used_in_jsx:
            print("[RECONCILE] state_and_handlersとjsx_returnが噛み合っていないため、jsx_returnを最終版に合わせて作り直します。")
            progress.update(generation, percent_end - 3, f"{progress_label}最終確認：jsx_returnを調整しています")

            reconcile_context = ""
            for step_name in ("imports_and_types", "state_and_handlers"):
                if step_name in sections:
                    reconcile_context += f"\n\n// --- {step_name} ---\n{sections[step_name]}"

            reconcile_prompt = build_section_prompt(
                "jsx_return", task, reconcile_context,
                reference_files=reference_files, knowledge_context=knowledge_context,
            )
            reconcile_prompt += (
                "\n\n重要: state_and_handlersで宣言されている状態・ハンドラー（"
                f"{', '.join(sorted(declared_names))}）を実際に使ったJSXにしてください。"
                "別の名前の変数を新たに作らないでください。"
            )
            section_result = ask_ai(reconcile_prompt, model="codellama")
            new_jsx = _extract_code_from_raw_text(section_result.get("code", ""))
            if not new_jsx and "actions" in section_result:
                for a in section_result.get("actions", []):
                    if a.get("content"):
                        new_jsx = _extract_code_from_raw_text(a.get("content"))
                        break
            new_jsx = _sanitize_jsx_section(new_jsx)

            # 作り直しても直らないなら、無理に上書きせず元のコードのまま残す
            if new_jsx.strip() and any(re.search(rf"\b{re.escape(n)}\b", new_jsx) for n in declared_names):
                sections["jsx_return"] = new_jsx
                full_code = assemble(sections)
                print("[RECONCILE] jsx_returnを最終版のstate_and_handlersに合わせて更新しました。")
            else:
                print("[RECONCILE] 作り直しても整合しなかったため、元のコードのまま進めます。")

    full_code = fix_react_imports(full_code)
    full_code = fix_component_props(full_code, sections.get("imports_and_types", ""), component_name)

    # ★2026-07-31 追加: 型チェック＆リトライ（自己検証機構）。
    # ここまでの _validate_section / _check_semantic_match は構文・意味レベルの
    # チェックに留まり、実際にTypeScriptとして型が通るか・importが正しいかは
    # 見ていなかった。ここで実際にtscを走らせて確認し、エラーがあれば
    # そのエラー内容をフィードバックとしてセクションを再生成する。
    # importエラーが多いため imports_and_types も再生成対象に含める。
    # MAX_TYPECHECK_RETRIES回試してもダメなら、_check_semantic_matchと同じ
    # フォールバック方針（諦めて現状のコードのまま進める）に合わせる。
    progress.update(generation, percent_end - 2, f"{progress_label}型チェック中")
    for typecheck_attempt in range(MAX_TYPECHECK_RETRIES + 1):
        typecheck_result = _typecheck_code(full_code, file_path)
        if not typecheck_result["has_errors"]:
            break

        if typecheck_attempt >= MAX_TYPECHECK_RETRIES:
            print(f"[TYPECHECK] {MAX_TYPECHECK_RETRIES}回試しても型エラーが解決しませんでした。エラーを残したまま現状のコードで進めます。")
            print(f"[TYPECHECK] 残存エラー:\n" + "\n".join(typecheck_result["errors"]))

            # ★2026-08-02 追加: 型エラーが解決しなかった経験を outcome="failure" として保存する。
            # これまでは _save_experience_async（executor.py）が書き込み成功時にしか
            # 経験を保存しておらず、失敗パターンがLAの記憶に一切残らなかった
            # （同じミスを繰り返す原因になっていた）。ここでは「何回リトライしても
            # 直らなかった型エラー」という、システムが自動検出できる失敗を記録する。
            try:
                from memory_store import save_experience
                import threading

                residual_errors = "\n".join(f"- {e}" for e in typecheck_result["errors"])

                def _save_failure():
                    try:
                        save_experience(
                            task=task,
                            code=full_code,
                            outcome="failure",
                            learned=f"型チェックが{MAX_TYPECHECK_RETRIES}回のリトライでも解決しなかった。残存エラー:\n{residual_errors}",
                            reflection="次回似たタスクをやる時は、このエラーパターンを避けるよう気をつける。",
                            file_path=file_path,
                        )
                    except Exception as inner_e:
                        print(f"[MEMORY] failure経験の保存に失敗しました（無視して続行）: {inner_e}")

                threading.Thread(target=_save_failure, daemon=True).start()
            except Exception as e:
                print(f"[MEMORY] failure経験の保存準備に失敗しました（無視して続行）: {e}")

            break

        print(f"[TYPECHECK] 型エラーを踏まえて再生成します（{typecheck_attempt + 1}/{MAX_TYPECHECK_RETRIES}回目）:\n" + "\n".join(typecheck_result["errors"]))
        progress.update(
            generation, percent_end - 2,
            f"{progress_label}型エラーを修正しています（{typecheck_attempt + 1}/{MAX_TYPECHECK_RETRIES}回目）",
        )

        typecheck_feedback = "\n".join(f"- {e}" for e in typecheck_result["errors"])

        for retry_section in ("imports_and_types", "state_and_handlers", "jsx_return"):
            retry_prompt = build_section_prompt(
                retry_section, task, context,
                reference_files=reference_files, knowledge_context=knowledge_context, mem=mem,
            )
            retry_prompt += (
                "\n\n前回の出力を実際にTypeScriptコンパイラ（tsc）でチェックしたところ、"
                f"以下の型エラーが出ました。必ず修正してください：\n{typecheck_feedback}"
            )
            section_result = ask_ai(retry_prompt, model="codellama")
            new_code = _extract_code_from_raw_text(section_result.get("code", ""))
            if not new_code and "actions" in section_result:
                for a in section_result.get("actions", []):
                    if a.get("content"):
                        new_code = _extract_code_from_raw_text(a.get("content"))
                        break
            if not new_code:
                # 再生成が空だった場合は、そのセクションだけ元のコードを維持する
                continue

            if retry_section == "state_and_handlers":
                new_code = _sanitize_state_section(new_code, component_name)
            elif retry_section == "jsx_return":
                new_code = _sanitize_jsx_section(new_code)

            sections[retry_section] = new_code

        full_code = assemble(sections)
        full_code = fix_react_imports(full_code)
        full_code = fix_component_props(full_code, sections.get("imports_and_types", ""), component_name)

    return {
        "component_name": component_name,
        "file_path": file_path,
        "full_code": full_code,
    }


def _split_code_to_sections(full_code: str, component_name: str) -> dict:
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


def run_loop_generation(task, generation, max_iterations=3, reference_files=None, mem=None):
    try:
        return _run_loop_generation_inner(task, generation, max_iterations, reference_files, mem=mem)
    except OllamaUnavailableError as e:
        msg = f"Ollamaが起動していません。起動してから再送してください。\n詳細: {e}"
        print(f"[ERROR] run_loop_generation: {msg}")
        progress.fail(generation, msg)
        return {"plan": [], "actions": [{"type": "answer", "content": msg}]}


def _run_loop_generation_inner(task, generation, max_iterations=3, reference_files=None, mem=None):
    iterations = min(max_iterations, MAX_LOOP_ITERATIONS)
    print(f"[LOOP] 開始 max_iterations={iterations}")

    progress.update(generation, 5, "初回生成を開始しています")
    result = _generate_one_component(
        task, generation,
        percent_start=5, percent_end=60,
        reference_files=reference_files,
        mem=mem,
    )
    component_name = result["component_name"]
    file_path = result["file_path"]
    full_code = result["full_code"]

    check = None
    semantic_match_failed = False

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
        new_code = _extract_code_from_raw_text(section_result.get("code", ""))
        if not new_code and "actions" in section_result:
            for a in section_result.get("actions", []):
                if a.get("content"):
                    new_code = _extract_code_from_raw_text(a.get("content"))
                    break

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
        semantic_match_failed = True

    # 【追加・2026-08-05】_check_semantic_matchがセクションを部分的に作り直した後、
    # 型チェック(_typecheck_code、_generate_one_component内では既に使っていた)を
    # 一度もやり直していなかった。そのため「意味的には一致」と誤判定されつつ、
    # 実際には未定義変数などの構造的なバグがパッチ時に紛れ込むケースがあった
    # (実例: 2026-08-05、次回点検日計算コンポーネントでlastInspectionDate等が
    # 宣言されずに使われ、会議室にも行かずそのまま出てしまった)。
    # ループを抜けた直後(早期break・exhaust問わず)に必ず型チェックし直す。
    progress.update(generation, 90, "自己修正後の型チェック中")
    typecheck_result = {"has_errors": False, "errors": []}
    for typecheck_attempt in range(MAX_TYPECHECK_RETRIES + 1):
        typecheck_result = _typecheck_code(full_code, file_path)
        if not typecheck_result["has_errors"]:
            break
        if typecheck_attempt >= MAX_TYPECHECK_RETRIES:
            print(f"[LOOP-TYPECHECK] {MAX_TYPECHECK_RETRIES}回試しても型エラーが解決しませんでした。")
            break

        print(f"[LOOP-TYPECHECK] 自己修正後に型エラーを検出、再修正します（{typecheck_attempt + 1}/{MAX_TYPECHECK_RETRIES}回目）:\n" + "\n".join(typecheck_result["errors"]))
        progress.update(generation, 90, f"型エラーを修正しています（{typecheck_attempt + 1}/{MAX_TYPECHECK_RETRIES}回目）")

        typecheck_feedback = "\n".join(f"- {e}" for e in typecheck_result["errors"])
        sections = _split_code_to_sections(full_code, component_name)
        for retry_section in ("imports_and_types", "state_and_handlers", "jsx_return"):
            retry_prompt = build_section_prompt(retry_section, task, full_code, reference_files=reference_files)
            retry_prompt += (
                "\n\n前回の出力を実際にTypeScriptコンパイラ（tsc）でチェックしたところ、"
                f"以下の型エラーが出ました。必ず修正してください：\n{typecheck_feedback}"
            )
            section_result = ask_ai(retry_prompt, model="codellama")
            new_code = _extract_code_from_raw_text(section_result.get("code", ""))
            if not new_code and "actions" in section_result:
                for a in section_result.get("actions", []):
                    if a.get("content"):
                        new_code = _extract_code_from_raw_text(a.get("content"))
                        break
            if not new_code:
                continue
            if retry_section == "state_and_handlers":
                new_code = _sanitize_state_section(new_code, component_name)
            elif retry_section == "jsx_return":
                new_code = _sanitize_jsx_section(new_code)
            sections[retry_section] = new_code
        full_code = _assemble_sections(sections, component_name)
        full_code = fix_react_imports(full_code)
        full_code = fix_component_props(full_code, sections.get("imports_and_types", ""), component_name)

    # 【追加・2026-08-05】「困ったら会議室」。semantic_match_failed(意味チェックが
    # 最後まで通らなかった)、または型エラーが最終的に残った場合に招集する。
    # 型エラーの方が客観的で信頼できる根拠なので、両方まとめて1回の会議で
    # 判断を仰ぐ(無駄に2回開かない)。会議は各回を完全に独立させて聞く
    # (meeting_vote.py参照、アンカリング防止)。
    human_note = None
    if semantic_match_failed or typecheck_result["has_errors"]:
        from meeting_vote import convene_meeting

        problems = []
        if semantic_match_failed:
            reason = check["reason"] if check else "詳細不明"
            problems.append(f"自己評価が最後まで「要件と一致」と判定しませんでした（直近の理由: {reason}）")
        if typecheck_result["has_errors"]:
            problems.append(
                "TypeScriptの型チェック(tsc)でエラーが解決しませんでした:\n"
                + "\n".join(f"  - {e}" for e in typecheck_result["errors"])
            )

        print(f"[LOOP] 会議室を招集します。理由: {'; '.join(problems)}")
        meeting_question = (
            f"コード生成タスク「{task}」について、{iterations}回の自己修正を試みましたが、"
            "以下の問題が残りました。このまま今の内容(ベストエフォート)で確定してよいか、"
            "それとも人に確認してもらうべきか判断してください。"
        )
        meeting_context = "\n".join(problems)
        meeting_result = convene_meeting(meeting_question, context=meeting_context, generation=generation)

        if meeting_result["needs_human"]:
            print(f"[LOOP] 会議室でも結論が出ませんでした。人に確認を求めます。ログ: {meeting_result['log_path']}")
            human_note = (
                f"⚠️「{task}」の自動生成に問題が残りました(会議室(教室の生徒{meeting_result['total_votes']}体)でも"
                f"結論が出ませんでした)。今の内容はベストエフォートです。確認をお願いします。"
                f"(会議ログ: {meeting_result['log_path']})"
            )
        else:
            print(f"[LOOP] 会議室の結論: {meeting_result['verdict']}（{meeting_result['vote_count']}/{meeting_result['total_votes']}票）")
            human_note = (
                f"ℹ️「{task}」の自動生成に問題が残りましたが、"
                f"会議室の多数決で「{meeting_result['verdict']}」という結論になったため、今の内容のまま進めます。"
                f"(会議ログ: {meeting_result['log_path']})"
            )

    suggestions = _generate_suggestions(task, f"loop generated {file_path}")
    actions = [{"type": "write_file", "path": file_path, "content": full_code, "task": task}]
    if human_note:
        actions.append({"type": "answer", "content": human_note})
    return {
        "plan": [f"loop generation ({iterations} iterations): {task}"],
        "actions": actions,
        "suggestions": suggestions,
    }


def run_split_generation(task, generation, mem=None):
    try:
        return _run_split_generation_inner(task, generation, mem=mem)
    except OllamaUnavailableError as e:
        msg = f"Ollamaが起動していません。起動してから再送してください。\n詳細: {e}"
        print(f"[ERROR] run_split_generation: {msg}")
        import progress as _p
        _p.fail(generation, msg)
        return {"plan": [], "actions": [{"type": "answer", "content": msg}]}


def _run_split_generation_inner(task, generation, mem=None):
    result = _generate_one_component(task, generation, percent_start=10, percent_end=90, mem=mem)
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
        child_field = conn.get("child")
        relationship = conn.get("relationship", "")

        if not parent_name or not child_field:
            continue

        # ★2026-07-30 追加: モデルが1つのconnectionに複数の子コンポーネントを
        #   "child": ["A", "B"] のようなリストで返すことがある（実例:
        #   タブ切り替え画面生成時に発生）。本来は1コネクション1子を想定しており、
        #   正規化しないと `child_name not in generated_files` で
        #   TypeError: unhashable type: 'list' が発生し、複数ファイル生成全体が
        #   失敗していた。リストなら複数コネクションとして展開して処理する。
        child_names = child_field if isinstance(child_field, list) else [child_field]

        for child_name in child_names:
            if not isinstance(child_name, str) or not child_name:
                print(f"[CONNECT] childが文字列ではありません（{child_name!r}）。スキップします。")
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

                # ★2026-07-30 追加: 連携結果を無条件で採用せず検証する。
                #   壊れたコード（波括弧の閉じ忘れ・子コンポーネントの重複混入等）
                #   であれば採用せず、連携前の元のコードのまま残す（動くコードを
                #   壊れたコードで上書きしない方を優先する）。
                connect_problems = _validate_connected_code(new_parent_code, parent_name, child_name)
                if connect_problems:
                    print(f"[CONNECT] {parent_name} の連携後コードに問題があるため採用せず、元のコードのまま残します: {connect_problems}")
                    continue

                new_parent_code = fix_react_imports(new_parent_code)
                new_parent_code = fix_component_props(new_parent_code, new_parent_code, parent_name)

                generated_files[parent_name]["full_code"] = new_parent_code
                print(f"[CONNECT] {parent_name} に {child_name} を組み込みました。")
            except Exception as e:
                print(f"[CONNECT] {parent_name} と {child_name} の連携に失敗しました（元のコードのまま残します）: {e}")
                continue

    return generated_files


def run_multi_file_generation(task, generation, reference_files=None, mem=None):
    try:
        return _run_multi_file_generation_inner(task, generation, reference_files, mem=mem)
    except OllamaUnavailableError as e:
        msg = f"Ollamaが起動していません。起動してから再送してください。\n詳細: {e}"
        print(f"[ERROR] run_multi_file_generation: {msg}")
        import progress as _p
        _p.fail(generation, msg)
        return {"plan": [], "actions": [{"type": "answer", "content": msg}]}


def _run_multi_file_generation_inner(task, generation, reference_files=None, mem=None):
    progress.update(generation, 5, "(計画中) 必要なファイルを検討しています")
    plan_result = ask_ai(build_multi_file_plan_prompt(task, MAX_MULTI_FILES), model="codellama")

    files_plan = plan_result.get("files", [])
    if not files_plan:
        print("[MULTI] 計画が空だったため、単一ファイル生成にフォールバックします。")
        return run_split_generation(task, generation, mem=mem)

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
                mem=mem,
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
        # 【2026-07-30 追加】ユーザーには要約しか見せていないが、実際の例外内容は
        # chat_historyに残しておく。「なんで失敗したの？」と聞かれたときに
        # 実際のエラー（例: 'dict' object has no attribute 'strip'）を踏まえて
        # 答えられるようにするため。
        for f in failed_files:
            chat_history.add_error(task, f"{f['component_name']} の生成失敗: {f['error']}")

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


MAX_HTML_MOCKUP_LOOP_ITERATIONS = 3


def _looks_like_html(text: str) -> bool:
    if not text:
        return False
    head = text.strip()[:200].lower()
    return "<!doctype html" in head or "<html" in head


# ★2026-08-06追加: 指示文が複雑になると、codellamaが単一ファイルという
# ルールを無視してJS/HTMLを別ファイルとして分割し、さらに"content"の値を
# JSON文字列("...")ではなくJSのテンプレートリテラル(`...`)で書いてしまう
# ことがある（実例: clock.js / clock.htmlの2ファイルに分割し、バッククォート
# を使った壊れたJSON）。この形は_parse_response()のjson.loadsは通らず、
# ```コードフェンスも無いため既存のフォールバックでも拾えない。
# ここでは生テキストから "path"/"content" のペアを正規表現でゆるく拾い、
# .htmlの中身が参照している外部<script src>/<link href>を、同じ生テキスト
# 内に見つかった.js/.cssの中身でベストエフォートにインライン結合する。
_PATH_CONTENT_RE = re.compile(
    r'"path"\s*:\s*"([^"]+)"[\s\S]{0,80}?"content"\s*:\s*(`([\s\S]*?)`|"((?:[^"\\]|\\.)*)")'
)


def _extract_named_blocks(raw_text: str) -> dict:
    blocks = {}
    for m in _PATH_CONTENT_RE.finditer(raw_text):
        path = m.group(1)
        content = m.group(3) if m.group(3) is not None else m.group(4)
        if content is None:
            continue
        if m.group(4) is not None:
            content = content.replace("\\n", "\n").replace('\\"', '"')
        filename = path.split("/")[-1]
        blocks[filename] = content
    return blocks


def _inline_external_refs(html: str, blocks: dict) -> str:
    def _script_sub(m):
        src = m.group(1).split("/")[-1]
        js = blocks.get(src)
        return f"<script>{js}</script>" if js is not None else m.group(0)

    def _link_sub(m):
        href = m.group(1).split("/")[-1]
        css = blocks.get(href)
        return f"<style>{css}</style>" if css is not None else m.group(0)

    html = re.sub(r'<script\s+src=["\']([^"\']+)["\']\s*>\s*</script>', _script_sub, html)
    html = re.sub(r'<link[^>]+href=["\']([^"\']+)["\'][^>]*>', _link_sub, html)
    return html


def _extract_html_action(result: dict):
    """
    ask_ai()の戻り値からHTMLコードを取り出し、path/filenameをmockups/配下の
    .htmlに正規化した write_file 相当のdictにして返す。見つからなければNone。

    ★2026-08-06追加: 指示文が長く複雑になると、codellamaが指定のJSON形式
    （{"actions": [{"type": "write_file", ...}]}）で包むのを忘れ、生の
    HTMLコードだけを返すことがある（実例: 針の中心配置を細かく指定した
    指示文で発生）。この場合ask_ai内部のフォールバックにより
    {"code": "..."}（```コードフェンス経由）または
    {"actions": [{"type": "answer", "content": "..."}]}（フェンスも無い場合）
    という形で返ってくるが、従来はtype=="write_file"しか見ていなかったため
    せっかく生成できていたコードを毎回「生成に失敗しました」として捨てていた。
    ここでは、それらのフォールバック形にHTMLらしき内容が入っていれば
    write_fileアクションとして救済する。
    """
    for action in result.get("actions", []):
        if action.get("type") == "write_file" and action.get("content"):
            path = action.get("path") or "mockup.html"
            filename = path.split("/")[-1]
            if not filename.lower().endswith(".html"):
                filename = re.sub(r"\.\w+$", "", filename) + ".html"
            return {"type": "write_file", "path": f"mockups/{filename}", "content": action["content"]}

    # フォールバック1: {"code": "..."} 形式（```コードフェンス経由で救済されたもの）
    code = result.get("code")
    if isinstance(code, str) and _looks_like_html(code):
        print("[HTML_MOCKUP] write_fileアクションが無いため、codeフィールドから救済しました。")
        return {"type": "write_file", "path": "mockups/mockup.html", "content": code}

    # フォールバック2: type=="answer"のcontentに生のHTMLがそのまま入っているケース
    raw_text = None
    for action in result.get("actions", []):
        content = action.get("content")
        if action.get("type") == "answer" and isinstance(content, str):
            raw_text = content
            if _looks_like_html(content):
                print("[HTML_MOCKUP] write_fileアクションが無いため、answerのcontentから救済しました。")
                return {"type": "write_file", "path": "mockups/mockup.html", "content": content}

    # フォールバック3: JSON自体が壊れている(バッククォートでcontentを書く、
    # HTML/JSを複数ファイルに分割する、など)場合。生テキストから
    # "path"/"content"のペアを正規表現でゆるく拾い、.htmlが参照している
    # 外部<script src>/<link href>を同じテキスト内の.js/.cssでインライン化する。
    if raw_text:
        blocks = _extract_named_blocks(raw_text)
        html_filename = next((f for f in blocks if f.lower().endswith(".html")), None)
        if html_filename:
            html_content = _inline_external_refs(blocks[html_filename], blocks)
            print(f"[HTML_MOCKUP] 壊れたJSON内の{html_filename}を正規表現で救済し、外部参照をインライン化しました。")
            return {"type": "write_file", "path": "mockups/mockup.html", "content": html_content}

        # .html片すら見つからない場合の最後の手段: 生テキスト中に埋まっている
        # <!DOCTYPE html>...</html>ブロックをそのまま拾う（外部参照は残り得る）
        m = re.search(r'(<!DOCTYPE html[\s\S]*?</html>)', raw_text, re.IGNORECASE)
        if m:
            print("[HTML_MOCKUP] 生テキスト中の<html>ブロックを正規表現で救済しました（外部参照が残っている可能性あり）。")
            return {"type": "write_file", "path": "mockups/mockup.html", "content": m.group(1)}

    return None


def _build_html_mockup_fix_prompt(task: str, current_code: str, reason: str) -> str:
    return f"""
以下のタスクに基づいて、単体で動くHTML+CSS(必要なら<script>内JS)のUIモックアップを1ファイルで生成してください。

タスク：
{task}

前回生成したコード：
{current_code}

前回のコードには以下の問題が指摘されました。必ず修正した、完全なHTMLファイル全体を出力してください（差分ではなく全文）：
- {reason}

ルール：
- HTMLとCSSは1つのファイルにまとめること（<style>タグをheadに含める）。外部ファイルへの参照は禁止。
- タスクで要求された動き・計算（例：時計の針の角度計算、カウントの増減など、見た目や機能を
  実際に成立させるために必要なロジック）は<script>タグ内のJavaScriptとして実装してよい。
  サーバーとの通信は書かないこと。
- 日本語のラベル・文言を使うこと。

JSONのみ返してください。typeは必ず"write_file"を使うこと：
{{
  "plan": ["string"],
  "actions": [
    {{
      "type": "write_file",
      "path": "mockups/わかりやすい名前.html",
      "content": "修正後のHTMLコード全体"
    }}
  ]
}}
""".strip()


def _check_html_semantic_match(task: str, full_code: str) -> dict:
    """
    _check_semantic_match（React用）のHTMLモックアップ版。
    セクション分割を持たないため problem_section は返さず、reasonのみを
    次の修正プロンプトにそのまま渡す。
    """
    prompt = f"""
あなたはコードレビューの担当者です。以下のタスク指示と、実際に生成されたHTML(CSS/JS込み)の
UIモックアップのコードを比較し、指示された内容・挙動が本当にコードに実装されているかを
客観的に判定してください。

タスク指示：
{task}

生成されたコード：
{full_code}

判定基準：
- タスクで要求された見た目・要素（ボタン、表示内容など）が実際にコードに存在するか
- タスクが「動き」や「計算」を要求している場合（例：時計の針が動く、カウントが増える等）、
  それを実現するJavaScriptのロジックが実際に書かれているか判定すること。見た目のパーツ
  （例：時計盤や針の形のCSS）だけがあって、肝心の計算・更新処理が無い、または常に
  固定値のまま変化しない場合は不一致と判定する
- 些細な見た目の違い（色、文言など）は不一致としない

以下のJSONのみ返してください：
{{
  "matches": true または false,
  "reason": "判定理由を一言で"
}}
"""
    try:
        result = ask_ai(prompt, model=pick_router_student())
        return {
            "matches": bool(result.get("matches", True)),
            "reason": result.get("reason", ""),
        }
    except OllamaUnavailableError:
        print("[WARN] HTMLモックアップの意味チェック中にOllama接続エラー。チェックをスキップします。")
        return {"matches": True, "reason": "チェックスキップ（Ollama未起動）"}
    except Exception as e:
        print(f"[WARN] HTMLモックアップの意味チェックに失敗しました（無視して進めます）: {e}")
        return {"matches": True, "reason": "検証スキップ"}


def run_html_mockup_generation_with_harness(task, generation, max_iterations=MAX_HTML_MOCKUP_LOOP_ITERATIONS):
    """
    ★2026-08-06追加: run_html_mockup_generation（一発生成のみ）だと、
    見た目のパーツはあるが挙動が伴っていない出力（例：アナログ時計の針が
    固定角度のまま動かない）を誰も検証していなかった。ループ生成
    （_run_loop_generation_inner）と同じ「自己評価→ダメなら修正→それでも
    ダメなら会議室」パターンをHTMLモックアップにも適用する。
    tsc型チェックはHTMLに適用対象が無いため行わない（意味チェックのみ）。
    """
    try:
        return _run_html_mockup_generation_with_harness_inner(task, generation, max_iterations)
    except OllamaUnavailableError as e:
        msg = f"Ollamaが起動していません。起動してから再送してください。\n詳細: {e}"
        print(f"[ERROR] run_html_mockup_generation_with_harness: {msg}")
        progress.fail(generation, msg)
        return {"plan": [], "actions": [{"type": "answer", "content": msg}]}


def _run_html_mockup_generation_with_harness_inner(task, generation, max_iterations=MAX_HTML_MOCKUP_LOOP_ITERATIONS):
    iterations = min(max(max_iterations, 1), MAX_LOOP_ITERATIONS)
    print(f"[HTML_MOCKUP_LOOP] 開始 max_iterations={iterations}")

    # 既存ファイル流用チェック（run_html_mockup_generationと同じロジック）
    _m = re.search(r'[\w\-]+\.html', task)
    if _m:
        from run_split_generation_python import _check_existing_file
        import os as _os
        _base = _os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
        _existing_path = _os.path.join(_base, "mockups", _m.group(0))
        if _check_existing_file(task, _existing_path):
            print("[HTML_MOCKUP_LOOP] Existing file satisfies requirements. Skipping.")
            with open(_existing_path, 'r', encoding='utf-8-sig') as f:
                existing_code = f.read()
            return {
                "plan": [f"html mockup (existing): {task}"],
                "actions": [{"type": "write_file", "path": f"mockups/{_m.group(0)}", "content": existing_code}],
            }

    progress.update(generation, 10, "UIモックアップの初回生成中")
    # ★2026-08-06: 一時的にgranite-code:8bへの切り替えを試したが、"plan": [],
    # "actions": [] という空のJSONしか返さず(JSON形式のルール自体には従うが
    # 中身を生成できない)、qwen2.5-coder:3bより明らかに悪かったため元に戻した。
    # granite-codeはコード補完寄りで、この「日本語指示+JSON形式で返す」という
    # 指示スタイルとの相性が悪いと思われる。
    result = ask_ai(build_html_mockup_prompt(task), model="codellama")
    action = _extract_html_action(result)

    if not action:
        return {
            "plan": [f"html mockup generation: {task}"],
            "actions": [{"type": "answer", "content": "モックアップの生成に失敗しました。もう一度試してください。"}],
        }

    file_path = action["path"]
    full_code = action["content"]

    check = None
    semantic_match_failed = False

    for i in range(1, iterations):
        pct = 10 + int(70 * (i / iterations))
        progress.update(generation, pct, f"自己評価中（{i}/{iterations}回目）")

        check = _check_html_semantic_match(task, full_code)
        print(f"[HTML_MOCKUP_LOOP {i}] matches={check['matches']} reason={check['reason']}")

        if check["matches"]:
            print(f"[HTML_MOCKUP_LOOP] {i}回目でOK。終了します。")
            break

        progress.update(generation, pct + 2, f"修正中（{i}/{iterations}回目）")
        print(f"[HTML_MOCKUP_LOOP {i}] 修正します。理由: {check['reason']}")

        fix_result = ask_ai(_build_html_mockup_fix_prompt(task, full_code, check["reason"]), model="codellama")
        fix_action = _extract_html_action(fix_result)
        if not fix_action:
            print("[HTML_MOCKUP_LOOP] 修正コードの取得に失敗したためスキップします。")
            continue

        full_code = fix_action["content"]
        file_path = fix_action["path"]
    else:
        print(f"[HTML_MOCKUP_LOOP] {iterations}回すべて完了。")
        semantic_match_failed = True

    # 【追加・2026-08-06】コードを読むだけの意味チェック(_check_html_semantic_match)は
    # 「実際に描画したら要素がズレている」類のバグを原理的に検知できない
    # (実例: アナログ時計でmargin-topオフセットが抜けて針が中心からズレていたが、
    # コードレビューでは2回とも見抜けなかった)。ここで実際にヘッドレスブラウザで
    # 描画したスクリーンショットをvisionモデルに見せて、見た目レベルで確認する。
    # playwright未導入など環境要因で確認できない場合はchecked=Falseが返り、
    # 何もせずスキップする(今までの挙動を壊さない)。
    visual_match_failed = False
    visual_check = None
    progress.update(generation, 92, "見た目を描画して確認しています")
    from html_render_check import check_html_visual_match

    visual_check = check_html_visual_match(task, full_code)
    print(f"[HTML_MOCKUP_LOOP] visual checked={visual_check['checked']} matches={visual_check['matches']} reason={visual_check['reason']}")

    if visual_check["checked"] and not visual_check["matches"]:
        progress.update(generation, 93, "見た目のズレを修正しています")
        print(f"[HTML_MOCKUP_LOOP] 見た目のズレを検出、1回だけ修正を試みます。理由: {visual_check['reason']}")
        fix_result = ask_ai(_build_html_mockup_fix_prompt(task, full_code, visual_check["reason"]), model="codellama")
        fix_action = _extract_html_action(fix_result)
        if fix_action:
            full_code = fix_action["content"]
            file_path = fix_action["path"]
            visual_check = check_html_visual_match(task, full_code)
            print(f"[HTML_MOCKUP_LOOP] 修正後の再確認 checked={visual_check['checked']} matches={visual_check['matches']} reason={visual_check['reason']}")

        if visual_check["checked"] and not visual_check["matches"]:
            visual_match_failed = True

    human_note = None
    if semantic_match_failed or visual_match_failed:
        from meeting_vote import convene_meeting

        reason = check["reason"] if check else "詳細不明"
        problems = []
        if semantic_match_failed:
            problems.append(f"コードレビューによる自己評価が最後まで「要件と一致」と判定しませんでした（直近の理由: {reason}）")
        if visual_match_failed:
            problems.append(f"実際に描画したスクリーンショットの見た目確認でNG判定でした（理由: {visual_check['reason']}）")

        print(f"[HTML_MOCKUP_LOOP] 会議室を招集します。理由: {'; '.join(problems)}")
        meeting_question = (
            f"HTMLモックアップ生成タスク「{task}」について、{iterations}回の自己修正を試みましたが、"
            "以下の問題が残りました。このまま今の内容(ベストエフォート)で確定してよいか、"
            "それとも人に確認してもらうべきか判断してください。"
        )
        meeting_context = "\n".join(problems)
        meeting_result = convene_meeting(meeting_question, context=meeting_context, generation=generation)

        if meeting_result["needs_human"]:
            print(f"[HTML_MOCKUP_LOOP] 会議室でも結論が出ませんでした。人に確認を求めます。ログ: {meeting_result['log_path']}")
            human_note = (
                f"⚠️「{task}」の自動生成に問題が残りました(会議室(教室の生徒{meeting_result['total_votes']}体)でも"
                f"結論が出ませんでした)。今の内容はベストエフォートです。確認をお願いします。"
                f"(会議ログ: {meeting_result['log_path']})"
            )
        else:
            print(f"[HTML_MOCKUP_LOOP] 会議室の結論: {meeting_result['verdict']}（{meeting_result['vote_count']}/{meeting_result['total_votes']}票）")
            human_note = (
                f"ℹ️「{task}」の自動生成に問題が残りましたが、"
                f"会議室の多数決で「{meeting_result['verdict']}」という結論になったため、今の内容のまま進めます。"
                f"(会議ログ: {meeting_result['log_path']})"
            )

    progress.update(generation, 95, "ファイルを保存しています")
    actions = [{"type": "write_file", "path": file_path, "content": full_code, "task": task}]
    if human_note:
        actions.append({"type": "answer", "content": human_note})

    return {
        "plan": [f"html mockup loop generation ({iterations} iterations): {task}"],
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
        result = ask_ai(prompt, model=pick_router_student())
        suggestions = result.get("suggestions", [])
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions[:3]]
        return []
    except Exception as e:
        print(f"[SUGGESTIONS] 提案生成に失敗しました（無視して進めます）: {e}")
        return []


def run_web_search(user, generation, mem=None):
    """
    ★2026-08-02 追加: Tavily Web検索APIを使い、外部の最新情報が必要な質問に
    答える。route()から intent=="web_search" と判定された場合のみ呼ばれる。

    web_search.py の import・APIキー未設定・通信失敗のいずれも例外を投げず
    dictで状態を返す設計にしてあるため、ここでは例外処理よりも
    「何が起きたかをユーザーに分かる言葉で伝える」ことを優先する
    （dotenv未インストールでバックエンド全体が落ちた2026-08-02の事故を
    踏まえ、この機能自体がルーティング全体を巻き込んで落ちないようにする）。
    """
    progress.update(generation, 20, "Web検索を実行しています")
    try:
        from web_search import tavily_search
    except Exception as e:
        print(f"[WEB_SEARCH] web_search.pyの読み込みに失敗しました: {e}")
        return {
            "plan": ["Web検索"],
            "actions": [{"type": "answer", "content": "Web検索機能の読み込みに失敗しました。"}],
        }

    search_result = tavily_search(user)

    if not search_result["available"]:
        return {
            "plan": ["Web検索"],
            "actions": [{
                "type": "answer",
                "content": "Web検索用のAPIキー（Tavily）がまだ設定されていません。メニューの「持ち物」で設定状況を確認してください。",
            }],
        }

    if search_result["error"]:
        return {
            "plan": ["Web検索"],
            "actions": [{"type": "answer", "content": f"Web検索でエラーが発生しました: {search_result['error']}"}],
        }

    if not search_result["results"] and not search_result["answer"]:
        return {
            "plan": ["Web検索"],
            "actions": [{"type": "answer", "content": "検索しましたが、関連する情報が見つかりませんでした。"}],
        }

    progress.update(generation, 60, "検索結果をまとめています")

    sources_text = "\n".join(
        f"- {r['title']}: {r['content'][:300]}\n  出典: {r['url']}"
        for r in search_result["results"]
    )

    summarize_prompt = f"""
あなたはユーザーの質問にWeb検索結果を踏まえて日本語で答えるアシスタントです。

ユーザーの質問：
{user}

Web検索結果：
{sources_text}

ルール：
- 検索結果に基づいて、事実に忠実に、わかりやすい日本語で回答すること。
- 検索結果に無い情報を勝手に補わないこと。
- 回答の最後に、参照した情報源のURLを箇条書きで示すこと。

以下のJSONのみ返してください：
{{
  "answer": "回答本文（末尾に出典URLの箇条書きを含む）"
}}
"""
    final_answer = ""
    try:
        summary_result = ask_ai(summarize_prompt, model="swallow")
        final_answer = (summary_result.get("answer") or "").strip()
    except Exception as e:
        print(f"[WEB_SEARCH] 要約生成に失敗しました（検索結果をそのまま返します）: {e}")

    if not final_answer:
        # 要約生成に失敗した場合のフォールバック：Tavilyの簡易回答＋出典一覧をそのまま返す
        fallback_lines = []
        if search_result["answer"]:
            fallback_lines.append(search_result["answer"])
        fallback_lines.append("")
        fallback_lines.append("参照:")
        for r in search_result["results"]:
            fallback_lines.append(f"- {r['title']}: {r['url']}")
        final_answer = "\n".join(fallback_lines).strip()

    progress.update(generation, 90, "回答をまとめています")

    # ★2026-08-02 追加: 検索結果をweb_search_log（shobo_knowledge等とは別の
    # コレクション）に記録する。生成時のcontextには自動で差し込まれない
    # （＝ニュース的な情報を恒久知識として混入させない）、あくまで記録のみ。
    # 保存に失敗してもユーザーへの回答自体には影響させないよう、
    # 別スレッド・try/exceptで隔離する。
    try:
        from memory_store import save_web_search_log
        import threading

        def _save_log():
            try:
                save_web_search_log(user, final_answer, search_result["results"])
            except Exception as inner_e:
                print(f"[MEMORY] Web検索ログの保存に失敗しました（無視して続行）: {inner_e}")

        threading.Thread(target=_save_log, daemon=True).start()
    except Exception as e:
        print(f"[MEMORY] Web検索ログの保存準備に失敗しました（無視して続行）: {e}")

    return {
        "plan": [f"Web検索: {user}"],
        "actions": [{"type": "answer", "content": final_answer}],
    }


def _build_intent_prompt(user: str) -> str:
    """
    意図判定用プロンプトの構築のみを切り出したもの。
    _classify_intent() 本体と、model_comparison.py（router役候補モデルの
    比較評価）の両方から、本番と全く同じプロンプトを使うために共有する。
    """
    return f"""
以下のユーザー入力の意図を判定してください。

ユーザー入力：
{user}

選択肢：
- "shobo_app": 消防点検管理システム(shobo-app)のReact画面・コンポーネントを実際に
  作ってほしいという依頼。「shobo-app」という言葉や、Customers.tsx/Staff.tsx/
  Settings.tsxのような具体的なファイル名、または「StaffList」「CustomerDetail」の
  ようなPascalCase形式のコンポーネント名が明示されている場合はこれを選ぶこと。
  拡張子(.tsx)の有無は判定に関係ない。「〜という名前で」「〜というコンポーネントを」
  のように、作るべきコンポーネントの名前が具体的に指定されていれば、拡張子が
  無くても shobo_app を選ぶこと。
- "html_mockup": shobo-app以外の、何らかのソフトウェアのUI画面・モックアップや、
  画面の一部として使えるような単体のUI部品を作ってほしいという依頼（例：
  「会員登録画面を作って」「お問い合わせフォームを作って」「ログイン画面のHTMLを
  作って」に加えて、「時計を作って」「アナログの時計を作って」「カウンターを作って」
  「タイマーを作って」のような、フォーム・画面という言葉を使っていなくても、
  見た目と動きを持つ画面部品(HTML+CSS+JSで表現できるもの)を作ってほしい依頼も
  含む）。
- "python_module": AI_001自身のバックエンド機能・ロジックをPythonコードとして
  実装してほしいという依頼（例：「自己改善ループを実装して」「新しいルーティング
  ロジックを追加して」「〜を判定する関数を作って」のような、画面ではなく
  サーバー側のPythonロジックの追加・変更を求める依頼）。
- "web_search": 検索エンジンでの調査、最新のニュース・時事情報、リアルタイムの
  状況など、AI_001の内部知識だけでは答えられない外部の実世界の情報を求めている
  依頼（例：「ガス爆発について検索して」「最新の火災ニュースを教えて」「〜について
  調べて」「今日の〜のニュースは？」）。文中に「検索して」「ググって」「調べて」の
  ような明確な検索指示があり、かつ対象が外部の実世界の出来事・情報である場合は
  これを選ぶこと。
- "other": 上記のどちらにも当たらない依頼（通常の会話、質問、記憶、UI以外の
  簡単なコード生成など、まとまったPythonモジュールとしての実装が不要なもの）。

判定基準：
- 「検索して」「ググって」「調べて」という言葉があっても、対象がAI_001自身の
  データ（ChromaDB件数、記憶した内容、これまでの生成物、shobo-appのコードなど）
  であれば "other" を選ぶこと。対象が外部の実世界の情報（ニュース、時事、
  一般知識、最新の出来事）であれば "web_search" を選ぶこと。
- 文章中に偶然それらしい単語（画面、コンポーネント、ファイル名など）が含まれて
  いるだけで、実際の作業依頼でない場合（例：資料の要約や引用）は "other" を選ぶこと。
- 「作って」「生成して」「実装して」のような明確な作業依頼があるかを重視すること。
- 依頼がReactの画面・UIに関するものなら shobo_app か html_mockup を優先して選ぶこと。
- 依頼がAI_001自体の振る舞い・処理ロジックに関するものなら python_module を選ぶこと。
- 「確認して」「教えて」「まとめて」「どうなってる？」「一覧を見せて」「調べて」のように、
  既存のデータ・状態・内容を調べたい／説明してほしいだけの依頼は、たとえ
  "プログラム"や"データ"のように他の選択肢を連想させる単語が含まれていても
  必ず "other" を選ぶこと。これらは新規のファイル・コンポーネントを作る依頼ではない。
- 「Staff」「Program」のような単語が含まれていても、それが実在の物件・画面を
  指しているとは限らない。依頼文の中に、作ってほしい画面やコンポーネントの
  内容が具体的に書かれていない場合は、勝手に画面名を創作せず "other" を選ぶこと。
- ChromaDB・DB・データベース・件数・集計のように、システム内部のデータ状況を
  問い合わせる依頼は、ファイル生成の依頼ではないため必ず "other" を選ぶこと。
- ★重要：文末が疑問形（「〜ですか？」「〜ありますか？」「〜何ですか？」「〜教えて
  ください」等）の場合、たとえ「消防設備」「点検」「空調」「排煙」のような
  shobo_app/html_mockupを連想させるドメイン単語が含まれていても、それは
  「知識を尋ねる質問」であって「画面を作ってほしい依頼」ではないため、
  必ず "other" を選ぶこと。
  例：「消防設備点検の実務知識として質問です。台風や豪雨などの気象災害時、
  点検で特に注意すべき点は何ですか？」→ "shobo_app"や"html_mockup"ではなく
  "other"（画面名やコンポーネント名の指定がなく、単に知識を尋ねているだけのため）。
  例：「空調・排煙設備が火災時に正常に作動しない場合、どのような問題が
  発生しますか？」→ 同様に "other"。
  「〜の画面を作って」「〜というコンポーネントを作って」のように、明確に
  生成物（画面・ファイル）を要求する動詞（作って／生成して）が伴わない限り、
  疑問文は常に "other" と判定すること。

以下のJSONのみ返してください：
{{
  "intent": "shobo_app" または "html_mockup" または "python_module" または "web_search" または "other",
  "reason": "判定理由を一言で"
}}
"""


def _build_routing_prompt(user: str) -> str:
    """
    モデル振り分け用プロンプトの構築のみを切り出したもの。
    _route_with_ollama() 本体と、model_comparison.py の両方から
    本番と全く同じプロンプトを使うために共有する。
    """
    return f"""
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


def _classify_intent(user: str) -> dict:
    """
    ★2026-07-26 変更: 意図判定はpick_router_student()によるランダム選択をやめ、
    固定で"router"(qwen2.5:3b)を使う。ROUTER_STUDENT_POOL内の精度の低いモデル
    （llama3_2/granite、正解率43.8%）がランダムに選ばれることで、消防設備の
    実務Q&Aが高頻度でshobo_app/html_mockupに誤判定される事故が実際に発生したため。
    他の用途（_check_semantic_match, _generate_suggestions, 振り分け判定）は
    引き続きpick_router_student()を使い、複数モデルの比較・評価目的を維持する。
    """
    intent_prompt = _build_intent_prompt(user)
    try:
        result = ask_ai(intent_prompt, model="router")
        intent = result.get("intent", "other")
        if intent not in ("shobo_app", "html_mockup", "python_module", "web_search", "other"):
            intent = "other"
        return {"intent": intent, "reason": result.get("reason", "")}
    except Exception as e:
        print(f"[WARN] 意図判定に失敗しました（otherとして続行します）: {e}")
        return {"intent": "other", "reason": "判定失敗"}


def _needs_clarification(user: str) -> dict:
    """
    ★2026-08-02 追加: 「明確化チェック」チェックボックスがONのときだけ、
    shobo_app/html_mockup生成に入る前に呼ばれる。判定に失敗した場合は
    生成を止めないよう needs_clarification=False（=そのまま生成へ進む）
    にフォールバックする。
    """
    try:
        result = ask_ai(build_clarification_check_prompt(user), model="router")
        return {
            "needs_clarification": bool(result.get("needs_clarification", False)),
            "question": result.get("question", ""),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        print(f"[WARN] 明確化チェックに失敗しました（スキップして続行します）: {e}")
        return {"needs_clarification": False, "question": "", "reason": "判定失敗"}


def route(user, mem, generation, reference_files=None, image_base64=None, mode="normal", max_iterations=1, clarify=False):

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

    if mode == "loop":
        print(f"[DEBUG] → run_loop_generation (max_iterations={max_iterations})")
        return run_loop_generation(user, generation, max_iterations=max_iterations, reference_files=reference_files, mem=mem)

    try:
      return _route_with_ollama(user, mem, generation, reference_files, mode, max_iterations, clarify)
    except OllamaUnavailableError as e:
        msg = f"Ollamaが起動していません。起動してから再送してください。\n詳細: {e}"
        print(f"[ERROR] route(): {msg}")
        progress.fail(generation, msg)
        return {"plan": [], "actions": [{"type": "answer", "content": msg}]}


def _route_with_ollama(user, mem, generation, reference_files, mode, max_iterations, clarify=False):
    progress.update(generation, 10, "依頼内容を判定しています")
    intent_check = _classify_intent(user)
    intent = intent_check["intent"]
    print(f"[INTENT] → {intent} ({intent_check['reason']})")

    if intent != "python_module" and _looks_like_python_module_request(user):
        print(f"[INTENT] キーワード判定により '{intent}' から 'python_module' に上書きします")
        intent = "python_module"

    # ★2026-08-02 追加: 「明確化チェック」チェックボックス。shobo_app/html_mockup
    # （画面・コンポーネント生成）に入る直前だけ動く。他のintent（雑談・記憶・
    # Web検索など）には適用しない＝チェックが邪魔にならないようにする。
    if clarify and intent in ("shobo_app", "html_mockup"):
        progress.update(generation, 12, "仕様が十分か確認しています")
        clarify_check = _needs_clarification(user)
        if clarify_check["needs_clarification"] and clarify_check["question"]:
            print(f"[CLARIFY] question={clarify_check['question']} reason={clarify_check['reason']}")
            return {
                "plan": ["明確化質問"],
                "actions": [{"type": "ask_clarification", "content": clarify_check["question"]}],
            }

    if intent == "shobo_app":
        print("[DEBUG] → run_multi_file_generation")
        return run_multi_file_generation(user, generation, reference_files=reference_files, mem=mem)

    if intent == "html_mockup":
        print("[DEBUG] → run_html_mockup_generation_with_harness")
        return run_html_mockup_generation_with_harness(user, generation)

    if intent == "web_search":
        print("[DEBUG] → run_web_search")
        return run_web_search(user, generation, mem=mem)

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
    routing_prompt = _build_routing_prompt(user)

    routing = ask_ai(routing_prompt, model=pick_router_student())
    selected_model = routing.get("model", "swallow")
    print(f"\n[ROUTER] → {selected_model} ({routing.get('reason', '')})")

    progress.update(generation, 50, f"{selected_model} で生成しています")
    # 【2026/07/22 追加】以前は ask_ai_with_rag が実装されているにもかかわらず
    # route() のどこからも呼ばれておらず、shobo_knowledge に蓄積した消防法令・
    # 建材・電気設備等の知識が通常の質問応答に一切反映されていなかった。
    # ここで user の入力自体を検索クエリとして渡し、関連知識があれば
    # プロンプトに注入するようにする。
    # 【2026-07-30 追加】直近のやりとり（chat_history）をプロンプトに差し込み、
    # 「なんで失敗したの？」のような直前の文脈依存の質問にも答えられるようにする。
    history = chat_history.get_recent_context()
    result = ask_ai_with_rag(build_prompt(user, mem, history=history), model=selected_model, knowledge_query=user)

    # 2段階ハルシネーション対策
    # Stage 1: LAが自己申告（consult_village）した場合
    # Stage 2: LAが答えを出しても確信度が低ければ村5人へ回す
    actions = result.get("actions") or []
    if actions:
        first = actions[0]
        if first.get("type") == "consult_village":
            question = first.get("question") or user
            print(f"[VILLAGE_CONSULT] LA自己申告→5人相談: {question[:60]}")
            return _consult_5_members(question, generation, la_answer="")
        if first.get("type") == "answer":
            answer_text = first.get("content") or ""
            if not _check_confidence(user, answer_text):
                print(f"[VILLAGE_CONSULT] 確信度低→5人相談")
                return _consult_5_members(user, generation, la_answer=answer_text)

    return result


def _check_confidence(question: str, answer: str) -> bool:
    """LAの回答に根拠があるか小モデルで採点する。Trueなら自信あり、Falseなら村へ回す。"""
    prompt = f"""以下の質問と回答を評価してください。

質問: {question}
回答: {answer[:300]}

この回答は事実に基づいた確かな情報ですか？それとも推測・創作が含まれていますか？

以下のJSONのみ返してください：
{{"confident": true または false, "reason": "一言で"}}
"""
    try:
        result = ask_ai(prompt, model="router")
        confident = result.get("confident", True)
        reason = result.get("reason", "")
        print(f"[CONFIDENCE] confident={confident} reason={reason}")
        return bool(confident)
    except Exception as e:
        print(f"[CONFIDENCE] 採点失敗（通す）: {e}")
        return True


def _consult_5_members(question: str, generation: int, la_answer: str = "") -> dict:
    """
    ランダムに選んだ5人の村メンバーに直接質問し、多数決で採用する回答を返す。
    village_chat の討論状態（_state）には一切触れない。
    """
    import os
    import random
    import json as _json

    progress.update(generation, 35, "仲間5人を選んでいます")

    # コテハンを読み込む（表示用。失敗しても続行）
    kote_file = os.path.join(os.path.dirname(__file__), "village_kote.json")
    try:
        with open(kote_file, encoding="utf-8") as f:
            kote_map = _json.load(f)
    except Exception:
        kote_map = {}

    # PARTICIPANTS をここで直接定義しても重複するので village_chat から取得
    try:
        import village_chat as vc
        candidates = list(vc.PARTICIPANTS)
    except Exception:
        # フォールバック：代表的な5モデルをハードコード
        candidates = [
            {"role": "swallow",    "persona": "実装担当"},
            {"role": "llama3_1",   "persona": "分析担当"},
            {"role": "qwen3",      "persona": "哲学担当"},
            {"role": "qwen_coder_14b", "persona": "技術担当"},
            {"role": "lfm_jp",     "persona": "まとめ役"},
        ]

    random.shuffle(candidates)
    selected = candidates[:5]

    answers = []
    for i, p in enumerate(selected):
        progress.update(generation, 40 + int(35 * i / 5), f"仲間に聞いています（{i+1}/5）")
        kote = kote_map.get(p["role"], p["persona"])
        prompt = f"""質問：{question}

この質問に率直に答えてください。
根拠のある答えのみ述べること。わからなければ「わかりません」と答えてください。

以下のJSONのみ返してください：
{{"answer": "回答"}}
"""
        try:
            res = ask_ai(prompt, model=p["role"])
            text = (res.get("answer") or res.get("message") or "").strip()
            if not text:
                acts = res.get("actions") or []
                if acts:
                    text = (acts[0].get("content") or acts[0].get("answer") or "").strip()
            if text:
                answers.append({"kote": kote, "persona": p["persona"], "answer": text})
                print(f"[VILLAGE_CONSULT] {kote}: {text[:40]}")
        except Exception as e:
            print(f"[VILLAGE_CONSULT] {p['role']} 失敗: {e}")

    if not answers:
        return {
            "plan": [],
            "actions": [{"type": "answer", "content": "わかりません。仲間への相談も失敗しました。"}],
        }

    progress.update(generation, 80, "仲間の意見を集計しています")

    answers_text = "\n".join(f"- {a['kote']}: {a['answer']}" for a in answers)
    vote_prompt = f"""以下は同じ質問に対する{len(answers)}人の回答です。

質問: {question}

回答一覧:
{answers_text}

最も多くの人が支持している回答を選び、その内容を自然な日本語で1〜2文にまとめてください。
「わかりません」が多数なら「わかりません」と返してください。

以下のJSONのみ返してください：
{{"majority_answer": "まとめた回答", "agreement": "同意人数（例：3/5）"}}
"""
    majority = ""
    agreement = ""
    try:
        vote_res = ask_ai(vote_prompt, model="swallow")
        majority = (vote_res.get("majority_answer") or "").strip()
        agreement = vote_res.get("agreement", "")
    except Exception as e:
        print(f"[VILLAGE_CONSULT] 多数決集計失敗: {e}")

    if not majority:
        majority = answers[0]["answer"]

    member_lines = "\n".join(f"・{a['kote']}: {a['answer'][:80]}" for a in answers)
    header = f"【仲間5人に相談しました（{agreement}）】" if agreement else "【仲間5人に相談しました】"
    content = f"{header}\n\n{majority}\n\n---\n{member_lines}"

    # wikiノートをバックグラウンドで記録
    try:
        import threading
        import wiki_maintainer
        threading.Thread(
            target=wiki_maintainer.analyze_and_write,
            args=(question, la_answer, "village_consulted", majority),
            daemon=True,
        ).start()
    except Exception as e:
        print(f"[WIKI_MAINTAINER] 起動失敗（無視）: {e}")

    return {
        "plan": ["村相談（5人多数決）"],
        "actions": [{"type": "answer", "content": content}],
    }