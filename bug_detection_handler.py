"""
バグ検出ハンドラー（修正版）

前回LAが生成したバージョンには以下の致命的な問題があった：
1. handle_submit_correctness() 内に TypeScript の型注釈
   (React.FormEvent<HTMLFormElement>) がそのまま書かれており、
   Pythonとして SyntaxError になる
2. `import requests` が2回重複していた
3. `check_imports()` が importlib を使っているのに import していなかった
4. check_component_name_consistency() が Step1 コンポーネントに対して
   "Step2" を期待するという、そもそも成立しないロジックだった
5. check_form_usage() は日本語コメント文字列を丸ごと比較しているだけで、
   実際のコード内容を何も検証していなかった
6. 使用例で run_bug_detection_handler(None) を呼んでおり、
   冒頭のNoneチェックで必ず失敗して終わる

このファイルでは、上記すべてを解消し、実際に
「生成された.tsxのコード文字列」を受け取って
静的にパターンチェックする、動く実装にしている。
TypeScriptの型は文字列（正規表現の照合対象）としてのみ扱い、
Python構文としては一切書かない。
"""

import importlib
import re
import requests


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


def check_imports(code: str, required_imports: list) -> list:
    """
    生成されたコード文字列の中に、必要なimport（例: "Step2"）が
    実際に import 文として存在するかをチェックする。
    importlib は使わず、正規表現でコード文字列そのものを検査する
    （こちらはPythonモジュールの検査ではなく、TSXコード文字列の検査のため）。
    """
    missing = []
    for name in required_imports:
        pattern = rf"import\s+{re.escape(name)}\s+from"
        if not re.search(pattern, code):
            missing.append(name)
    return missing


def check_component_name_consistency(component_name: str, expected_name: str) -> bool:
    """
    コンポーネント名が、計画時に指定された名前と一致しているかを確認する。
    例: build_split_task_prompt で component_name="Step1" と決めたなら、
    実際に export default function Step1() になっているべき。
    """
    return component_name == expected_name


def check_form_usage(code: str, has_form_event_handler: bool) -> list:
    """
    FormEvent型を受け取るハンドラー（handleSubmitなど）が定義されているのに、
    jsx側で <form onSubmit=...> が使われていない場合を検出する。

    TypeScriptの型は文字列パターンとしてのみ扱う（実行はしない）。
    """
    problems = []

    uses_form_event_type = bool(
        re.search(r"React\.FormEvent<HTMLFormElement>", code)
    )
    has_form_tag = bool(re.search(r"<form\b", code))

    if (has_form_event_handler or uses_form_event_type) and not has_form_tag:
        problems.append(
            "FormEvent型のハンドラーが定義されているのに<form onSubmit=...>が"
            "使われていません。<div>+onClickでsubmitを模倣しないでください。"
        )

    return problems


def check_event_scope(code: str) -> list:
    """
    onClick={() => handleXxx(event)} のように、コールバック関数の
    パラメータに event が定義されていないのに event を参照している
    パターンを検出する。
    """
    problems = []
    if re.search(r"onClick=\{\(\)\s*=>\s*[\w.]+\(event\)\}", code):
        problems.append(
            "onClickのコールバックがパラメータなしの () => ... なのに "
            "event を参照しています。event未定義のReferenceErrorになります。"
        )
    return problems


def run_bug_detection_handler(code: str, component_name: str, expected_name: str,
                               required_imports: list = None,
                               has_form_event_handler: bool = False) -> dict:
    """
    生成された.tsxのコード文字列に対して、一連の静的チェックを行う。
    例外を投げず、問題点のリストを返す（他の_validate_section系関数と
    インターフェースを揃えるため）。
    """
    if code is None:
        return {"ok": False, "problems": ["code が None です。生成結果を渡してください。"]}

    problems = []
    required_imports = required_imports or []

    missing_imports = check_imports(code, required_imports)
    if missing_imports:
        problems.append(f"必要なimportが不足しています: {', '.join(missing_imports)}")

    if not check_component_name_consistency(component_name, expected_name):
        problems.append(
            f"コンポーネント名が一致しません: 期待={expected_name}, 実際={component_name}"
        )

    problems.extend(check_form_usage(code, has_form_event_handler))
    problems.extend(check_event_scope(code))

    return {"ok": len(problems) == 0, "problems": problems}


def log_start_processing():
    print("[STARTING PROCESSING]")


def log_end_processing(success: bool):
    print("[PROCESSING COMPLETE]" if success else "[PROCESSING FAILED]")


def log_debug_info(message: str):
    print(f"[DEBUG INFO]: {message}")


if __name__ == "__main__":
    # 使用例：Step1.tsx を想定したチェック
    sample_code = """
    import React, { useState } from 'react';
    import Step2 from './Step2';

    export default function Step1(props) {
      const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
      };
      return (
        <div>
          <button onClick={() => handleSubmit(event)}>回答する</button>
        </div>
      );
    }
    """

    log_start_processing()
    result = run_bug_detection_handler(
        code=sample_code,
        component_name="Step1",
        expected_name="Step1",
        required_imports=["Step2"],
        has_form_event_handler=True,
    )
    if result["ok"]:
        log_end_processing(True)
    else:
        log_end_processing(False)
        for p in result["problems"]:
            log_debug_info(p)