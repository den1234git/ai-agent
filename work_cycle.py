"""
AI_001の「仕事」機能（旧learning.pyの後継）。

learning.py が「外部から知識を検索して保存するだけ」だったのに対し、
work_cycle.py は shobo-app 自身のコードに対して実際に手を動かすタスクを行う。

4サイクルをローテーションする：
  1. scan_and_fix   : 既存コードをスキャンし、問題点と修正案をログに書く（自動適用はしない）
  2. write_tests    : テストが無いコンポーネントにテストコードを書く
  3. write_docs     : 生成済みコンポーネントごとに個別ドキュメント(.md)を書く
  4. self_improve   : 過去の失敗パターンから、AI_001自身の改善提案をログに書く

「仕事させる」ボタンから、3時間/6時間/9時間/∞（手動停止まで）を選んで起動する。
時間になったら自動的に止まる。∞の場合は progress.abort() が押されるまで続く。

使うモデルは差し替え可能にしてある（model_name引数）。
将来的に同じサイクルを複数モデルに投げて比較する拡張を見込んでいる。

【2026/07/13 修正】
write_tests / write_docs で生成される内容が、実際のコンポーネント機能と無関係な
「カウンター（増やすボタン、Count: N）」パターンを繰り返す問題が判明した。
原因は、build_prompt.py の build_section_prompt に実装されている anti_copy_rule
（出力例をそのままコピーすることを禁止する指示）が、build_section_prompt を経由しない
work_cycle.py 独自のプロンプト組み立てには適用されていなかったため。
今回、_run_write_tests_cycle / _run_write_docs_cycle のプロンプトに同趣旨の
「絶対厳守」ブロックを直接追加し、対症療法として対応した（build_section_prompt経由への
全面的な作り直しは影響範囲が大きいため見送り、まずは軽量な修正で様子を見る）。
"""

import os
import re
import time
import json
from datetime import datetime

from ai import ask_ai, OllamaUnavailableError
import progress as _p

WORK_LOGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "work_logs"
)

# shobo-appのソースディレクトリ（環境に合わせて変更可）
SHOBO_APP_DIR = os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
SHOBO_SRC_DIR = os.path.join(SHOBO_APP_DIR, "src")

# 個別ドキュメントの出力先
DOCS_OUTPUT_DIR = os.path.join(SHOBO_APP_DIR, "docs", "components")

WORK_CYCLE_TYPES = ["scan_and_fix", "write_tests", "write_docs", "self_improve"]

DEFAULT_MODEL = "codellama"  # = qwen2.5-coder:3b（models.jsonの設定に従う）

# self_improve が改善提案の対象にできるファイル（AI_001自身のコード）
SELF_IMPROVE_TARGET_FILES = [
    "router.py",
    "build_prompt.py",
    "bug_detection_handler.py",
    "ai.py",
]


# ─────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────

def _list_source_files(extensions=(".tsx", ".ts")) -> list:
    """
    shobo-app/src配下のソースファイル一覧を取得する。
    node_modules等は除外。取得に失敗しても空リストを返して処理は継続する。
    """
    files = []
    try:
        for root, dirs, filenames in os.walk(SHOBO_SRC_DIR):
            dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__tests__")]
            for fname in filenames:
                if fname.endswith(extensions):
                    files.append(os.path.join(root, fname))
    except Exception as e:
        print(f"[WORK] ソースファイル一覧の取得に失敗しました: {e}")
    return files


def _pick_target_file(files: list, recently_scanned: list) -> str | None:
    """
    直近スキャンしたファイルを避けつつ、次に処理するファイルを1つ選ぶ。
    全部やると重いので毎サイクル1ファイルだけ処理する。
    """
    if not files:
        return None
    candidates = [f for f in files if f not in recently_scanned]
    if not candidates:
        candidates = files  # 一周したらリセット
    return candidates[0]


def _write_work_log(cycle_type: str, title: str, body: str) -> str:
    """
    1サイクル分の作業ログをテキストファイルに書き出す。
    scan_and_fix と self_improve はここに「提案」を書くだけで、
    コードの自動書き換えは一切行わない。
    """
    os.makedirs(WORK_LOGS_DIR, exist_ok=True)
    now = datetime.now()
    filename = f"{cycle_type}_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(WORK_LOGS_DIR, filename)

    lines = [
        f"=== AI_001 作業ログ（{cycle_type}） ===",
        f"日時: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"件名: {title}",
        "",
        body,
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[WORK] 作業ログを書き出しました: {filepath}")
    return filepath


# ─────────────────────────────────────────
# サイクル1: scan_and_fix（提案のみ、自動適用しない）
# ─────────────────────────────────────────

_recently_scanned_files = []  # プロセス内メモリで直近スキャン履歴を保持（簡易実装）


def _run_scan_and_fix_cycle(generation, model_name: str = DEFAULT_MODEL) -> dict:
    global _recently_scanned_files

    print(f"[DEBUG] recently_scanned_files={_recently_scanned_files}")

    _p.update(generation, 10, "スキャン対象ファイルを選んでいます")
    files = _list_source_files()
    print(f"[DEBUG] files count={len(files)}")
    target = _pick_target_file(files, _recently_scanned_files)
    print(f"[DEBUG] target selected={target}")

    if target is None:
        log_path = _write_work_log("scan_and_fix", "対象ファイルなし", "src配下にスキャン対象のファイルが見つかりませんでした。")
        return {"status": "skipped", "log_path": log_path}

    _recently_scanned_files.append(target)
    if len(_recently_scanned_files) > 20:
        _recently_scanned_files.pop(0)

    try:
        with open(target, "r", encoding="utf-8-sig") as f:
            code = f.read()
    except Exception as e:
        log_path = _write_work_log("scan_and_fix", f"読み込み失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    _p.update(generation, 40, f"{os.path.basename(target)} を分析しています")

    prompt = f"""
以下のTypeScript/Reactのコードをレビューしてください。

ファイルパス: {target}

コード:
{code[:4000]}

以下の観点でチェックし、問題があれば具体的な修正案を提示してください：
- 未使用のimport
- 型の不整合や any の乱用
- 命名の不統一
- 重複コード
- Reactのアンチパターン（不要な再レンダリング要因など）

問題が無ければ「問題なし」とだけ書いてください。
コードは書き換えず、レビューコメントのみを日本語で出力してください。
"""

    _p.update(generation, 70, "レビュー結果をまとめています")
    try:
        result = ask_ai(prompt, model=model_name, json_mode=False)
        review_text = ""
        if isinstance(result, dict):
            review_text = result.get("code") or result.get("content") or str(result)
        else:
            review_text = str(result)
    except OllamaUnavailableError as e:
        log_path = _write_work_log("scan_and_fix", f"Ollama未起動: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}
    except Exception as e:
        log_path = _write_work_log("scan_and_fix", f"レビュー失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    log_path = _write_work_log("scan_and_fix", target, review_text)
    return {"status": "done", "target": target, "log_path": log_path}


# ─────────────────────────────────────────
# サイクル2: write_tests（テストが無いコンポーネントに書く）
# ─────────────────────────────────────────

def _has_existing_test(component_path: str) -> bool:
    """
    同名の .test.tsx / .spec.tsx が __tests__ 配下、または同階層に無いか確認する。
    """
    base = os.path.splitext(os.path.basename(component_path))[0]
    dirname = os.path.dirname(component_path)
    candidates = [
        os.path.join(dirname, "__tests__", f"{base}.test.tsx"),
        os.path.join(dirname, f"{base}.test.tsx"),
        os.path.join(dirname, f"{base}.spec.tsx"),
    ]
    return any(os.path.exists(c) for c in candidates)


def _run_write_tests_cycle(generation, model_name: str = DEFAULT_MODEL) -> dict:
    _p.update(generation, 10, "テスト未整備のコンポーネントを探しています")
    print(f"[DEBUG] SHOBO_SRC_DIR={SHOBO_SRC_DIR}")
    files = _list_source_files(extensions=(".tsx",))
    print(f"[DEBUG] found {len(files)} tsx files: {files[:5]}")
    # pages配下のコンポーネントを優先対象にする
    component_files = [f for f in files if "pages" in f or "components" in f]
    print(f"[DEBUG] component_files={component_files}")

    target = None
    for f in component_files:
        exists = _has_existing_test(f)
        print(f"[DEBUG]   {f} -> has_existing_test={exists}")
        if not exists:
            target = f
            break

    if target is None:
        log_path = _write_work_log("write_tests", "対象なし", "テスト未整備のコンポーネントは見つかりませんでした（全て整備済み）。")
        return {"status": "skipped", "log_path": log_path}

    try:
        with open(target, "r", encoding="utf-8-sig") as f:
            code = f.read()
    except Exception as e:
        log_path = _write_work_log("write_tests", f"読み込み失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    _p.update(generation, 40, f"{os.path.basename(target)} のテストを書いています")

    component_name = os.path.splitext(os.path.basename(target))[0]
    prompt = f"""
以下のReactコンポーネントに対するテストコードを書いてください。

ファイル: {target}
コンポーネント名: {component_name}

コード:
{code[:4000]}

要件：
- React Testing Library + Jest を使うこと
- 主要な表示内容とユーザー操作（ボタンクリックなど）を最低限テストすること
- import文には正しい相対パスを使うこと（同階層に配置される想定: ./{component_name}）
- コードのみを出力し、説明文は不要

絶対厳守（重要）：
- 上のコード内に実際に存在する要素・関数名・state・props・ラベル文言のみをテスト対象にすること
- 「カウンター」「増やすボタン」「Count: N」のような、上のコードに存在しない
  一般的なサンプルパターンを流用することは固く禁止する
- 必ず {component_name} の実際の中身（実在するonClickハンドラ名、実際に画面に
  表示される文言、実際のuseState変数名など）を読み取った上で、それに基づいて
  テストを書くこと。コードを見ずに書いたような当てずっぽうのテストは不可

出力形式について（絶対厳守）：
- 出力は import 文から始まる、実際にJestで実行できるTypeScript/TSXのソースコード
  そのものであること。JSON形式（{{ "test": {{...}} }} のようなテスト計画・
  テスト項目の一覧オブジェクト）で出力することは固く禁止する
- 必ず describe(...) と test(...) または it(...) を使った、実行可能な
  実装コードを書くこと。テスト内容を文章や配列で列挙するだけの出力は不可
- 出力例（この形式に必ず従うこと。中身は{component_name}の実際の内容に置き換える）：
  import {{ render, screen, fireEvent }} from '@testing-library/react';
  import {component_name} from '../{component_name}';

  describe('{component_name}', () => {{
    test('画面が表示される', () => {{
      render(<{component_name} />);
      // ここに実際の検証コード
    }});
  }});
"""

    _p.update(generation, 70, "テストコードを生成しています")
    try:
        result = ask_ai(prompt, model=model_name, json_mode=False)
        test_code = ""
        if isinstance(result, dict):
            test_code = result.get("code") or result.get("content") or ""
        if not test_code:
            test_code = str(result)
    except OllamaUnavailableError as e:
        log_path = _write_work_log("write_tests", f"Ollama未起動: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}
    except Exception as e:
        log_path = _write_work_log("write_tests", f"生成失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    # __tests__ ディレクトリに保存
    test_dir = os.path.join(os.path.dirname(target), "__tests__")
    try:
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, f"{component_name}.test.tsx")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        print(f"[WORK] テストファイルを保存しました: {test_path}")
    except Exception as e:
        log_path = _write_work_log("write_tests", f"保存失敗: {target}", f"生成はできたが保存に失敗: {e}\n\n---生成内容---\n{test_code}")
        return {"status": "error", "log_path": log_path}

    log_path = _write_work_log("write_tests", target, f"テストファイルを作成しました: {test_path}")
    return {"status": "done", "target": target, "test_path": test_path, "log_path": log_path}


# ─────────────────────────────────────────
# サイクル3: write_docs（個別ドキュメント生成）
# ─────────────────────────────────────────

def _has_existing_doc(component_name: str) -> bool:
    doc_path = os.path.join(DOCS_OUTPUT_DIR, f"{component_name}.md")
    return os.path.exists(doc_path)


def _run_write_docs_cycle(generation, model_name: str = DEFAULT_MODEL) -> dict:
    _p.update(generation, 10, "ドキュメント未整備のコンポーネントを探しています")
    files = _list_source_files(extensions=(".tsx",))
    component_files = [f for f in files if "pages" in f or "components" in f]

    target = None
    for f in component_files:
        name = os.path.splitext(os.path.basename(f))[0]
        if not _has_existing_doc(name):
            target = f
            break

    if target is None:
        log_path = _write_work_log("write_docs", "対象なし", "ドキュメント未整備のコンポーネントは見つかりませんでした。")
        return {"status": "skipped", "log_path": log_path}

    try:
        with open(target, "r", encoding="utf-8-sig") as f:
            code = f.read()
    except Exception as e:
        log_path = _write_work_log("write_docs", f"読み込み失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    component_name = os.path.splitext(os.path.basename(target))[0]
    _p.update(generation, 40, f"{component_name} のドキュメントを書いています")

    prompt = f"""
以下のReactコンポーネントについて、開発者向けドキュメント（Markdown）を書いてください。

コンポーネント名: {component_name}
コード:
{code[:4000]}

以下の項目を含めること：
- 概要（このコンポーネントが何をするか）
- props一覧（あれば）
- 主要な内部状態（useStateなど）
- 使用例（簡単なコードスニペット）

絶対厳守（重要）：
- 上のコード内に実際に存在するprops・state・関数・ラベル文言のみを記述すること
- 「カウンター」「増やすボタン」のような、上のコードに存在しない一般的な
  サンプルパターンを流用することは固く禁止する
- 必ず {component_name} の実際の中身を読み取った上で、それに基づいてドキュメントを
  書くこと。コードを見ずに書いたような当てずっぽうの説明は不可
- 出力はMarkdownの本文そのものであること。JSON形式（{{...}}のような
  構造化オブジェクト）で出力することは固く禁止する。見出し(#)や箇条書き(-)を
  使った通常のMarkdown文書として書くこと

Markdown形式で、コードのみを出力し、余計な前置きは不要です。
"""

    _p.update(generation, 70, "ドキュメントを生成しています")
    try:
        result = ask_ai(prompt, model=model_name, json_mode=False)
        doc_text = ""
        if isinstance(result, dict):
            doc_text = result.get("code") or result.get("content") or ""
        if not doc_text:
            doc_text = str(result)
    except OllamaUnavailableError as e:
        log_path = _write_work_log("write_docs", f"Ollama未起動: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}
    except Exception as e:
        log_path = _write_work_log("write_docs", f"生成失敗: {target}", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    try:
        os.makedirs(DOCS_OUTPUT_DIR, exist_ok=True)
        doc_path = os.path.join(DOCS_OUTPUT_DIR, f"{component_name}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_text)
        print(f"[WORK] ドキュメントを保存しました: {doc_path}")
    except Exception as e:
        log_path = _write_work_log("write_docs", f"保存失敗: {target}", f"生成はできたが保存に失敗: {e}\n\n---生成内容---\n{doc_text}")
        return {"status": "error", "log_path": log_path}

    log_path = _write_work_log("write_docs", target, f"ドキュメントを作成しました: {doc_path}")
    return {"status": "done", "target": target, "doc_path": doc_path, "log_path": log_path}


# ─────────────────────────────────────────
# サイクル4: self_improve（提案のみ、自動適用しない）
# ─────────────────────────────────────────

def _collect_recent_error_patterns() -> list:
    """
    error_memory.py の error_patterns.json から、頻出エラーパターンの上位を集める。
    存在しない・取得失敗しても空リストを返して処理は継続する。
    """
    try:
        import error_memory
        patterns = error_memory.load_patterns()
        sorted_patterns = sorted(patterns, key=lambda x: x.get("count", 0), reverse=True)
        return sorted_patterns[:5]
    except Exception as e:
        print(f"[WORK] error_memory からの取得に失敗しました（無視して続行）: {e}")
        return []


def _read_self_improve_target_files() -> str:
    """
    self_improve の提案対象となるファイルの実際の中身を読み込み、
    プロンプトに埋め込むためのテキストにまとめる。

    【2026/07/23 修正】以前はエラーパターンの統計だけを見て改善案を
    出していたため、実際にはbuild_prompt.pyに存在しないget_user_input()や
    bug_detection_handler.pyに存在しないvalidate_data()のような、実在しない
    関数名を創作して提案してしまう問題があった（対象コードを一度も読み込んで
    いなかったことが原因）。実際のソースコードから関数一覧を読み込ませることで、
    実在する関数のみを対象にした提案を出せるようにする。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blocks = []
    for filename in SELF_IMPROVE_TARGET_FILES:
        filepath = os.path.join(script_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                content = f.read()
            func_defs = re.findall(r"^def\s+(\w+)\s*\(", content, re.MULTILINE)
            blocks.append(f"■ {filename} に実在する関数一覧:\n" + "\n".join(f"  - {name}()" for name in func_defs))
        except Exception as e:
            blocks.append(f"■ {filename}: 読み込み失敗（{e}）")
    return "\n\n".join(blocks)


def _run_self_improve_cycle(generation, model_name: str = "router") -> dict:
    _p.update(generation, 10, "直近の失敗パターンを集めています")
    patterns = _collect_recent_error_patterns()

    if not patterns:
        log_path = _write_work_log("self_improve", "対象なし", "頻出エラーパターンが見つかりませんでした（error_patterns.jsonが空か未生成）。")
        return {"status": "skipped", "log_path": log_path}

    patterns_text = "\n".join(
        f"- {p.get('type', '不明')}: {p.get('detail', '')}（発生回数: {p.get('count', 0)}回）"
        for p in patterns
    )

    success_summary = ""
    try:
        import error_memory
        success_summary = error_memory.get_top_success_patterns_prompt()
    except Exception as e:
        print(f"[WORK] success_patterns の取得に失敗しました（無視して続行）: {e}")

    success_block = ""
    if success_summary:
        success_block = f"""
参考：現在うまくいっているパターン（これを崩さないよう提案すること）：
{success_summary}
"""

    _p.update(generation, 30, "対象ファイルの実際のコードを読み込んでいます")
    actual_functions = _read_self_improve_target_files()

    prompt = f"""
あなたはAI_001という名前のAIエージェント自身の改善案を考える役割です。

頻出している失敗パターン（発生回数が多い順）：
{patterns_text}
{success_block}

以下は、提案の対象にできるファイルに実際に存在する関数の一覧です。
これ以外の関数名を提案に使うことは絶対に禁止します。
{actual_functions}

これらの失敗を今後防ぐために、AI_001のコード生成プロンプトやバリデーションロジックに
どのような改善を加えるべきか、具体的な提案を日本語で書いてください。

- 上記の「実在する関数一覧」に無い関数名を、絶対に提案に含めないこと
- 提案する関数が本当に一覧に存在するか、書く前に必ず確認すること
- 「router.pyの_validate_sectionに〇〇のチェックを追加する」のように、
  ファイル名や実在する関数名レベルで具体的な提案にすること
- 3つ程度の提案にまとめること
- 発生回数が多いエラーを優先的に扱うこと
"""

    _p.update(generation, 50, "改善案を考えています")
    try:
        result = ask_ai(prompt, model=model_name, json_mode=False)
        proposal_text = ""
        if isinstance(result, dict):
            proposal_text = result.get("code") or result.get("content") or str(result)
        else:
            proposal_text = str(result)
    except OllamaUnavailableError as e:
        log_path = _write_work_log("self_improve", "Ollama未起動", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}
    except Exception as e:
        log_path = _write_work_log("self_improve", "提案生成失敗", f"エラー: {e}")
        return {"status": "error", "log_path": log_path}

    log_path = _write_work_log("self_improve", "AI_001改善提案", proposal_text)
    return {"status": "done", "log_path": log_path}

# ─────────────────────────────────────────
# メインループ（3/6/9時間 or 無限）
# ─────────────────────────────────────────

def run_work_cycle(generation: int, duration_hours: float = None, model_name: str = DEFAULT_MODEL, max_cycles: int = None) -> dict:
    """
    「仕事させる」ボタンから呼ばれるメイン関数。

    duration_hours: 3 / 6 / 9 を指定するとその時間で自動終了。
                     None を指定すると無限（progress.abort()まで）。
    max_cycles: 指定すると、そのサイクル数に達した時点で終了する
                （2026/07/23 追加：「1回」ボタン用。max_cycles=1 なら
                scan_and_fixだけを1回実行して終了する）。
    model_name: 各サイクルで使うコード生成モデル（デフォルトはqwen2.5-coder:3b）。
    """
    deadline = None
    if duration_hours is not None:
        deadline = time.time() + duration_hours * 3600
        print(f"[WORK] 仕事モードを開始します（{duration_hours}時間・モデル={model_name}）")
    elif max_cycles is not None:
        print(f"[WORK] 仕事モードを開始します（{max_cycles}回のみ・モデル={model_name}）")
    else:
        print(f"[WORK] 仕事モードを開始します（無期限・モデル={model_name}）")

    cycle_count = 0
    results = []

    while True:
        if not _p.is_current_generation(generation):
            print(f"[WORK] 中断を検知しました（{cycle_count}サイクル実行後）。仕事モードを終了します。")
            break
        if deadline is not None and time.time() >= deadline:
            print(f"[WORK] 指定時間（{duration_hours}時間）に達したため終了します（{cycle_count}サイクル実行）。")
            break
        if max_cycles is not None and cycle_count >= max_cycles:
            print(f"[WORK] 指定サイクル数（{max_cycles}回）に達したため終了します。")
            break

        cycle_type = WORK_CYCLE_TYPES[cycle_count % len(WORK_CYCLE_TYPES)]
        cycle_count += 1
        _p.update(generation, 0, f"{cycle_count}回目（{cycle_type}）を実行中")

        try:
            if cycle_type == "scan_and_fix":
                result = _run_scan_and_fix_cycle(generation, model_name)
            elif cycle_type == "write_tests":
                result = _run_write_tests_cycle(generation, model_name)
            elif cycle_type == "write_docs":
                result = _run_write_docs_cycle(generation, model_name)
            else:  # self_improve
                result = _run_self_improve_cycle(generation, model_name="router")
        except Exception as e:
            print(f"[WORK] {cycle_count}回目（{cycle_type}）でエラーが発生しました（無視して続行）: {e}")
            time.sleep(30)
            continue

        result["cycle_type"] = cycle_type
        result["cycle_count"] = cycle_count
        results.append(result)

        if not _p.is_current_generation(generation):
            print(f"[WORK] 中断を検知しました（{cycle_count}サイクル実行後）。仕事モードを終了します。")
            break
        if deadline is not None and time.time() >= deadline:
            print(f"[WORK] 指定時間（{duration_hours}時間）に達したため終了します（{cycle_count}サイクル実行）。")
            break
        if max_cycles is not None and cycle_count >= max_cycles:
            print(f"[WORK] 指定サイクル数（{max_cycles}回）に達したため終了します。")
            break

        # サイクル間の休憩（60秒、中断・時間切れチェック付き）
        for _ in range(60):
            if not _p.is_current_generation(generation):
                break
            if deadline is not None and time.time() >= deadline:
                break
            if max_cycles is not None and cycle_count >= max_cycles:
                break
            time.sleep(1)

    return {
        "status": "done",
        "cycle_count": cycle_count,
        "results": results,
    }