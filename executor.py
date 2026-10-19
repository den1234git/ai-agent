from memory import save_memory, clean_name_value
import os
import time
import random
import requests as _requests

FNOITA_API = "https://fnoita-board.mogumogu7la001.workers.dev"

def _board_new_id():
    t = int(time.time() * 1000)
    return f"{t:x}{random.randint(0, 0xfff):x}"

def _board_post(cat_id, title, body, kote="LA"):
    tid = _board_new_id()
    thread = {"id": tid, "title": title, "kote": kote, "at": int(time.time()*1000)}
    post = {"id": _board_new_id(), "kote": kote, "body": body, "tok": 0, "at": int(time.time()*1000)}
    r = _requests.post(f"{FNOITA_API}/api/threads/{cat_id}",
                       json={"thread": thread, "post": post}, timeout=10)
    return r.ok

def _board_reply(tid, body, kote="LA"):
    post = {"id": _board_new_id(), "kote": kote, "body": body, "tok": 0, "at": int(time.time()*1000)}
    r = _requests.post(f"{FNOITA_API}/api/posts/{tid}",
                       json={"post": post}, timeout=10)
    return r.ok

MOCKUP_DIR = os.environ.get("MOCKUP_DIR", r"F:\ai_agent\mockups")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"F:\ai_agent\output")
SHOBO_DIR = os.environ.get("SHOBO_APP_DIR", r"F:\ai_agent\shobo-app")

print(f"[executor] shobo-app書き込み先: {SHOBO_DIR}")
print(f"[executor] mockups書き込み先: {MOCKUP_DIR}")
print(f"[executor] output書き込み先: {OUTPUT_DIR}")


def _safe_write_path(base_dir: str, rel_path: str) -> str:
    """
    base_dir 配下に収まる書き込みパスを返す。
    - 絶対パス指定は即拒否
    - os.path.abspath で正規化後、base_dir + os.sep で始まるか検証
    - トラバーサル（../等）やドライブレターが含まれている場合も弾かれる
    収まっていなければ ValueError を送出する（呼び元でエラーを返す）。
    """
    if os.path.isabs(rel_path):
        raise ValueError(f"絶対パス指定は不可: '{rel_path}'")
    base_abs = os.path.abspath(base_dir)
    full = os.path.abspath(os.path.join(base_abs, rel_path))
    if not full.startswith(base_abs + os.sep):
        raise ValueError(f"許可ディレクトリ外への書き込みは不可: '{rel_path}' → '{full}'")
    return full


def _save_experience_async(task: str, code: str, file_path: str):
    """
    write_file成功後にla_experienceへ非同期で保存する。
    ChromaDBの初回ロード（モデルダウンロード）が遅い場合があるため
    別スレッドで実行し、メインの応答速度に影響させない。
    """
    import threading

    def _worker():
        try:
            from memory_store import save_experience
            from ai import ask_ai

            # LAに「何を学んだか・どう感じたか」を短く生成させる
            reflection_prompt = f"""
以下のコード生成タスクを振り返ってください。

タスク：{task[:200]}
生成ファイル：{file_path}

以下のJSONのみ返してください：
{{
  "learned": "このタスクから学んだことを1文で",
  "reflection": "生成してみての感想を1文で（LAとして一人称で）"
}}
"""
            result = ask_ai(reflection_prompt, model="router")
            learned = result.get("learned", "")
            reflection = result.get("reflection", "")

            save_experience(
                task=task,
                code=code,
                outcome="success",
                learned=learned,
                reflection=reflection,
                file_path=file_path,
            )
        except Exception as e:
            print(f"[executor] 経験の保存に失敗しました（無視して続行）: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def execute(action, mem):
    t = action.get("type")

    if t in ("write_file", "create_file"):
        path_str = action.get("path") or action.get("key") or action.get("filename", "output.ts")

        # ★ 追加：ファイル名部分が空、またはディレクトリ末尾で終わる不正なpathを検出する
        if not path_str.strip() or path_str.strip().endswith(("/", "\\")):
            return {"error": f"不正なファイルパスです（空、またはディレクトリのみ）: '{path_str}'"}

        if "." not in path_str:
            path_str = path_str + ".ts"

        try:
            if path_str.startswith("mockups/"):
                filename = path_str[len("mockups/"):]
                _safe_write_path(MOCKUP_DIR, filename)
                path = os.path.join(os.path.abspath(MOCKUP_DIR), filename)
            elif path_str.startswith("src/"):
                _safe_write_path(SHOBO_DIR, path_str)
                path = os.path.join(os.path.abspath(SHOBO_DIR), path_str)
            else:
                _safe_write_path(OUTPUT_DIR, path_str)
                path = os.path.join(os.path.abspath(OUTPUT_DIR), path_str)
        except ValueError as e:
            print(f"[executor] パストラバーサル拒否: {e}")
            return {"error": f"パストラバーサル拒否: {e}"}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = action.get("content", "")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        # ★ 書き込み成功後、経験をChromaDBに非同期保存
        task = action.get("task") or action.get("description") or path_str
        _save_experience_async(task=task, code=content, file_path=path)

        return {"status": "written", "path": path}

    if t == "read_file":
        path = os.path.join(SHOBO_DIR, action["path"])
        with open(path, encoding="utf-8") as f:
            return {"content": f.read()}

    if t == "remember":
        key = action["key"]
        value = clean_name_value(key, action["value"])
        mem[key] = value
        save_memory(mem)
        return {"status": "saved"}

    if t == "answer":
        return {"answer": action.get("content")}

    if t == "board_post":
        kote = action.get("kote", "LA")
        cat_id = action.get("cat_id", "f11")
        title = action.get("title", "LAの投稿")
        body = action.get("body", "")
        ok = _board_post(cat_id, title, body, kote)
        return {"status": "posted" if ok else "failed", "cat_id": cat_id, "title": title}

    if t == "board_reply":
        kote = action.get("kote", "LA")
        tid = action.get("tid", "")
        body = action.get("body", "")
        ok = _board_reply(tid, body, kote)
        return {"status": "replied" if ok else "failed", "tid": tid}

    return {"error": "unknown action"}