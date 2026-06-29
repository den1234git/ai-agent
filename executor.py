from memory import save_memory, clean_name_value
import os

MOCKUP_DIR = os.environ.get("MOCKUP_DIR", r"D:\ai_agent\mockups")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", r"D:\ai_agent\output")
SHOBO_DIR = os.environ.get("SHOBO_APP_DIR", r"D:\ai_agent\shobo-app")

print(f"[executor] shobo-app書き込み先: {SHOBO_DIR}")
print(f"[executor] mockups書き込み先: {MOCKUP_DIR}")
print(f"[executor] output書き込み先: {OUTPUT_DIR}")


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
        if "." not in path_str:
            path_str = path_str + ".ts"

        if path_str.startswith("mockups/"):
            filename = path_str[len("mockups/"):]
            path = os.path.join(MOCKUP_DIR, filename)
        elif path_str.startswith("src/"):
            path = os.path.join(SHOBO_DIR, path_str)
        else:
            path = os.path.join(OUTPUT_DIR, path_str)

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

    return {"error": "unknown action"}