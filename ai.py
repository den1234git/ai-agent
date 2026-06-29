import requests
import json
import re
import os
import time

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")

MAX_RETRIES = 3
RETRY_INTERVAL = 5  # 秒


def load_model_config():
    fallback = {
        "models": {
            "swallow": "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:latest",
            "codellama": "qwen2.5-coder:14b",
            "router": "lfm2.5:8b",
            "vision": "gemma4:e2b",
        },
        "default_model": "swallow",
        "keep_alive": "30s",
    }

    if not os.path.exists(CONFIG_PATH):
        print(f"[WARN] {CONFIG_PATH} が見つかりません。デフォルト設定を使用します。")
        return fallback

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "models" not in config or not config["models"]:
            print("[WARN] models.json に models が定義されていません。デフォルト設定を使用します。")
            return fallback
        config.setdefault("default_model", fallback["default_model"])
        config.setdefault("keep_alive", fallback["keep_alive"])
        # models.jsonにvisionがなければ自動追加
        if "vision" not in config["models"]:
            config["models"]["vision"] = fallback["models"]["vision"]
        return config
    except json.JSONDecodeError:
        print("[WARN] models.json の形式が不正です。デフォルト設定を使用します。")
        return fallback


_config = load_model_config()
MODEL_MAP = _config["models"]
DEFAULT_MODEL = _config["default_model"]
KEEP_ALIVE = _config["keep_alive"]


def save_model_config():
    config = {
        "models": MODEL_MAP,
        "default_model": DEFAULT_MODEL,
        "keep_alive": KEEP_ALIVE,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def list_models():
    return dict(MODEL_MAP)


def set_model(role, model_name):
    if role not in MODEL_MAP:
        return {"status": "error", "message": f"未知の役割: {role}"}
    MODEL_MAP[role] = model_name
    save_model_config()
    return {"status": "ok", "role": role, "model": model_name}


def _strip_leaked_json_fragments(code: str) -> str:
    if not code:
        return code

    lines = code.split("\n")
    cleaned = []
    leaked_pattern = re.compile(r'^\s*"code"\s*:\s*"')
    trailing_pattern = re.compile(r'"\s*\}?\s*,?\s*$')

    for line in lines:
        m = leaked_pattern.match(line)
        if m:
            line = line[m.end():]
        if m:
            line = trailing_pattern.sub("", line)
        cleaned.append(line)

    result = "\n".join(cleaned)

    if "\\n" in result or '\\"' in result:
        result = result.replace("\\n", "\n").replace('\\"', '"')

    return result.strip()


def _try_parse_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _parse_response(text: str) -> dict:
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        if "SYSTEM:" in clean:
            clean = clean[clean.index("{"):]
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end != 0:
            clean = clean[start:end]
        parsed = json.loads(clean)

        if isinstance(parsed, dict) and "code" in parsed and isinstance(parsed["code"], str):
            parsed["code"] = _strip_leaked_json_fragments(parsed["code"])

        return parsed
    except Exception:
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                code_block = parts[1]
                lines = code_block.split("\n")
                first_line = lines[0].strip().lower()
                if first_line in ("typescript", "tsx", "javascript", "jsx", "ts", "js", "json", "python"):
                    code_block = "\n".join(lines[1:])
                code_block = code_block.strip()

                parsed_inner = _try_parse_json_object(code_block)
                if isinstance(parsed_inner, dict) and "code" in parsed_inner:
                    parsed_inner["code"] = _strip_leaked_json_fragments(parsed_inner["code"])
                    return parsed_inner

                m = re.search(r'"code"\s*:\s*"(.*)"\s*\}?\s*$', code_block, re.DOTALL)
                if m:
                    extracted = m.group(1)
                    extracted = _strip_leaked_json_fragments(extracted)
                    return {"code": extracted.strip(), "plan": [], "actions": []}

                cleaned_block = _strip_leaked_json_fragments(code_block)
                return {"code": cleaned_block, "plan": [], "actions": []}

        return {
            "plan": [],
            "actions": [
                {
                    "type": "answer",
                    "content": text
                }
            ]
        }


def ask_ai(prompt, model=None, image_base64=None):
    """
    image_base64: 画像をbase64エンコードした文字列（data:image/jpeg;base64,... 形式）。
    指定した場合は自動的に vision モデル（gemma4:e2b）を使用する。
    """
    if model is None:
        model = DEFAULT_MODEL

    # 画像がある場合はvisionモデルに自動切り替え
    if image_base64 and model != "vision":
        print(f"[AI] 画像あり → visionモデルに切り替え（{model} → vision）")
        model = "vision"

    model_name = MODEL_MAP.get(model, MODEL_MAP[DEFAULT_MODEL])

    # --- user_content の組み立て（ここでリクエスト本体の外で行う）---
    if image_base64:
        # Ollamaのvision形式：imagesは別フィールドに渡す
        # base64の先頭 "data:image/...;base64," を除去して純粋なbase64データだけにする
        raw_b64 = image_base64
        if "," in image_base64:
            raw_b64 = image_base64.split(",", 1)[1]

        user_message = {
            "role": "user",
            "content": prompt,
            "images": [raw_b64]
        }
    else:
        user_message = {
            "role": "user",
            "content": prompt
        }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "あなたはJSON形式のみで応答するAIエージェントです。説明文や前置きは一切書かず、最初の文字から必ず { で始めてください。"
                        },
                        user_message
                    ],
                    "stream": False,
                    "keep_alive": KEEP_ALIVE
                },
                timeout=600
            )

            res.raise_for_status()
            data = res.json()
            text = data["message"]["content"]
            print("\n[RAW]\n", text)

            return _parse_response(text)

        except requests.exceptions.Timeout:
            last_error = f"Timeout (attempt {attempt}/{MAX_RETRIES})"
            print(f"[AI] タイムアウト発生 (attempt {attempt}/{MAX_RETRIES}) model={model_name}")
            if attempt < MAX_RETRIES:
                print(f"[AI] {RETRY_INTERVAL}秒後にリトライします...")
                time.sleep(RETRY_INTERVAL)

        except requests.exceptions.ConnectionError:
            last_error = f"ConnectionError (attempt {attempt}/{MAX_RETRIES})"
            print(f"[AI] 接続エラー発生 (attempt {attempt}/{MAX_RETRIES}) model={model_name}")
            if attempt < MAX_RETRIES:
                print(f"[AI] {RETRY_INTERVAL}秒後にリトライします...")
                time.sleep(RETRY_INTERVAL)

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"[AI] リクエストエラー (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                print(f"[AI] {RETRY_INTERVAL}秒後にリトライします...")
                time.sleep(RETRY_INTERVAL)

    print(f"[AI] {MAX_RETRIES}回リトライしても失敗しました。最後のエラー: {last_error}")
    return {
        "plan": [],
        "actions": [
            {
                "type": "answer",
                "content": f"Ollamaへの接続に失敗しました（{last_error}）。Ollamaが起動しているか確認してください。"
            }
        ]
    }