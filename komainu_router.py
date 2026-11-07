# server/komainu_router.py
import os, json, requests
from groq import Groq
from cerebras.cloud.sdk import Cerebras

GROQ_MODEL = "llama-3.3-70b-versatile"
CEREBRAS_MODEL = "llama3.3-70b"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

def get_system(role="judge"):
    if role == "judge":
        return 'あなたは狛犬先生。出力は必ずJSONのみ: {"winner":"A or B","reason":"100字以内","score_a":0-100,"score_b":0-100}'
    return "あなたは狛犬先生コメンテーター"

def _set_model(name: str):
    try:
        import progress
        progress.set_active_model(name)
    except Exception:
        pass


def call_komainu(prompt: str, role="judge"):
    system = get_system(role)
    errors = []

    # 1. Groq
    try:
        if os.getenv("GROQ_API_KEY"):
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            print("[Komainu] Trying Groq-70B...")
            _set_model(f"先生 (Groq {GROQ_MODEL})")
            res = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
                temperature=0.3, response_format={"type":"json_object"}
            )
            return json.loads(res.choices[0].message.content)
    except Exception as e:
        errors.append(f"Groq: {e}")

    # 2. Cerebras
    try:
        if os.getenv("CEREBRAS_API_KEY"):
            client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
            print("[Komainu] Trying Cerebras-70B...")
            _set_model(f"先生 (Cerebras {CEREBRAS_MODEL})")
            res = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            )
            raw = res.choices[0].message.content
            return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception as e:
        errors.append(f"Cerebras: {e}")

    # 3. OpenRouter
    try:
        if os.getenv("OPENROUTER_API_KEY"):
            print("[Komainu] Trying OpenRouter-70B-free...")
            _set_model(f"先生 (OpenRouter {OPENROUTER_MODEL})")
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", "Content-Type":"application/json"},
                json={"model": OPENROUTER_MODEL, "messages":[{"role":"system","content":system},{"role":"user","content":prompt}]},
                timeout=30
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception as e:
        errors.append(f"OpenRouter: {e}")

    raise RuntimeError(f"All Komainu APIs failed: {errors}")
# --- Judge用ラッパー ---
def call_komainu_judge(prompt: str):
    """採点用 - 既存のフォールバック付きjudgeを呼ぶだけ"""
    return call_komainu(prompt, role="judge")