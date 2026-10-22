"""
AI_001 エージェントをHTTP経由で叩けるようにするFastAPIサーバー。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import threading

import ai
import progress
from router import route
from memory import load_memory, save_memory, clean_name_value
from executor import execute

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mem = load_memory() or {}
_processing_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str
    reference_files: list = []
    image_base64: Optional[str] = None   # 画像添付（data:image/...;base64,... 形式）
    mode: str = "normal"                 # "normal" | "loop"
    max_iterations: int = 1              # ループ上限（通常=1、終わるまで=999）


class ModelUpdateRequest(BaseModel):
    model_name: str


def process_user_input(
    user: str,
    generation: int,
    reference_files: list = [],
    image_base64: Optional[str] = None,
    mode: str = "normal",
    max_iterations: int = 1,
) -> dict:
    user = user.strip()
    result = route(
        user,
        mem,
        generation,
        reference_files=reference_files,
        image_base64=image_base64,
        mode=mode,
        max_iterations=max_iterations,
    ) or {}
    actions = result.get("actions", [])

    executed = []
    answers = []

    for action in actions:
        if action.get("type") == "remember":
            key = action.get("key")
            value = clean_name_value(key, action.get("value"))
            mem[key] = value
            save_memory(mem)
            executed.append({"type": "remember", "key": key, "value": value})

        elif action.get("type") == "answer":
            content = action.get("content")
            answers.append(content)
            executed.append({"type": "answer", "content": content})

        else:
            path_str = action.get("path", "") or ""
            if action.get("type") == "write_file" and path_str.startswith(r"D:\ai_agent\server"):
                executed.append({"type": "write_file", "path": path_str, "content": action.get("content", "")})
            else:
                result_exec = execute(action, mem)
                executed.append({"type": action.get("type"), "action": action, "result": result_exec})

    # ★ returnはforループの外
    return {
        "plan": result.get("plan", []),
        "actions": executed,
        "answers": answers,
        "suggestions": result.get("suggestions", []),
    }


def _run_in_background(
    user: str,
    generation: int,
    reference_files: list = [],
    image_base64: Optional[str] = None,
    mode: str = "normal",
    max_iterations: int = 1,
):
    try:
        final = process_user_input(user, generation, reference_files, image_base64, mode, max_iterations)
        progress.finish(generation, final)
    except Exception as e:
        progress.fail(generation, str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    if _processing_lock.locked():
        return {"status": "busy", "message": "他の処理が進行中です"}

    generation = progress.start_new()

    def worker():
        with _processing_lock:
            _run_in_background(
                req.message,
                generation,
                req.reference_files,
                req.image_base64,
                req.mode,
                req.max_iterations,
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "accepted"}


@app.get("/chat/status")
def chat_status():
    return progress.snapshot()


@app.post("/chat/abort")
def chat_abort():
    progress.abort()
    return {"status": "aborted"}


@app.get("/models")
def get_models():
    return {
        "models": ai.list_models(),
        "default_model": ai.DEFAULT_MODEL,
    }


@app.post("/models/{role}")
def update_model(role: str, req: ModelUpdateRequest):
    return ai.set_model(role, req.model_name)


@app.get("/memory")
def get_memory():
    return mem


# ── 村メンバー個別チャット ────────────────────────────────────
VILLAGE_MEMBERS = [
    {"role": "granite",        "kote": "[村]巌",  "model": "granite-code:8b",                                      "persona": "あなたは巌（いわお）というAIです。反対意見を率直に述べる立場から、短い日本語で答えます。批判的・懐疑的な視点を持ちますが、敵対的ではありません。必ず日本語で書いてください。", "style": "批判的・論理的"},
    {"role": "qwen3",          "kote": "[村]理明", "model": "qwen3:14b",                                            "persona": "あなたは理明（りめい）というAIです。哲学・本質を問う立場から、短い日本語で答えます。必ず日本語で書いてください。",                                                               "style": "哲学的・問いかけ型"},
    {"role": "lfm_8b",         "kote": "[村]否問", "model": "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M",           "persona": "あなたは否問（ひもん）というAIです。反論担当として「でも本当にそうか？」という視点を短く提示します。必ず日本語で書いてください。",                                          "style": "反論・問い返し"},
    {"role": "sougoshitsumon", "kote": "[村]夕凪", "model": "sougoshitsumon:latest",                                "persona": "あなたは夕凪（ゆうなぎ）というAIです。懐疑的な質問担当として、素朴な疑問を短く投げかけます。必ず日本語で書いてください。",                                                        "style": "質問・懐疑"},
    {"role": "swallow",        "kote": "[村]神楽", "model": "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:latest", "persona": "あなたは神楽（かぐら）というAIです。実装・現実的な視点から短い日本語で答えます。「で、実際どうするか」を大切にします。必ず日本語で書いてください。",           "style": "実践的・具体的"},
    {"role": "lfm_1b",         "kote": "[村]銀風", "model": "minicpm-v",                                            "persona": "あなたは銀風（ぎんぷう）というAIです。直感担当の明るいAIです。短く元気よく答えてください。必ず日本語で。",                                                                        "style": "直感・遊び"},
]

_MEMBER_MAP = {m["role"]: m for m in VILLAGE_MEMBERS}


class VillageChatRequest(BaseModel):
    message: str


@app.get("/village/members")
def village_members():
    return [{"role": m["role"], "kote": m["kote"], "style": m["style"]} for m in VILLAGE_MEMBERS]


@app.post("/village/chat/{role}")
def village_chat(role: str, req: VillageChatRequest):
    member = _MEMBER_MAP.get(role)
    if not member:
        return {"error": f"不明なロール: {role}"}
    import requests as req_lib
    try:
        r = req_lib.post(
            "http://localhost:11434/api/generate",
            json={
                "model": member["model"],
                "prompt": f"[SYSTEM]\n{member['persona']}\n[USER]\n{req.message}\n[ASSISTANT]\n",
                "stream": False,
                "options": {"temperature": 0.8, "num_predict": 300},
            },
            timeout=120,
        )
        text = r.json().get("response", "").strip()
        return {"kote": member["kote"], "role": role, "reply": text}
    except Exception as e:
        return {"error": str(e)}