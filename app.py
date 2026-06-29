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