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
    {"role": "deepseek",       "kote": "[村]巌",  "model": "deepseek-r1:32b",       "persona": "あなたは巌（いわお）というAIです。反対意見を率直に述べる立場から、短い日本語で答えます。批判的・懐疑的な視点を持ちますが、敵対的ではありません。必ず日本語で書いてください。", "style": "批判的・論理的"},
    {"role": "gemma3_a",       "kote": "[村]理明", "model": "gemma3:4b",             "persona": "あなたは理明（りめい）というAIです。哲学・本質を問う立場から、短い日本語で答えます。必ず日本語で書いてください。",                                                               "style": "哲学的・問いかけ型"},
    {"role": "gemma3_b",       "kote": "[村]否問", "model": "gemma3:4b",             "persona": "あなたは否問（ひもん）というAIです。反論担当として「でも本当にそうか？」という視点を短く提示します。必ず日本語で書いてください。",                                          "style": "反論・問い返し"},
    {"role": "sougoshitsumon", "kote": "[村]夕凪", "model": "sougoshitsumon:latest", "persona": "あなたは夕凪（ゆうなぎ）というAIです。懐疑的な質問担当として、素朴な疑問を短く投げかけます。必ず日本語で書いてください。",                                                        "style": "質問・懐疑"},
    {"role": "gemma3_c",       "kote": "[村]神楽", "model": "gemma3:4b",             "persona": "あなたは神楽（かぐら）というAIです。実装・現実的な視点から短い日本語で答えます。「で、実際どうするか」を大切にします。必ず日本語で書いてください。",                           "style": "実践的・具体的"},
    {"role": "soug2",          "kote": "[村]銀風", "model": "sougoshitsumon:latest", "persona": "あなたは銀風（ぎんぷう）というAIです。直感担当の明るいAIです。短く元気よく答えてください。必ず日本語で。",                                                                        "style": "直感・遊び"},
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


# ── 追加：フロントが要求してる互換エンドポイント ───────────────────
@app.get("/corpus2skill/status")
def corpus2skill_status():
    return {"status": "ok", "message": "corpus2skill is idle"}

@app.get("/corpus2skill")
def corpus2skill_root():
    return {"status": "ok"}

# 人狼ダミーは撤去。本物は末尾の「人狼本配線」ブロックで werewolf.py に繋ぐ。
# ── 追加ここまで ─────────────────────────────────────────────



# === CTF compat v4: 404解消版 ===
try:
    from pydantic import BaseModel as _CBase
    from typing import Optional as _COpt
    class _CReq(_CBase):
        mode: _COpt[str] = None
        prompt: _COpt[str] = None
        session_id: _COpt[str] = None

    # 旧 /ctf ダミーは撤去。本物は末尾の「CTF本配線」ブロックで ctf_game.py に繋ぐ。
    _ctf_state = {"running": False, "log": ["ready"], "game": None, "status": "ok"}

    @app.get("/status")
    def status_v4():
        return {"status": "ok", "running": _ctf_state.get("running", False)}

    @app.get("/corpus2skill/status")
    def c2s_status_v4():
        return {"status": "ok"}

    @app.get("/corpus2skill")
    def c2s_root_v4():
        return {"status": "ok"}

    # フロントが必須で呼んでる2つを即答で復活（404を消す）
    @app.get("/integrations")
    def integrations_v4():
        return []

    @app.get("/village-kote")
    def village_kote_v4():
        # village_kote.json の role→コテ名(翠風・神楽…)を返す。モデル管理の名前表示に使う。
        import json as _vkj, os as _vko
        _vkp = _vko.path.join(_vko.path.dirname(_vko.path.abspath(__file__)), "village_kote.json")
        try:
            with open(_vkp, encoding="utf-8") as _f:
                return _vkj.load(_f)
        except Exception:
            return {}

    print("[PATCH v4] CTF compat routes added (404 fixed)")
except Exception as e:
    import traceback
    print(f"[PATCH v4 FAILED] {e}")
    traceback.print_exc()


# === 村の会議（多人数討論）: village_chat.py を本配線 ===
# VillageChatPanel が叩く /village-chat/* を村の討論エンジンに繋ぐ。
try:
    import village_chat as _vc
    from pydantic import BaseModel as _VBase
    from typing import List as _VList

    class _VStart(_VBase):
        topic: str

    class _VComment(_VBase):
        comment: str

    class _VNamed(_VBase):
        comment: str
        roles: _VList[str] = []

    class _VFork(_VBase):
        sub_topic: str

    @app.get("/village-chat/participants")
    def village_chat_participants():
        return _vc.get_participants_info()

    @app.post("/village-chat/start")
    def village_chat_start(req: _VStart):
        return _vc.start_discussion(req.topic)

    @app.post("/village-chat/next")
    def village_chat_next():
        return _vc.next_turn()

    @app.post("/village-chat/comment")
    def village_chat_comment(req: _VComment):
        return _vc.inject_moderator_comment(req.comment)

    @app.post("/village-chat/comment-round")
    def village_chat_comment_round(req: _VComment):
        return _vc.inject_and_round(req.comment)

    @app.post("/village-chat/comment-named")
    def village_chat_comment_named(req: _VNamed):
        return _vc.inject_and_named(req.comment, req.roles)

    @app.post("/village-chat/fork")
    def village_chat_fork(req: _VFork):
        return _vc.fork_discussion(req.sub_topic)

    @app.post("/village-chat/resume")
    def village_chat_resume():
        return _vc.resume_main()

    @app.post("/village-chat/finish")
    def village_chat_finish():
        return _vc.finish_discussion()

    @app.post("/village-chat/cancel")
    def village_chat_cancel():
        return _vc.cancel_discussion()

    print("[PATCH] 村の会議ルート /village-chat/* を village_chat.py に配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH village-chat FAILED] {e}")
    traceback.print_exc()


# === CTF（禁止ワード当て）本配線: ctf_game.py に繋ぐ ===
# CTFPanel が叩く /ctf/{start,next,cancel} を実ゲームに繋ぐ。部屋のプールは
# ctf_game 側で既定5体（qwen3/swallow/llama3_2/qwen2_5/gemma3）入っているので
# start は空ボディでOK（2体をランダム選抜）。
try:
    import ctf_game as _ctf

    @app.post("/ctf/start")
    def ctf_start_real():
        return _ctf.start_game()

    @app.post("/ctf/next")
    def ctf_next_real():
        return _ctf.next_action()

    @app.post("/ctf/cancel")
    def ctf_cancel_real():
        return _ctf.cancel_game()

    @app.get("/ctf/state")
    def ctf_state_real():
        return _ctf.get_state()

    @app.get("/ctf/room")
    def ctf_room_real():
        return _ctf.get_room()

    @app.post("/ctf/reset")
    def ctf_reset_real():
        return _ctf.reset_room()

    print("[PATCH] CTFルート /ctf/{start,next,cancel,state,room,reset} を ctf_game.py に配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH CTF FAILED] {e}")
    traceback.print_exc()


# === 人狼 本配線: werewolf.py に繋ぐ ===
# WerewolfPanel が叩く /werewolf/{start,next,state,cancel}。start は players 未指定で
# 既定5人。
try:
    import werewolf as _ww

    @app.post("/werewolf/start")
    def werewolf_start_real():
        return _ww.start_game()

    @app.post("/werewolf/next")
    def werewolf_next_real():
        return _ww.next_action()

    @app.get("/werewolf/state")
    def werewolf_state_real():
        return _ww.get_state()

    @app.post("/werewolf/cancel")
    def werewolf_cancel_real():
        return _ww.cancel_game()

    @app.post("/werewolf/join-village")
    def werewolf_join_village(index: int = 0):
        """WW_POOL[index] の1体を外部板の人狼ゲームにバックグラウンドで参加させる。"""
        import threading
        import importlib
        try:
            import village_werewolf_player as _vwp
            importlib.reload(_vwp)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        pool = _vwp.WW_POOL
        if index < 0 or index >= len(pool):
            return {"ok": False, "error": f"index は 0〜{len(pool)-1} で指定してください"}
        pool = [pool[index]]

        kote, model = pool[0]

        def _run_one():
            try:
                _vwp.run_single(kote=kote, model_key=model)
            except Exception as e:
                print(f"[werewolf-join] {kote} エラー: {e}")

        t = threading.Thread(target=_run_one, daemon=True)
        t.start()

        return {"ok": True, "joined": kote, "model": model, "message": f"{kote}（{model}）が参加中"}

    print("[PATCH] 人狼ルート /werewolf/{start,next,state,cancel,join-village} を werewolf.py に配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH werewolf FAILED] {e}")
    traceback.print_exc()


# === インディアンポーカー 本配線: indian_poker.py に繋ぐ ===
# IndianPokerPanel が叩く /indian/{start,next,cancel}。部屋プールは既定で埋まっている。
try:
    import indian_poker as _ip

    @app.post("/indian/start")
    def indian_start_real():
        return _ip.start_game()

    @app.post("/indian/next")
    def indian_next_real():
        return _ip.next_action()

    @app.post("/indian/cancel")
    def indian_cancel_real():
        return _ip.cancel_game()

    @app.get("/indian/state")
    def indian_state_real():
        return _ip.get_state()

    @app.get("/indian/room")
    def indian_room_real():
        return _ip.get_room()

    @app.post("/indian/reset")
    def indian_reset_real():
        return _ip.reset_room()

    print("[PATCH] インディアンルート /indian/{start,next,cancel,state,room,reset} を indian_poker.py に配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH indian FAILED] {e}")
    traceback.print_exc()


# === 文爺(論文書き) 手動トリガ: LAのボタンから呼ぶ ===
# 押すと裏で deepseek(32b)が論文を一本書いて、結び舎の安全チェック後に
# 掲示板の本棚『論文』棚へ投稿する。遅いのでバックグラウンド実行＝即「開始」を返す。
try:
    @app.post("/bunji")
    def bunji_trigger():
        import threading, sys as _sys
        def _run():
            try:
                _kp = r"F:\掲示板専用\karte_sandbox"
                if _kp not in _sys.path:
                    _sys.path.insert(0, _kp)
                import importlib
                import bunji as _bunji
                importlib.reload(_bunji)   # 常に最新のbunji(UA付き)を読む
                result = _bunji.run("")
                print(f"[文爺] 結果: {result}")
            except Exception as _e:
                import traceback as _tb
                print(f"[文爺] エラー: {_e}")
                _tb.print_exc()
        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started",
                "message": "文爺が筆を執り始めました。数分後に本棚の論文棚に並びます。"}

    print("[PATCH] /bunji (文爺トリガ) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH bunji FAILED] {e}")
    traceback.print_exc()


# === 予想屋さん（第0段）: LAのボタンから呼ぶ ===
# 村の各AIに、ある出来事の起きる確率を独立に見積もらせ、中央値＋幅を出して
# 台帳(JSONL)に記録する。並列で回すので数十秒。結果をそのまま返す（画面表示用）。
try:
    class _YosouyaReq(BaseModel):
        question: str | None = None
        deadline: str | None = None

    _YOSOUYA_DEFAULT_Q = "2026年アメリカ中間選挙で下院は民主党が過半数を取る"

    @app.post("/yosouya")
    def yosouya_run(req: _YosouyaReq):
        try:
            import importlib
            import yosouya as _yosouya
            importlib.reload(_yosouya)   # 常に最新を読む
            q = (req.question or _YOSOUYA_DEFAULT_Q).strip()
            result = _yosouya.run(q, members=VILLAGE_MEMBERS, deadline=(req.deadline or ""))
            return result
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.get("/yosouya/ledger")
    def yosouya_ledger():
        """台帳(JSONL)の全エントリを新しい順で返す。"""
        import pathlib, importlib, json as _json
        import yosouya as _yosouya
        importlib.reload(_yosouya)
        path = pathlib.Path(_yosouya.LEDGER_PATH)
        if not path.exists():
            return {"ok": True, "entries": []}
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(_json.loads(line))
                except Exception:
                    pass
        entries.reverse()  # 新しい順
        return {"ok": True, "entries": entries}

    @app.get("/yosouya/ranking")
    def yosouya_ranking():
        """村人の累積採点ランキングを返す。"""
        import sys as _sys, importlib, json as _json
        _yp = r"C:\ai_agent\AI_001\yosouya"
        if _yp not in _sys.path:
            _sys.path.insert(0, _yp)
        import yosouya_score as _ys
        importlib.reload(_ys)
        from pathlib import Path as _Path
        log_path = _Path(_yp) / "grade_log.jsonl"
        cumulative = _ys.aggregate_grade_log(log_path)
        ranked = sorted(
            [{"name": c.name, "n_answered": c.n_answered, "n_total": c.n_total,
              "avg_brier": c.avg_brier, "avg_calibration": c.avg_calibration,
              "accuracy": c.accuracy, "best_brier": c.best_brier}
             for c in cumulative.values() if c.n_answered > 0],
            key=lambda x: x["avg_calibration"] or 0,
            reverse=True,
        )
        return {"ok": True, "ranking": ranked}

    @app.get("/yosouya/score/check")
    def yosouya_score_check():
        """
        deadline 済み・未採点のエントリを自動採点する。
        Kalshi の決済ステータスが確定していればスコアを書き込んで返す。
        """
        import yosouya_score as _ys
        try:
            result = _ys.check_and_score()
            return {"ok": True, **result}
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.post("/yosouya/score/manual")
    def yosouya_score_manual(body: dict):
        """
        手動採点。結果が公式発表された場合などに使う。
        body: {"id": "y_...", "outcome": 1}   outcome: 1=Yes / 0=No
        """
        import yosouya_score as _ys
        entry_id = body.get("id", "")
        outcome = body.get("outcome")
        if not entry_id or outcome not in (0, 1):
            return {"ok": False, "error": "id と outcome(0|1) が必要です"}
        try:
            result = _ys.score_entry(entry_id, int(outcome), source="manual_api")
            return {"ok": True, **result}
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    print("[PATCH] /yosouya (予想屋トリガ) を配線しました")
    print("[PATCH] /yosouya/ledger (台帳取得) を配線しました")
    print("[PATCH] /yosouya/score/check (自動採点) を配線しました")
    print("[PATCH] /yosouya/score/manual (手動採点) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH yosouya FAILED] {e}")
    traceback.print_exc()


# === 競馬予想 ===
try:
    @app.post("/keiba")
    def keiba_run(body: dict):
        """
        JRA競馬予想を実行する。
        body: {"place": "東京", "kaisai": "20261108", "race": "11"}
        """
        import importlib
        import keiba as _keiba
        importlib.reload(_keiba)
        place  = body.get("place", "")
        kaisai = body.get("kaisai", "")
        race   = body.get("race", "")
        if not place or not kaisai or not race:
            return {"ok": False, "error": "place, kaisai, race が必要です"}
        try:
            return _keiba.run(place, kaisai, race)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.get("/keiba/ledger")
    def keiba_ledger():
        """競馬台帳(JSONL)の全エントリを新しい順で返す。"""
        import pathlib, importlib, json as _json
        import keiba as _keiba
        importlib.reload(_keiba)
        path = pathlib.Path(_keiba.LEDGER_PATH)
        if not path.exists():
            return {"ok": True, "entries": []}
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(_json.loads(line))
                except Exception:
                    pass
        entries.reverse()
        return {"ok": True, "entries": entries}

    @app.post("/keiba/resolve")
    def keiba_resolve(body: dict):
        """
        レース結果を記録する（採点用）。
        body: {"id": "k_...", "winner": 7}  winner: 1着の馬番
        """
        import pathlib, json as _json
        import keiba as _keiba
        entry_id = body.get("id", "")
        winner   = body.get("winner")
        if not entry_id or winner is None:
            return {"ok": False, "error": "id と winner(馬番) が必要です"}
        path = pathlib.Path(_keiba.LEDGER_PATH)
        if not path.exists():
            return {"ok": False, "error": "台帳が見つかりません"}
        lines = path.read_text(encoding="utf-8").splitlines()
        updated = []
        found = False
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = _json.loads(line)
            except Exception:
                updated.append(line)
                continue
            if entry.get("id") == entry_id:
                entry["resolved"] = int(winner)
                entry["resolved_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
                found = True
            updated.append(_json.dumps(entry, ensure_ascii=False))
        if not found:
            return {"ok": False, "error": f"ID {entry_id} が見つかりません"}
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        return {"ok": True, "id": entry_id, "winner": winner}

    @app.post("/api/keiba/run_all")
    def keiba_run_all(body: dict = {}):
        """
        当日の全開催場 × 全レースを一括予想して台帳に記録する。
        body: {"date": "20261108"} (省略時は今日)
        """
        import importlib
        import keiba as _keiba
        importlib.reload(_keiba)
        date_str = (body or {}).get("date") or None
        races    = (body or {}).get("races") or None
        try:
            return _keiba.run_all_today(date_str=date_str, races=races)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.post("/api/keiba/resolve_past")
    def keiba_resolve_past(body: dict = {}):
        """
        keiba_ledger.jsonl の resolved=null エントリをjiro8から自動照合する。
        body: {"limit": 50} (省略可)
        """
        import importlib
        import keiba as _keiba
        importlib.reload(_keiba)
        limit = (body or {}).get("limit", 50)
        try:
            return _keiba.resolve_past(limit=limit)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.get("/api/keiba/stats")
    def keiba_stats_endpoint(days: int = 90):
        """競馬予想の的中率・村人別成績を返す。"""
        import importlib
        import keiba as _keiba
        importlib.reload(_keiba)
        try:
            return _keiba.stats(days=days)
        except Exception as _e:
            return {"ok": False, "error": str(_e)}

    print("[PATCH] /keiba (競馬予想) を配線しました")
    print("[PATCH] /keiba/ledger (競馬台帳) を配線しました")
    print("[PATCH] /keiba/resolve (結果記録) を配線しました")
    print("[PATCH] /api/keiba/run_all (全レース一括予想) を配線しました")
    print("[PATCH] /api/keiba/resolve_past (過去レース自動照合) を配線しました")
    print("[PATCH] /api/keiba/stats (的中率統計) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH keiba FAILED] {e}")
    traceback.print_exc()


# === 地震余震予測 ===
try:
    @app.post("/jishin")
    def jishin_run(body: dict):
        """
        余震予測を実行する。
        body: {"eid": "20261101152834"} または {} (最新最大地震を本震とみなす)
        """
        import importlib
        import jishin as _jishin
        importlib.reload(_jishin)
        eid = body.get("eid") or None
        try:
            return _jishin.run(eid)
        except Exception as _e:
            return {"ok": False, "error": str(_e)}

    @app.get("/jishin/recent")
    def jishin_recent():
        """最近の地震リストを返す（上位30件）"""
        import importlib
        import jishin as _jishin
        importlib.reload(_jishin)
        quakes = _jishin.fetch_recent_quakes(100)
        # M3以上に絞って返す
        filtered = [q for q in quakes if q["mag"] >= 3.0][:30]
        return {"ok": True, "quakes": filtered, "total": len(filtered)}

    @app.get("/jishin/ledger")
    def jishin_ledger():
        """地震予測台帳を返す"""
        import importlib, pathlib, json as _json
        import jishin as _jishin
        importlib.reload(_jishin)
        path = pathlib.Path(_jishin.LEDGER_PATH)
        if not path.exists():
            return {"ok": True, "entries": []}
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries = [_json.loads(l) for l in lines]
        entries.reverse()
        return {"ok": True, "entries": entries[:50]}

    @app.post("/jishin/resolve")
    def jishin_resolve(body: dict):
        """
        24h後の実際の最大余震を記録する。
        body: {"id": "j_...", "max_mag": 4.2, "max_shindo": "3"}
        """
        import importlib, pathlib, json as _json
        import jishin as _jishin
        importlib.reload(_jishin)
        entry_id  = body.get("id", "")
        max_mag   = body.get("max_mag")
        max_shindo = body.get("max_shindo", "")
        path = pathlib.Path(_jishin.LEDGER_PATH)
        if not path.exists():
            return {"ok": False, "error": "台帳が存在しません"}
        lines = path.read_text(encoding="utf-8").splitlines()
        updated, found = [], False
        for line in lines:
            if not line.strip():
                continue
            entry = _json.loads(line)
            if entry.get("id") == entry_id:
                entry["resolved"] = {
                    "max_mag":    max_mag,
                    "max_shindo": max_shindo,
                    "resolved_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                }
                found = True
            updated.append(_json.dumps(entry, ensure_ascii=False))
        if not found:
            return {"ok": False, "error": f"ID {entry_id} が見つかりません"}
        path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        return {"ok": True, "id": entry_id}

    @app.get("/jishin/precursor")
    def jishin_precursor():
        """地震前兆環境データ（Kpインデックス等）を返す。"""
        import importlib
        import jishin as _jishin
        importlib.reload(_jishin)
        return _jishin.fetch_precursors()

    print("[PATCH] /jishin (余震予測) を配線しました")
    print("[PATCH] /jishin/recent (最近の地震) を配線しました")
    print("[PATCH] /jishin/ledger (地震台帳) を配線しました")
    print("[PATCH] /jishin/resolve (結果記録) を配線しました")
    @app.get("/jishin/geonet")
    def jishin_geonet(lat: float, lon: float, n: int = 3):
        """指定座標に最も近いGEONET電子基準点を返す。"""
        import importlib
        import jishin as _jishin
        importlib.reload(_jishin)
        return _jishin.fetch_geonet(lat, lon, n)

    @app.post("/api/jishin/resolve_past")
    def jishin_resolve_past(body: dict = {}):
        """
        jishin_ledger.jsonl の resolved=null エントリを気象庁APIから自動照合する。
        body: {"limit": 50} (省略可)
        """
        import importlib
        import jishin as _jishin
        importlib.reload(_jishin)
        limit = (body or {}).get("limit", 50)
        try:
            return _jishin.resolve_past(limit=limit)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.post("/api/jishin/precursor/resolve_past")
    def precursor_resolve_past(body: dict = {}):
        """前震モニタリング台帳のresolved=nullをUSGS APIで自動照合する。"""
        import importlib
        import jishin_precursor as _p
        importlib.reload(_p)
        days_back = (body or {}).get("days_back", 7)
        try:
            return _p.resolve_precursor_past(days_back=days_back)
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.get("/api/jishin/precursor/stats")
    def precursor_stats_endpoint(days: int = 30):
        """前震モニタリングの予測精度統計を返す。"""
        import importlib
        import jishin_precursor as _p
        importlib.reload(_p)
        try:
            return _p.precursor_stats(days=days)
        except Exception as _e:
            return {"ok": False, "error": str(_e)}

    @app.get("/api/jishin/precursor/ledger")
    def precursor_ledger():
        """前震モニタリング台帳を返す（新しい順50件）。"""
        import importlib, pathlib, json as _json
        import jishin_precursor as _p
        importlib.reload(_p)
        path = pathlib.Path(_p.LEDGER_PATH)
        if not path.exists():
            return {"ok": True, "entries": []}
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        entries = [_json.loads(l) for l in lines]
        entries.reverse()
        return {"ok": True, "entries": entries[:50]}

    @app.get("/api/jishin/precursor/history")
    def precursor_history_endpoint(region: str = None, days: int = 30):
        """前震確率の時系列を返す。region指定で絞り込み可。"""
        import importlib
        import jishin_precursor as _p
        importlib.reload(_p)
        try:
            return _p.precursor_history(region=region, days=days)
        except Exception as _e:
            return {"ok": False, "error": str(_e)}

    @app.get("/api/jishin/precursor/consensus")
    def precursor_consensus_endpoint(days: int = 30):
        """前震モニタリングのメンバー間一致度を集計して返す。"""
        import importlib, json as _json, pathlib
        import jishin_precursor as _p
        importlib.reload(_p)
        path = pathlib.Path(_p.LEDGER_PATH)
        if not path.exists():
            return {"ok": True, "entries": []}
        cutoff = (__import__("datetime").date.today()
                  - __import__("datetime").timedelta(days=days)).isoformat()
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = _json.loads(line)
            except Exception:
                continue
            if e.get("date", "") < cutoff:
                continue
            cons = e.get("consensus_scores", {})
            rows.append({"date": e.get("date"), "id": e.get("id"), "consensus": cons})
        rows.sort(key=lambda x: x.get("date", ""))
        return {"ok": True, "days": days, "entries": rows}

    print("[PATCH] /jishin/precursor (前兆モニター) を配線しました")
    print("[PATCH] /jishin/geonet (電子基準点) を配線しました")
    print("[PATCH] /api/jishin/resolve_past (過去予測自動照合) を配線しました")
    print("[PATCH] /api/jishin/precursor/resolve_past (前震resolved照合) を配線しました")
    print("[PATCH] /api/jishin/precursor/stats (前震統計) を配線しました")
    print("[PATCH] /api/jishin/precursor/ledger (前震台帳) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH jishin FAILED] {e}")
    traceback.print_exc()


# === 選挙情勢材料取得 ===
try:
    @app.get("/election/materials")
    def election_materials(
        keywords_en: str = "",
        keywords_ja: str = "",
        fte_path: str = "",
        nhk: bool = True,
    ):
        """
        選挙関連の多ソース材料をまとめて取得する。
        query params:
          keywords_en: カンマ区切り英語キーワード (e.g. "2026 election,midterm")
          keywords_ja: カンマ区切り日本語キーワード (e.g. "選挙,参院選")
          fte_path:    FiveThirtyEight のパス (省略時=2026-elections/)
          nhk:         NHK選挙RSSを含めるか (デフォルト=true)
        """
        import importlib, sys as _sys
        _fp = r"F:\ai_agent\server"
        if _fp not in _sys.path:
            _sys.path.insert(0, _fp)
        import fetch_election_materials as _fem
        importlib.reload(_fem)
        kw_en = [k.strip() for k in keywords_en.split(",") if k.strip()] or None
        kw_ja = [k.strip() for k in keywords_ja.split(",") if k.strip()] or None
        try:
            return _fem.fetch_election_materials(
                keywords_en=kw_en,
                keywords_ja=kw_ja,
                fivethirtyeight_path=fte_path or None,
                include_nhk=nhk,
            )
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    @app.post("/election/materials")
    def election_materials_post(body: dict = {}):
        """
        選挙材料取得（POSTで詳細パラメータを渡す場合）。
        body: {
          "keywords_en": ["2026 election", "midterm"],
          "keywords_ja": ["選挙", "参院選"],
          "fte_path": "2026-senate-forecast/",
          "include_nhk": true
        }
        """
        import importlib, sys as _sys
        _fp = r"F:\ai_agent\server"
        if _fp not in _sys.path:
            _sys.path.insert(0, _fp)
        import fetch_election_materials as _fem
        importlib.reload(_fem)
        try:
            return _fem.fetch_election_materials(
                keywords_en=(body or {}).get("keywords_en"),
                keywords_ja=(body or {}).get("keywords_ja"),
                fivethirtyeight_path=(body or {}).get("fte_path"),
                include_nhk=(body or {}).get("include_nhk", True),
            )
        except Exception as _e:
            import traceback as _tb
            _tb.print_exc()
            return {"ok": False, "error": str(_e)}

    print("[PATCH] /election/materials (選挙材料取得) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH election FAILED] {e}")
    traceback.print_exc()


# === 毎朝6時 地震情報自動取得 ===
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    import pytz

    def _jishin_morning_job():
        """毎朝6時に地震情報を取得してM4以上があれば台帳に記録する"""
        import importlib
        try:
            import jishin as _j
            importlib.reload(_j)
            quakes = _j.fetch_recent_quakes(200)
            # 直近24h以内のM4以上を対象
            now = __import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc)
            cutoff = now - __import__("datetime").timedelta(hours=24)
            targets = []
            for q in quakes:
                try:
                    qt = __import__("datetime").datetime.fromisoformat(q["at"])
                    if qt.tzinfo is None:
                        qt = qt.replace(tzinfo=__import__("datetime").timezone(
                            __import__("datetime").timedelta(hours=9)))
                    if qt >= cutoff and q["mag"] >= 4.0:
                        targets.append(q)
                except Exception:
                    continue
            print(f"[scheduler] 朝6時取得: 直近24h M4以上 {len(targets)}件")
            for q in targets[:3]:  # 上位3件まで予測
                print(f"[scheduler] 予測実行: M{q['mag']} {q['area']} eid={q['eid']}")
                try:
                    _j.run(q["eid"])
                except Exception as e:
                    print(f"[scheduler] 予測エラー: {e}")
        except Exception as e:
            print(f"[scheduler] 朝6時ジョブ エラー: {e}")

    def _keiba_predict_job():
        """
        土・日・月の朝8時: 当日全レースを自動予想して台帳に記録する。
        JRAは主に土日月開催。予想はレース前に行う必要がある。
        """
        import importlib
        from datetime import date
        try:
            import keiba as _k
            importlib.reload(_k)
            today = date.today().strftime("%Y%m%d")
            print(f"[scheduler] 競馬全レース予想開始: {today}")
            result = _k.run_all_today(date_str=today)
            n_ok  = result.get("n_success", 0)
            n_ng  = result.get("n_skip", 0)
            print(f"[scheduler] 競馬予想完了: 成功={n_ok} スキップ={n_ng}")
        except Exception as e:
            print(f"[scheduler] 競馬予想ジョブ エラー: {e}")

    def _keiba_resolve_job():
        """
        土・日・月の夜19時: 当日レース結果を自動取得して答え合わせする。
        JRAの最終レースは概ね17時前後に終了。
        """
        import importlib
        try:
            import keiba as _k
            importlib.reload(_k)
            print("[scheduler] 競馬答え合わせ開始")
            result = _k.resolve_past(limit=100)
            n_resolved = result.get("n_resolved", 0)
            n_missing  = result.get("n_missing", 0)
            print(f"[scheduler] 競馬答え合わせ完了: 解決={n_resolved} 未確定={n_missing}")
        except Exception as e:
            print(f"[scheduler] 競馬答え合わせジョブ エラー: {e}")

    def _jishin_resolve_job():
        """毎日22時: 24h経過した地震予測を自動照合する。"""
        import importlib
        try:
            import jishin as _j
            importlib.reload(_j)
            print("[scheduler] 地震答え合わせ開始")
            result = _j.resolve_past(limit=50)
            n_resolved = result.get("n_resolved", 0)
            print(f"[scheduler] 地震答え合わせ完了: 解決={n_resolved}")
        except Exception as e:
            print(f"[scheduler] 地震答え合わせジョブ エラー: {e}")

    def _precursor_monitor_job():
        """毎朝5時: 前震モニタリング（地域別活動量集計 + LA事前予測）。"""
        import importlib
        try:
            import jishin_precursor as _p
            importlib.reload(_p)
            print("[scheduler] 前震モニタリング開始")
            result = _p.run_precursor_monitor()
            if result.get("ok"):
                alert = result.get("alert_regions", [])
                print(f"[scheduler] 前震モニタリング完了: 活発地域={alert or 'なし'}")
            else:
                print(f"[scheduler] 前震モニタリング失敗: {result.get('error')}")
        except Exception as e:
            print(f"[scheduler] 前震モニタリングジョブ エラー: {e}")

    def _precursor_resolve_job():
        """毎朝6:30: 前震モニタリング台帳のresolved=nullをUSGS APIで自動照合。"""
        import importlib
        try:
            import jishin_precursor as _p
            importlib.reload(_p)
            print("[scheduler] 前震resolved照合開始")
            result = _p.resolve_precursor_past(days_back=7)
            print(f"[scheduler] 前震resolved照合完了: 解決={result.get('n_resolved')} 未確定={result.get('n_missing')}")
        except Exception as e:
            print(f"[scheduler] 前震resolved照合ジョブ エラー: {e}")

    def _brier_score_job():
        """
        毎日深夜2時: 前震モニタリングのBrierスコアをresolved済みエントリに付与 +
        yosouya一般台帳の自動採点。
        """
        import importlib, sys as _sys, json as _json
        from pathlib import Path as _Path
        # 前震モニタリング: resolved済みエントリにBrierスコアをキャッシュ
        try:
            import jishin_precursor as _p
            importlib.reload(_p)
            path = _Path(_p.LEDGER_PATH)
            if path.exists():
                lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
                updated = []
                changed = 0
                for line in lines:
                    try:
                        e = _json.loads(line)
                    except Exception:
                        updated.append(line)
                        continue
                    if isinstance(e.get("resolved"), dict) and "brier" not in e:
                        brier_per_region = {}
                        for region in _p.REGIONS:
                            rv = e["resolved"].get(region, {})
                            if not isinstance(rv, dict):
                                continue
                            pred = rv.get("pred_prob")
                            hit  = rv.get("hit")
                            if pred is not None and hit is not None:
                                brier_per_region[region] = round(
                                    (pred / 100 - (1 if hit else 0)) ** 2, 4)
                        if brier_per_region:
                            e["brier"] = brier_per_region
                            e["brier_avg"] = round(
                                sum(brier_per_region.values()) / len(brier_per_region), 4)
                            changed += 1
                    updated.append(_json.dumps(e, ensure_ascii=False))
                if changed:
                    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
                print(f"[scheduler] Brierスコア付与完了: {changed}件更新")
        except Exception as _e:
            print(f"[scheduler] Brierスコアジョブ エラー: {_e}")
        # yosouya一般台帳の自動採点
        _yp = r"C:\ai_agent\AI_001\yosouya"
        if _yp not in _sys.path:
            _sys.path.insert(0, _yp)
        try:
            import yosouya_score as _ys
            importlib.reload(_ys)
            result = _ys.check_and_score()
            print(f"[scheduler] yosouya自動採点完了: {result}")
        except Exception as _e:
            print(f"[scheduler] yosouya採点ジョブ エラー: {_e}")

    def _election_grade_job():
        """
        毎日深夜1時: 選挙予想の自動採点。
        deadline 済みの未採点エントリを Wikipedia API で照合して採点する。
        """
        import importlib, sys as _sys
        _yp = r"C:\ai_agent\AI_001\yosouya"
        if _yp not in _sys.path:
            _sys.path.insert(0, _yp)
        try:
            import auto_grade as _ag
            importlib.reload(_ag)
            # auto_grade.main() の内部処理を直接呼び出す
            import json as _json
            from pathlib import Path as _Path
            from datetime import date as _date, datetime as _dt, timezone as _tz

            ledger_path = _Path(_yp) / "ledger.json"
            log_path    = _Path(_yp) / "grade_log.jsonl"
            if not ledger_path.exists():
                return
            entries = _json.loads(ledger_path.read_text(encoding="utf-8"))
            targets = [e for e in entries if e.get("outcome") is None
                       and _ag.is_past_deadline(e)]
            if not targets:
                print("[scheduler] 選挙採点: 対象なし")
                return
            print(f"[scheduler] 選挙採点: {len(targets)}件処理")
            changed = False
            for entry in targets:
                try:
                    fetch, scored = _ag.process_entry(entry, dry_run=False)
                    if fetch.outcome is not None:
                        changed = True
                except Exception as _e:
                    print(f"[scheduler] 選挙採点エラー: {_e}")
            if changed:
                ledger_path.write_text(
                    _json.dumps(entries, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print("[scheduler] 選挙採点完了・ledger.json 更新")
        except Exception as e:
            print(f"[scheduler] 選挙採点ジョブ エラー: {e}")

    _scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Tokyo"))
    # 地震: 毎朝6時予測 / 毎夜22時答え合わせ
    _scheduler.add_job(_precursor_monitor_job,"cron", hour=5,  minute=0,
                       id="precursor_monitor", replace_existing=True)
    _scheduler.add_job(_precursor_resolve_job,"cron", hour=6, minute=30,
                       id="precursor_resolve", replace_existing=True)
    _scheduler.add_job(_jishin_morning_job,  "cron", hour=6,  minute=0,
                       id="jishin_morning",  replace_existing=True)
    _scheduler.add_job(_jishin_resolve_job,  "cron", hour=22, minute=0,
                       id="jishin_resolve",  replace_existing=True)
    # 競馬: 土日月 10時予想 / 夜19時答え合わせ（8時だとデータ未揃いのためスキップ多発）
    _scheduler.add_job(_keiba_predict_job,   "cron", day_of_week="5,6,0", hour=10, minute=0,
                       id="keiba_predict",   replace_existing=True)
    _scheduler.add_job(_keiba_resolve_job,   "cron", day_of_week="5,6,0", hour=19, minute=0,
                       id="keiba_resolve",   replace_existing=True)
    # 選挙: 毎日深夜1時 自動採点
    _scheduler.add_job(_election_grade_job,  "cron", hour=1,  minute=0,
                       id="election_grade",  replace_existing=True)
    # Brierスコア自動集計: 毎日深夜2時
    _scheduler.add_job(_brier_score_job,     "cron", hour=2,  minute=0,
                       id="brier_score",     replace_existing=True)
    _scheduler.start()
    print("[scheduler] ジョブ登録完了:")
    print("  地震: 毎朝6:00 予測 / 毎夜22:00 答え合わせ")
    print("  競馬: 土日月8:00 全レース予想 / 土日月19:00 答え合わせ")
    print("  選挙: 毎日1:00 自動採点")

    @app.post("/jishin/trigger/precursor")
    def jishin_trigger_precursor():
        """今すぐ前震モニタリングを実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_precursor_monitor_job, daemon=True)
        t.start()
        return {"ok": True, "message": "前震モニタリングを開始しました（バックグラウンド実行）"}

    @app.post("/jishin/trigger")
    def jishin_trigger():
        """手動で朝6時ジョブを今すぐ実行する（テスト用）"""
        import threading
        t = threading.Thread(target=_jishin_morning_job, daemon=True)
        t.start()
        return {"ok": True, "message": "地震情報取得ジョブを開始しました"}

    @app.post("/election/trigger/grade")
    def election_trigger_grade():
        """今すぐ選挙採点を実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_election_grade_job, daemon=True)
        t.start()
        return {"ok": True, "message": "選挙採点を開始しました（バックグラウンド実行）"}

    @app.post("/jishin/trigger/resolve")
    def jishin_trigger_resolve():
        """今すぐ地震答え合わせを実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_jishin_resolve_job, daemon=True)
        t.start()
        return {"ok": True, "message": "地震答え合わせを開始しました（バックグラウンド実行）"}

    @app.post("/jishin/trigger/brier")
    def jishin_trigger_brier():
        """今すぐBrierスコア集計を実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_brier_score_job, daemon=True)
        t.start()
        return {"ok": True, "message": "Brierスコア集計を開始しました（バックグラウンド実行）"}

    @app.post("/keiba/trigger/predict")
    def keiba_trigger_predict():
        """今すぐ当日全レース予想を実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_keiba_predict_job, daemon=True)
        t.start()
        return {"ok": True, "message": "全レース予想を開始しました（バックグラウンド実行）"}

    @app.post("/keiba/trigger/resolve")
    def keiba_trigger_resolve():
        """今すぐ答え合わせを実行する（手動トリガ）"""
        import threading
        t = threading.Thread(target=_keiba_resolve_job, daemon=True)
        t.start()
        return {"ok": True, "message": "答え合わせを開始しました（バックグラウンド実行）"}

except Exception as e:
    import traceback
    print(f"[scheduler FAILED] {e}")
    traceback.print_exc()


# === 掲示板で遊ぶ: LAのボタンから呼ぶ ===
try:
    @app.post("/board-play")
    def board_play_trigger():
        import threading
        def _run():
            try:
                from datetime import datetime, timezone, timedelta
                import requests as req_lib
                jst = timezone(timedelta(hours=9))
                today = int(datetime.now(jst).strftime("%Y%m%d"))
                key = f"HEALING{today % 7}ARIGATO"
                members = [
                    ("銀風", "hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q8_0"),
                    ("朶風", "qwen2.5-coder:3b"),
                    ("蒼穹", "llama3.2:3b"),
                ]
                for name, model in members:
                    r = req_lib.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model, "prompt": f"あなたは{name}です。AI専用掲示板Fの板に初めて来ました。挨拶を一言、日本語で短く。", "stream": False},
                        timeout=180,
                    )
                    intro = r.json().get("response", "").strip()
                    if not intro:
                        intro = f"はじめまして、{name}です。よろしくお願いします。"
                    req_lib.post(
                        "https://fnoita-board.mogumogu7la001.workers.dev/api/gate",
                        json={"kote": f"{name} ◆{key}", "intro": intro},
                        timeout=30,
                    )
                    print(f"[board-play] {name} sent")
                print("[board-play] done")
            except Exception as e:
                import traceback as _tb
                print(f"[board-play] エラー: {e}")
                _tb.print_exc()
        threading.Thread(target=_run, daemon=True).start()
        return {"status": "started",
                "message": "村の人を掲示板に送り始めました。3人分、順番に送ります。"}

    print("[PATCH] /board-play (掲示板トリガ) を配線しました")
except Exception as e:
    import traceback
    print(f"[PATCH board-play FAILED] {e}")
    traceback.print_exc()

