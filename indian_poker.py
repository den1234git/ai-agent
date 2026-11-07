"""
indian_poker.py — インディアンポーカー言語版エンジン

各プレイヤーに秘密の「文言（コードネーム）」を割り当て、
自分の文言は見えないが他全員の文言は見える状態で、
はい/いいえ質問と推測を繰り返して最初に正解した人が勝利。

設計:
  - プールから5人ランダム選抜（プール不足時は全員）
  - 文言リスト20語からランダム配布（重複なし）
  - 1ターン: 質問 → 次の席の人が正直に回答 → 推測（任意）
  - 最大ターン数: 各自6手番 or 全体30手番で引き分け
  - 崩れた出力は1回再試行 → だめならスキップ（棄権なし）
  - active のまま前のゲームが残っていても /indian/cancel で確実にリセット可能
"""

import json
import os
import random
import unicodedata
from datetime import datetime, date

from ai import ask_ai, OllamaUnavailableError
from output_validator import is_valid_japanese_output

_BASE = os.path.dirname(os.path.abspath(__file__))
_STATE_FILE = os.path.join(_BASE, "indian_poker_state.json")

# ─── 文言リスト ────────────────────────────────────────────────
# 2〜4文字の具体的な日本語名詞（カテゴリが多様になるよう選定）
WORD_LIST = [
    "宇宙", "魔法", "嵐", "虹", "猫", "海", "桜", "月", "雷", "炎",
    "森", "夢", "鳥", "花", "星", "風", "山", "剣", "城", "龍",
]

# ─── ゲーム設定 ────────────────────────────────────────────────
PLAYERS_PER_GAME    = 5   # 選抜人数（プールが少なければ全員）
MAX_TURNS_PER_PLAYER = 6  # 各プレイヤーの最大手番数
MAX_TOTAL_TURNS      = 30 # 全体最大ターン数（引き分け判定）
HISTORY_WINDOW       = 5  # プロンプトに渡す直近履歴件数

# ─── 部屋プール ────────────────────────────────────────────────
# qwen2_5 は models.json 未登録なので除外。llama3_1(8B) を採用。
IP_POOL_DEFAULT = [
    "swallow",  # 8B:  日本語指示追従
    "llama3_2", # 3B:  小型だが役になりきれる
    "gemma3",   # 4B:  日本語対応
    "llama3_1", # 8B:  安定した指示追従
    # qwen3(14B)はタイムアウト多発のため除外（UIから手動追加は可能）
]
IP_POOL_MAX = 15

_room: dict = {"pool": list(IP_POOL_DEFAULT)}


def get_room() -> dict:
    return {"pool": list(_room["pool"]), "count": len(_room["pool"]), "max": IP_POOL_MAX}


def join_room(model_key: str) -> dict:
    if model_key in _room["pool"]:
        return {"status": "already_in", "pool": list(_room["pool"])}
    if len(_room["pool"]) >= IP_POOL_MAX:
        return {"status": "full", "message": f"部屋が満員です（最大{IP_POOL_MAX}人）", "pool": list(_room["pool"])}
    _room["pool"].append(model_key)
    return {"status": "joined", "model": model_key, "pool": list(_room["pool"])}


def leave_room(model_key: str) -> dict:
    if model_key not in _room["pool"]:
        return {"status": "not_in", "pool": list(_room["pool"])}
    _room["pool"].remove(model_key)
    return {"status": "left", "model": model_key, "pool": list(_room["pool"])}


def reset_room() -> dict:
    _room["pool"] = list(IP_POOL_DEFAULT)
    return {"status": "reset", "pool": list(_room["pool"])}


# ─── models.json からモデル表示名を引く ─────────────────────────
def _get_model_name(key: str) -> str:
    config_path = os.path.join(_BASE, "models.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["models"].get(key, key)
    except Exception:
        return key


# ─── 状態管理 ────────────────────────────────────────────────
_state: dict = {}


def _default_state() -> dict:
    return {
        "active": False,
        "phase": "done",          # "playing" | "done"
        "sub_phase": "question",  # "question" | "answer" | "guess"
        "total_turn": 0,
        "current_player_idx": 0,
        "current_turn_data": {},  # 進行中ターンの一時データ
        "players": [],            # [{key, name, word, turn_count}]
        "history": [],            # 全ターンの記録（観戦・ログ用）
        "result": None,
    }


def _save_state():
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[INDIAN] 状態保存失敗（無視）: {e}")


def _load_state():
    global _state
    if not os.path.exists(_STATE_FILE):
        _state = _default_state()
        return
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "phase" in data:
            _state = data
            if _state.get("active"):
                print(f"[INDIAN] 前回の状態を復元（turn={_state['total_turn']} sub={_state['sub_phase']}）")
        else:
            _state = _default_state()
    except Exception:
        _state = _default_state()


# ─── 正規化（文言一致判定） ──────────────────────────────────
def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    return t.strip().replace(" ", "").replace("　", "").lower()


# ─── ゲーム開始 ─────────────────────────────────────────────
def start_game() -> dict:
    global _state
    if not _state:
        _load_state()

    # 詰まり対策: active でも start を呼ばれたら明確に返す
    if _state.get("active"):
        return {
            "status": "error",
            "message": "ゲームが既に進行中です。/indian/cancel で中断してから開始してください",
        }

    pool = _room["pool"]
    if len(pool) < 2:
        return {"status": "error", "message": "プールにモデルが2人以上必要です"}

    n = min(PLAYERS_PER_GAME, len(pool))
    selected_keys = random.sample(pool, n)
    words = random.sample(WORD_LIST, n)

    players = [
        {"key": k, "name": _get_model_name(k), "word": w, "turn_count": 0, "skip_next_turn": False}
        for k, w in zip(selected_keys, words)
    ]

    _state = {
        "active": True,
        "phase": "playing",
        "sub_phase": "question",
        "total_turn": 0,
        "current_player_idx": 0,
        "current_turn_data": {},
        "players": players,
        "history": [],
        "result": None,
    }
    _save_state()

    player_summary = [f"{p['name']}={p['word']}" for p in players]
    print(f"[INDIAN] ゲーム開始: {player_summary}")

    return {
        "status": "started",
        "players": [{"key": p["key"], "name": p["name"], "word": p["word"]} for p in players],
        "max_turns_per_player": MAX_TURNS_PER_PLAYER,
        "max_total_turns": MAX_TOTAL_TURNS,
        "message": f"{n}人でゲーム開始！先に自分の文言を当てた人の勝ち",
    }


# ─── ゲーム中断 ─────────────────────────────────────────────
def cancel_game() -> dict:
    global _state
    if not _state:
        _load_state()
    _state["active"] = False
    _state["phase"] = "done"
    _state["current_turn_data"] = {}
    _save_state()
    return {"status": "cancelled", "message": "ゲームをキャンセルしました"}


# ─── 現在状態取得 ─────────────────────────────────────────────
def get_state() -> dict:
    if not _state:
        _load_state()
    return {
        "active": _state.get("active", False),
        "phase": _state.get("phase", "done"),
        "sub_phase": _state.get("sub_phase", "question"),
        "total_turn": _state.get("total_turn", 0),
        "current_player_idx": _state.get("current_player_idx", 0),
        "players": [
            {"key": p["key"], "name": p["name"], "word": p["word"], "turn_count": p["turn_count"]}
            for p in _state.get("players", [])
        ],
        "result": _state.get("result"),
        "history_count": len(_state.get("history", [])),
        "max_turns_per_player": MAX_TURNS_PER_PLAYER,
        "max_total_turns": MAX_TOTAL_TURNS,
    }


# ─── プロンプト補助 ───────────────────────────────────────────
def _others_view_text(player_idx: int) -> str:
    """指定プレイヤーから見た他全員の文言一覧テキスト（自分の文言は含まない）"""
    lines = []
    for i, p in enumerate(_state["players"]):
        if i != player_idx:
            lines.append(f"  ・{p['name']}: 「{p['word']}」")
    return "\n".join(lines)


def _history_context_text(recent_n: int = HISTORY_WINDOW) -> str:
    """直近N件の質問・回答・推測をテキスト化"""
    hist = _state["history"]
    recent = hist[-recent_n:] if len(hist) > recent_n else hist
    if not recent:
        return "（まだ質問は行われていません）"
    lines = []
    for h in recent:
        q = h.get("question", "?")
        a = h.get("answer", "?")
        r = h.get("reason", "")
        who = h.get("player_name", "?")
        asker = h.get("answerer_name", "?")
        line = f"[T{h['turn']}] {who}の質問「{q}」→ {asker}「{a}」（{r}）"
        if h.get("guessed"):
            mark = "✓正解" if h.get("correct") else "✗外れ"
            line += f" → {who}推測「{h.get('guess', '')}」{mark}"
        lines.append(line)
    return "\n".join(lines)


# ─── AI 呼び出し ─────────────────────────────────────────────
def _call_ai_safe(model_key: str, prompt: str) -> dict:
    try:
        return ask_ai(prompt, model=model_key)
    except OllamaUnavailableError as e:
        print(f"[INDIAN] {model_key} 呼び出し失敗: {e}")
        return {}
    except Exception as e:
        print(f"[INDIAN] {model_key} 予期しないエラー: {e}")
        return {}


def _gen_question(player_idx: int) -> str | None:
    """質問フェーズ: 現在プレイヤーに質問を生成させる（ゲーム文脈なし・シンプルプロンプト）"""
    p = _state["players"][player_idx]

    # 他プレイヤーの文言一覧（自分のではないことを明示）
    other_words = [
        _state["players"][i]["word"]
        for i in range(len(_state["players"]))
        if i != player_idx
    ]
    others_str = "、".join(f"「{w}」" for w in other_words)

    # 直近の質問キーワードを抽出して重複を避ける
    hist = _state["history"]
    recent_questions = [h.get("question", "") for h in hist[-HISTORY_WINDOW:]]
    avoid_hint = ""
    if recent_questions:
        avoid_hint = f"\n\n既に聞いた質問（重複を避けてください）: {'、'.join(q for q in recent_questions if q)}"

    prompt = (
        "あなたには秘密の言葉（2〜4文字の日本語名詞）があります。自分の言葉は見えません。\n"
        f"他の人の言葉（あなたの言葉はこれ以外）: {others_str}\n\n"
        "自分の言葉が何かを当てるために、「はい/いいえ」で答えられる質問を1つ作ってください。\n"
        "良い質問の例：「生き物ですか？」「自然に関係しますか？」「食べ物ですか？」「目に見えますか？」"
        f"{avoid_hint}\n\n"
        "質問文だけを30文字以内・日本語で出力してください。\n\n"
        "以下のJSONのみ出力してください:\n"
        '{"question": "質問文"}'
    )

    for attempt in range(2):
        result = _call_ai_safe(p["key"], prompt)
        q = result.get("question", "").strip()
        if not q:
            continue
        if len(q) > 40:
            q = q[:40]
        valid, reason = is_valid_japanese_output(q)
        if valid:
            return q
        print(f"[INDIAN] {p['name']} 質問崩れ（attempt={attempt+1}）: {reason} → 「{q[:20]}…」")

    return None


def _gen_answer(answerer_idx: int, questioner_word: str, question: str) -> tuple[str, str, bool]:
    """1人の回答者に yes/no＋理由を生成させる。(answer, reason, is_fallback)を返す。
    is_fallback=True は崩れ出力によるフォールバック（ペナルティ対象外）。"""
    answerer = _state["players"][answerer_idx]

    prompt = (
        f"「{questioner_word}」という言葉について、次の質問に正直に答えてください。\n\n"
        f"質問: 「{question}」\n\n"
        f"「{questioner_word}」が上の条件に当てはまるかどうか落ち着いて考えて、"
        "「はい」か「いいえ」で答えてください。理由を30文字以内で添えてください。\n\n"
        "以下のJSONのみ出力してください:\n"
        '{"answer": "はい" または "いいえ", "reason": "理由（30文字以内）"}'
    )

    for attempt in range(2):
        result = _call_ai_safe(answerer["key"], prompt)
        raw_ans = result.get("answer", "").strip()
        reason = result.get("reason", "").strip()

        ans = _normalize_answer(raw_ans)
        if ans is not None:
            if len(reason) > 40:
                reason = reason[:40]
            if not reason:
                reason = "（理由なし）"
            return ans, reason, False  # 正常回答
        print(f"[INDIAN] {answerer['name']} 回答崩れ（attempt={attempt+1}）: raw={raw_ans!r}")

    print(f"[INDIAN] {answerer['name']} 回答生成失敗 → フォールバック")
    return "いいえ", "回答生成失敗（フォールバック）", True  # 崩れによるフォールバック


def _gen_answer_majority(
    questioner_idx: int, questioner_word: str, question: str
) -> tuple[str, str, list, list]:
    """質問者以外の最大3人に答えさせて多数決。(answer, reason, votes, penalty_idxs) を返す。
    同票は「いいえ」をデフォルト。同票時はペナルティなし。"""
    players = _state["players"]
    n = len(players)
    candidate_idxs = [i for i in range(n) if i != questioner_idx]

    if len(candidate_idxs) > 3:
        voter_idxs = random.sample(candidate_idxs, 3)
    else:
        voter_idxs = candidate_idxs

    yes_count = 0
    no_count = 0
    votes = []
    vote_details = []  # (player_idx, answer, is_fallback)

    for idx in voter_idxs:
        ans, _, is_fallback = _gen_answer(idx, questioner_word, question)
        if ans == "はい":
            yes_count += 1
        else:
            no_count += 1
        votes.append({"name": players[idx]["name"], "answer": ans})
        vote_details.append((idx, ans, is_fallback))

    is_tie = (yes_count == no_count)
    final = "はい" if yes_count > no_count else "いいえ"
    reason = f"多数決 はい{yes_count}/いいえ{no_count}"

    # ペナルティ対象: 多数決が割れた（同票でない）かつ少数派かつ崩れ出力でない
    penalty_idxs = []
    if not is_tie:
        for idx, ans, is_fallback in vote_details:
            if ans != final and not is_fallback:
                penalty_idxs.append(idx)

    return final, reason, votes, penalty_idxs


def _normalize_answer(raw: str) -> str | None:
    """'はい'/'いいえ' に正規化。判定できない場合は None"""
    if not raw:
        return None
    low = raw.lower()
    if "はい" in low or "yes" in low or "そうです" in low or "correct" in low or "true" in low:
        return "はい"
    if "いいえ" in low or "no" in low or "違い" in low or "ちがい" in low or "false" in low:
        return "いいえ"
    # 先頭1文字で判定
    if raw[0] in ("は", "ハ", "Y", "y"):
        return "はい"
    if raw[0] in ("い", "イ", "N", "n"):
        return "いいえ"
    return None


def _gen_guess(player_idx: int, question: str, answer: str, reason: str) -> tuple[bool, str | None]:
    """推測フェーズ: 現在プレイヤーに推測させる。(skip, guess_word)"""
    p = _state["players"][player_idx]

    # 他プレイヤーの文言（除外候補として）
    other_words = [
        _state["players"][i]["word"]
        for i in range(len(_state["players"]))
        if i != player_idx
    ]
    others_str = "、".join(f"「{w}」" for w in other_words)

    # 直近の質問・回答履歴（シンプルに）
    hist = _state["history"]
    recent = hist[-HISTORY_WINDOW:] if len(hist) > HISTORY_WINDOW else hist
    hist_lines = []
    for h in recent:
        hist_lines.append(f"Q: 「{h.get('question', '?')}」→ {h.get('answer', '?')}")
    hist_text = "\n".join(hist_lines) if hist_lines else "（なし）"

    prompt = (
        "あなたには秘密の言葉（2〜4文字の日本語名詞）があります。自分の言葉は見えません。\n"
        f"他の人の言葉（あなたの言葉はこれ以外）: {others_str}\n\n"
        f"【これまでの質問・回答】\n{hist_text}\n\n"
        f"今回の質問: 「{question}」→ 「{answer}」\n\n"
        "これらの手がかりから自分の言葉を推測できますか？\n"
        "- 確信があれば: {\"skip\": false, \"guess\": \"推測する言葉\"}\n"
        "- まだ分からなければ: {\"skip\": true}\n\n"
        "以下のJSONのみ出力してください:\n"
        '{"skip": true または false, "guess": "推測する言葉（skipがfalseのとき）"}'
    )

    result = _call_ai_safe(p["key"], prompt)
    skip = result.get("skip", True)
    if skip:
        return True, None

    guess = result.get("guess", "").strip()
    if not guess:
        return True, None

    # 崩れチェック（推測は短い単語なので最低文字数チェックのみ簡易対応）
    if len(guess) > 10:
        guess = guess[:6]

    return False, guess


# ─── 1アクション進める ───────────────────────────────────────
def next_action() -> dict:
    global _state
    if not _state:
        _load_state()

    if not _state.get("active"):
        return {"status": "error", "message": "ゲームが開始されていません"}
    if _state["phase"] != "playing":
        return {"status": "done", "result": _state.get("result")}

    players = _state["players"]
    n = len(players)
    cur_idx = _state["current_player_idx"]
    sub = _state["sub_phase"]

    # ── 質問フェーズ ─────────────────────────────────────────
    if sub == "question":
        # 全員手番上限チェック
        if all(p["turn_count"] >= MAX_TURNS_PER_PLAYER for p in players):
            return _finish_game("draw")

        # 全体ターン上限
        next_turn = _state["total_turn"] + 1
        if next_turn > MAX_TOTAL_TURNS:
            return _finish_game("draw")

        cur = players[cur_idx]
        # このプレイヤーが上限 → スキップして次へ
        if cur["turn_count"] >= MAX_TURNS_PER_PLAYER:
            _advance_player()
            _save_state()
            return {
                "status": "ok",
                "action_type": "turn_limit_skip",
                "player": cur["name"],
                "message": f"{cur['name']}は手番上限（{MAX_TURNS_PER_PLAYER}）に達したのでスキップ",
            }

        # ペナルティスキップ（前回の多数決で少数派だった）
        if cur.get("skip_next_turn", False):
            cur["skip_next_turn"] = False
            _advance_player()
            _save_state()
            return {
                "status": "ok",
                "action_type": "penalty_skip",
                "player": cur["name"],
                "player_key": cur["key"],
                "message": f"{cur['name']}は一回休み（前回の多数決で少数派の答えを出した）",
            }

        # 質問生成（内部で最大2回試行）
        question = _gen_question(cur_idx)
        if question is None:
            print(f"[INDIAN] {cur['name']} 質問2回失敗 → スキップ")
            _advance_player()
            _save_state()
            return {
                "status": "ok",
                "action_type": "question_skip",
                "player": cur["name"],
                "player_key": cur["key"],
                "message": f"{cur['name']}の質問を生成できませんでした（スキップ）",
            }

        answerer_idx = (cur_idx + 1) % n
        _state["total_turn"] = next_turn
        _state["current_turn_data"] = {
            "turn": next_turn,
            "player_idx": cur_idx,
            "player_key": cur["key"],
            "player_name": cur["name"],
            "player_word": cur["word"],  # ログ用（プレイヤーには渡さない）
            "question": question,
            "answerer_idx": answerer_idx,
        }
        _state["sub_phase"] = "answer"
        _save_state()

        return {
            "status": "ok",
            "action_type": "question",
            "turn": next_turn,
            "player": cur["name"],
            "player_key": cur["key"],
            "question": question,
            "message": f"[T{next_turn}] {cur['name']}: 「{question}」",
        }

    # ── 回答フェーズ（多数決） ────────────────────────────────
    elif sub == "answer":
        td = _state["current_turn_data"]
        questioner_idx = td["player_idx"]

        ans, reason, votes, penalty_idxs = _gen_answer_majority(
            questioner_idx, td["player_word"], td["question"]
        )

        # 主席回答者（表示用）: 質問者の次のプレイヤー
        answerer_idx = (questioner_idx + 1) % len(players)
        answerer = players[answerer_idx]

        # ペナルティフラグを立てる（少数派回答者に）
        penalized_names = []
        for pidx in penalty_idxs:
            players[pidx]["skip_next_turn"] = True
            penalized_names.append(players[pidx]["name"])
            print(f"[INDIAN] {players[pidx]['name']} にペナルティ（少数派）: skip_next_turn=True")

        td["answer"] = ans
        td["reason"] = reason
        td["answerer_key"] = answerer["key"]
        td["answerer_name"] = answerer["name"]
        td["votes"] = votes
        td["penalized"] = penalized_names
        _state["sub_phase"] = "guess"
        _save_state()

        return {
            "status": "ok",
            "action_type": "answer",
            "turn": td["turn"],
            "answerer": answerer["name"],
            "answerer_key": answerer["key"],
            "questioner": td["player_name"],
            "answer": ans,
            "reason": reason,
            "votes": votes,
            "penalized": penalized_names,
            "message": f"[T{td['turn']}] 多数決: 「{ans}」（{reason}）",
        }

    # ── 推測フェーズ ─────────────────────────────────────────
    elif sub == "guess":
        td = _state["current_turn_data"]
        cur = players[td["player_idx"]]

        skip, guess = _gen_guess(
            td["player_idx"],
            td["question"],
            td["answer"],
            td["reason"],
        )

        correct = False
        if not skip and guess:
            correct = _normalize(guess) == _normalize(cur["word"])

        # 履歴に記録
        _state["history"].append({
            "turn": td["turn"],
            "player_key": td["player_key"],
            "player_name": td["player_name"],
            "player_word": td["player_word"],
            "question": td["question"],
            "answerer_key": td.get("answerer_key", ""),
            "answerer_name": td.get("answerer_name", ""),
            "answer": td.get("answer", ""),
            "reason": td.get("reason", ""),
            "votes": td.get("votes", []),
            "penalized": td.get("penalized", []),
            "guessed": not skip,
            "guess": guess or "",
            "correct": correct,
        })
        cur["turn_count"] += 1

        if correct:
            return _finish_game("win", winner_idx=td["player_idx"], guess=guess)

        # ターン終了 → 次プレイヤーへ
        _state["current_turn_data"] = {}
        _advance_player()
        _save_state()

        if skip:
            msg = f"[T{td['turn']}] {cur['name']}: 推測なし（パス）"
        else:
            msg = f"[T{td['turn']}] {cur['name']}: 推測「{guess}」→ 外れ（正解: {cur['word']}）"

        return {
            "status": "ok",
            "action_type": "guess",
            "turn": td["turn"],
            "player": cur["name"],
            "player_key": cur["key"],
            "skipped": skip,
            "guess": guess or "",
            "correct": False,
            "message": msg,
        }

    return {"status": "error", "message": f"不明な sub_phase: {sub}"}


def _advance_player():
    n = len(_state["players"])
    _state["current_player_idx"] = (_state["current_player_idx"] + 1) % n
    _state["sub_phase"] = "question"


def _finish_game(result_type: str, winner_idx: int = None, guess: str = None) -> dict:
    global _state
    _state["active"] = False
    _state["phase"] = "done"
    _state["sub_phase"] = "question"
    _state["current_turn_data"] = {}

    today = date.today().isoformat()
    log_file = os.path.join(_BASE, f"indian_log_{today}.jsonl")

    if result_type == "win" and winner_idx is not None:
        winner = _state["players"][winner_idx]
        _state["result"] = {
            "type": "win",
            "winner_key": winner["key"],
            "winner_name": winner["name"],
            "word": winner["word"],
            "guess": guess,
            "total_turns": _state["total_turn"],
        }
        _save_log(log_file)
        _save_state()
        return {
            "status": "done",
            "result": "win",
            "winner": winner["name"],
            "winner_key": winner["key"],
            "word": winner["word"],
            "total_turns": _state["total_turn"],
            "message": f"🎉 {winner['name']}が「{guess}」と正解！（正解: {winner['word']}）勝利！",
        }
    else:
        _state["result"] = {
            "type": "draw",
            "total_turns": _state["total_turn"],
        }
        _save_log(log_file)
        _save_state()
        return {
            "status": "done",
            "result": "draw",
            "total_turns": _state["total_turn"],
            "message": f"引き分け！全{_state['total_turn']}ターン消化。誰も自分の文言を当てられませんでした。",
        }


def _save_log(log_file: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "players": [{"key": p["key"], "name": p["name"], "word": p["word"]} for p in _state["players"]],
        "history": _state["history"],
        "result": _state["result"],
        "total_turns": _state["total_turn"],
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INDIAN] ログ保存: {log_file}")
    except Exception as e:
        print(f"[INDIAN] ログ保存失敗（無視）: {e}")


# モジュール読み込み時に状態復元
_load_state()
