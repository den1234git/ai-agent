"""
werewolf.py — AI人狼ゲームエンジン

AI同士（村のモデル）がテキストだけで遊ぶ軽い人狼。
ルールはコードが仕切り、モデルは発言と投票だけ。

プレイヤー: router / llama3_2 / swallow / granite / lfm_jp（5人）
役職: 村人 / 人狼のみ（最小構成。占い師・騎士等は将来拡張予定）
進行: 昼議論 → 投票 → 処刑 → 夜襲撃 のラウンドを繰り返す

勝利条件:
  村人勝利: 人狼を全員処刑
  人狼勝利: 生存人狼数 >= 生存村人数

コンテキスト肥大対策:
  各モデルに渡す履歴は「今ラウンド昼発言・最大4件」+
  「前ラウンドまでのイベントサマリー（最大2ラウンド分）」のみ。
  全履歴は渡さない。
"""

import json
import os
import random
import time
from datetime import datetime, date

from ai import ask_ai, OllamaUnavailableError
from output_validator import is_valid_japanese_output

# 議論発言が崩れた場合のフォールバック文
_SPEECH_FALLBACK = "…（うまく話せなかった）"

_BASE = os.path.dirname(os.path.abspath(__file__))
_STATE_FILE = os.path.join(_BASE, "werewolf_state.json")
_KOTE_FILE  = os.path.join(_BASE, "village_kote.json")

PLAYERS = ["router", "llama3_2", "swallow", "granite", "lfm_jp"]
WEREWOLF_COUNT = 1  # 5人中1人が人狼（5人構成の標準）

# 板投稿先カテゴリ（ゲーム広場）
BOARD_CAT_ID = "f11"

_state: dict = {}


# ─────────────────────────────────────────────────
# 状態の永続化
# ─────────────────────────────────────────────────

def _save_state():
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WEREWOLF] 状態保存失敗（無視）: {e}")


def _load_state():
    global _state
    if not os.path.exists(_STATE_FILE):
        _state = _default_state()
        return
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "phase" in data and "players" in data:
            _state = data
            if _state.get("active"):
                print(f"[WEREWOLF] 前回のゲーム状態を復元しました（R{_state['round']} {_state['phase']}）")
        else:
            _state = _default_state()
    except Exception:
        _state = _default_state()


def _default_state() -> dict:
    return {
        "active": False,
        "round": 0,
        "phase": "done",
        "sub_index": 0,
        "players": {},
        "player_order": [],
        "day_transcript": [],
        "votes": {},
        "event_log": [],
        "result": None,
        "log": [],
    }


def _reset():
    global _state
    _state = _default_state()
    try:
        if os.path.exists(_STATE_FILE):
            os.remove(_STATE_FILE)
    except Exception:
        pass


# ─────────────────────────────────────────────────
# コテハン（village_kote.json から共有）
# ─────────────────────────────────────────────────

def _load_kote() -> dict:
    try:
        with open(_KOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────
# ゲーム進行補助
# ─────────────────────────────────────────────────

def _get_alive_players() -> list:
    """生存プレイヤーをゲーム開始時にシャッフルした発言順で返す。"""
    order = _state.get("player_order") or PLAYERS
    return [r for r in order if _state["players"].get(r, {}).get("alive")]


def _get_alive_werewolves() -> list:
    return [r for r in _get_alive_players()
            if _state["players"][r]["role_type"] == "werewolf"]


def _get_alive_villagers() -> list:
    return [r for r in _get_alive_players()
            if _state["players"][r]["role_type"] == "villager"]


def _check_win():
    """勝利判定。勝者文字列 ('village'|'werewolf') or None を返す。"""
    wolves = _get_alive_werewolves()
    villagers = _get_alive_villagers()
    if not wolves:
        return "village"
    if len(wolves) >= len(villagers):
        return "werewolf"
    return None


def _log_event(msg: str):
    _state["event_log"].append(msg)


def _append_log(entry: dict):
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    _state["log"].append(entry)


def _vote_tally(votes: dict | None = None) -> dict:
    """投票途中の集計を返す（表示用）。

    tally は生存者全員をキーにした得票数。
    clinched は、残り票が全部2位候補に入っても逆転できない場合の確定候補。
    同票で並ぶ可能性が残る場合は確定にしない。
    """
    votes = _state.get("votes", {}) if votes is None else votes
    alive = _get_alive_players()
    tally: dict = {r: 0 for r in alive}
    for target in votes.values():
        if target in tally:
            tally[target] += 1

    votes_cast = sum(tally.values())
    votes_total = len(alive)
    votes_remaining = max(votes_total - votes_cast, 0)
    max_votes = max(tally.values()) if tally else 0
    leaders = [r for r, v in tally.items() if max_votes > 0 and v == max_votes]

    clinched = None
    if len(leaders) == 1 and votes_remaining > 0:
        leader = leaders[0]
        others_max = max([v for r, v in tally.items() if r != leader] or [0])
        if tally[leader] > others_max + votes_remaining:
            clinched = leader

    return {
        "tally": tally,
        "votes_cast": votes_cast,
        "votes_total": votes_total,
        "votes_remaining": votes_remaining,
        "max_votes": max_votes,
        "leaders": leaders,
        "clinched": clinched,
    }


def _vote_tally_text(info: dict | None = None) -> str:
    """集計を1行テキストにする（ログ・画面表示用）。"""
    info = _vote_tally() if info is None else info
    parts = [f"{target}:{count}票" for target, count in info["tally"].items() if count > 0]
    body = "、".join(parts) if parts else "まだ票なし"
    text = f"{body}（{info['votes_cast']}/{info['votes_total']}票）"
    if info.get("clinched"):
        text += f" / {info['clinched']}の処刑確定"
    return text


def _alive_list_text() -> str:
    parts = []
    for r in _get_alive_players():
        p = _state["players"][r]
        parts.append(f"{p['kote']}({r})")
    return "、".join(parts) if parts else "なし"


def _event_summary_text() -> str:
    """前ラウンドまでのイベントサマリー（最大2ラウンド分・処刑/犠牲行のみ）。"""
    lines = []
    cur = _state.get("round", 1)
    for e in _state["event_log"]:
        # 現ラウンドのものは除外（今ラウンドの発言で補う）
        if e.startswith(f"R{cur}"):
            continue
        if any(k in e for k in ["処刑", "犠牲"]):
            lines.append(e)
    return "\n".join(lines[-4:]) if lines else "（なし）"


def _day_transcript_text(last_n: int = 4) -> str:
    items = _state["day_transcript"][-last_n:]
    if not items:
        return "（まだ発言なし）"
    return "\n".join(f"{t['kote']}({t['role']}): {t['message']}" for t in items)


# ─────────────────────────────────────────────────
# モデル呼び出し補助
# ─────────────────────────────────────────────────

def _extract_field(result: dict, *keys) -> str:
    """ask_ai の戻り値から文字列フィールドを取り出す。"""
    for k in keys:
        v = result.get(k)
        if v and isinstance(v, str):
            return v.strip()
    for action in (result.get("actions") or []):
        if isinstance(action, dict):
            content = action.get("content") or ""
            if content:
                return content.strip()
    return ""


def _call_model(role: str, prompt: str) -> dict:
    """ask_ai を呼ぶ。失敗は空dict で返す（ゲームを止めない）。"""
    try:
        return ask_ai(prompt, model=role)
    except OllamaUnavailableError as e:
        print(f"[WEREWOLF] {role} 呼び出し失敗（棄権）: {e}")
        return {}
    except Exception as e:
        print(f"[WEREWOLF] {role} 予期しないエラー（棄権）: {e}")
        return {}


# ─────────────────────────────────────────────────
# プロンプト生成
# ─────────────────────────────────────────────────

def _role_prompt(role: str) -> str:
    role_type = _state["players"][role]["role_type"]
    kote = _state["players"][role]["kote"]
    if role_type == "werewolf":
        return (f"あなたは人狼です（コテハン: {kote}）。"
                "正体を隠しながら村人を混乱させ、処刑を回避してください。")
    return (f"あなたは村人です（コテハン: {kote}）。"
            "発言・議論・投票で人狼を見つけて処刑してください。"
            "【厳禁】あなたは村人です。『自分が人狼だ』とは絶対に言わないでください。")


def _build_discuss_prompt(role: str) -> str:
    return (
        f"あなたは人狼ゲームのプレイヤーです。\n"
        f"{_role_prompt(role)}\n\n"
        f"【ラウンド {_state['round']}・昼の議論】\n"
        f"【生存者一覧】{_alive_list_text()}\n\n"
        f"【前ラウンドまでの出来事】\n{_event_summary_text()}\n\n"
        f"【今ラウンドの発言（直近4件）】\n{_day_transcript_text(4)}\n\n"
        "人狼を見つけるために（または正体を隠しながら）1〜2文で発言してください。"
        "必ず日本語で。\n"
        "以下のJSONのみ返してください：\n"
        '{"message": "発言内容（1〜2文、日本語）"}'
    )


def _build_vote_prompt(role: str) -> str:
    alive_others = [r for r in _get_alive_players() if r != role]
    others_text = "、".join(alive_others)
    return (
        f"あなたは人狼ゲームのプレイヤーです。\n"
        f"{_role_prompt(role)}\n\n"
        f"【ラウンド {_state['round']}・投票】\n"
        "今ラウンドの発言を振り返り、誰を処刑すべきか投票してください。\n\n"
        f"【今ラウンドの発言まとめ】\n{_day_transcript_text(99)}\n\n"
        f"【投票できる相手（自分以外の生存者）】\n{others_text}\n\n"
        "以下のJSONのみ返してください：\n"
        '{"vote": "role名（上記リストから1つ）", "reason": "理由1文"}'
    )


def _build_attack_prompt(role: str) -> str:
    targets = _get_alive_villagers()
    targets_text = "、".join(targets)
    return (
        f"夜です。あなたは人狼です。\n"
        f"{_role_prompt(role)}\n\n"
        f"【ラウンド {_state['round']}・夜】\n"
        f"【襲撃できる村人一覧】{targets_text}\n\n"
        "誰を襲撃しますか？\n"
        "以下のJSONのみ返してください：\n"
        '{"attack": "role名（上記リストから1つ）"}'
    )


# ─────────────────────────────────────────────────
# フェーズ別ステップ処理
# ─────────────────────────────────────────────────

import re as _ww_re

def _is_self_werewolf_confession(message: str, kote: str) -> bool:
    """村人が『自分は人狼』と自爆宣言しているか（granite等の崩れ対策）。"""
    if not message:
        return False
    m = message.replace(" ", "").replace("　", "")
    # 「私は人狼」「自分が人狼」「俺は人狼」「{kote}は人狼」等
    pats = [r"私は人狼", r"私が人狼", r"自分は人狼", r"自分が人狼",
            r"俺は人狼", r"僕は人狼", r"わたしは人狼"]
    if kote:
        k = kote.replace(" ", "")
        pats.append(_ww_re.escape(k) + r"は人狼")
        pats.append(_ww_re.escape(k) + r"が人狼")
    return any(_ww_re.search(p, m) for p in pats)


def _step_day_discuss() -> dict:
    alive = _get_alive_players()
    idx = _state["sub_index"]

    if idx >= len(alive):
        # 全員発言済み → 投票フェーズへ
        _state["phase"] = "day_vote"
        _state["sub_index"] = 0
        _save_state()
        return {
            "status": "phase_change",
            "phase": "day_vote",
            "round": _state["round"],
            "message": f"R{_state['round']}の昼議論終了。投票フェーズへ移行します",
        }

    role = alive[idx]
    player = _state["players"][role]
    kote = player["kote"]

    prompt = _build_discuss_prompt(role)
    result = _call_model(role, prompt)

    message = _extract_field(result, "message")
    if not message:
        message = "（発言なし）"
    message = message[:150]

    # ─── 崩れた出力チェック ─────────────────────────────────
    # このプレイヤーの今ラウンドの直前発言を prev として渡す
    prev_msgs_by_role = [
        t["message"] for t in _state["day_transcript"] if t["role"] == role
    ]
    is_valid, invalid_reason = is_valid_japanese_output(message, prev_msgs_by_role)
    # 村人が『自分は人狼』と自爆宣言したら無効扱い（granite等の崩れ対策）
    if is_valid and player["role_type"] == "villager" and _is_self_werewolf_confession(message, kote):
        is_valid = False
        invalid_reason = "村人の自爆宣言"
    speech_fallback = False

    if not is_valid:
        print(f"[WEREWOLF] {role}の発言が無効（{invalid_reason}）→ 再試行")
        result2 = _call_model(role, prompt)
        message2 = _extract_field(result2, "message")
        if message2:
            message2 = message2[:150]
            is_valid2, _ = is_valid_japanese_output(message2, prev_msgs_by_role)
            if is_valid2 and player["role_type"] == "villager" and _is_self_werewolf_confession(message2, kote):
                is_valid2 = False
            if is_valid2:
                message = message2
            else:
                print(f"[WEREWOLF] {role}の再試行も無効 → フォールバック")
                message = _SPEECH_FALLBACK
                speech_fallback = True
        else:
            print(f"[WEREWOLF] {role}の再試行も空 → フォールバック")
            message = _SPEECH_FALLBACK
            speech_fallback = True
    # ────────────────────────────────────────────────────────

    entry = {"role": role, "kote": kote, "message": message, "speech_fallback": speech_fallback}
    _state["day_transcript"].append(entry)
    _state["sub_index"] += 1

    _log_event(f"R{_state['round']}昼: {kote}「{message[:30]}」")
    _append_log({
        "type": "speak",
        "round": _state["round"],
        "phase": "day",
        "role": role,
        "kote": kote,
        "message": message,
        "speech_fallback": speech_fallback,
    })
    _save_state()

    return {
        "status": "ok",
        "action_type": "speak",
        "round": _state["round"],
        "actor": role,
        "kote": kote,
        "message": message,
        "speech_fallback": speech_fallback,
        "phase": "day_discuss",
        "sub_index": _state["sub_index"],
    }


def _step_day_vote() -> dict:
    alive = _get_alive_players()
    idx = _state["sub_index"]

    if idx >= len(alive):
        # 全員投票済み → 集計・処刑
        return _resolve_vote()

    role = alive[idx]
    player = _state["players"][role]
    kote = player["kote"]
    alive_others = [r for r in alive if r != role]

    prompt = _build_vote_prompt(role)
    result = _call_model(role, prompt)

    vote = _extract_field(result, "vote")
    reason = _extract_field(result, "reason")

    # 投票先の検証と救済（1回再試行 → ランダム）
    if not vote or vote not in alive_others:
        if vote:
            print(f"[WEREWOLF] {role}の投票「{vote}」が無効 → 再試行")
            result2 = _call_model(role, prompt)
            vote = _extract_field(result2, "vote")
        if not vote or vote not in alive_others:
            print(f"[WEREWOLF] {role}の投票「{vote}」無効 → ランダム選択")
            vote = random.choice(alive_others) if alive_others else role

    _state["votes"][role] = vote
    _state["sub_index"] += 1

    tally_info = _vote_tally()
    tally_text = _vote_tally_text(tally_info)

    _append_log({
        "type": "vote",
        "round": _state["round"],
        "voter": role,
        "voter_kote": kote,
        "target": vote,
        "reason": reason,
        "tally": tally_info["tally"],
        "votes_cast": tally_info["votes_cast"],
        "votes_total": tally_info["votes_total"],
        "votes_remaining": tally_info["votes_remaining"],
        "clinched": tally_info["clinched"],
        "tally_text": tally_text,
    })
    _save_state()

    return {
        "status": "ok",
        "action_type": "vote",
        "round": _state["round"],
        "actor": role,
        "kote": kote,
        "vote": vote,
        "reason": reason,
        "phase": "day_vote",
        "tally": tally_info["tally"],
        "votes_cast": tally_info["votes_cast"],
        "votes_total": tally_info["votes_total"],
        "votes_remaining": tally_info["votes_remaining"],
        "clinched": tally_info["clinched"],
        "tally_text": tally_text,
    }


def _resolve_vote() -> dict:
    """投票を集計して処刑を実行する。"""
    votes = _state["votes"]
    alive = _get_alive_players()

    # 得票数を集計
    tally: dict = {r: 0 for r in alive}
    for target in votes.values():
        if target in tally:
            tally[target] += 1

    # 最多得票者を選ぶ（同数はランダム）
    max_votes = max(tally.values()) if tally else 0
    candidates = [r for r, v in tally.items() if v == max_votes]
    executed = random.choice(candidates)

    _state["players"][executed]["alive"] = False
    executed_kote = _state["players"][executed]["kote"]
    executed_role_type = _state["players"][executed]["role_type"]
    role_jp = "人狼" if executed_role_type == "werewolf" else "村人"
    tally_str = "、".join(f"{k}:{v}" for k, v in tally.items())

    _log_event(
        f"R{_state['round']}昼: {executed_kote}({executed})が処刑（{role_jp}）。票: {tally_str}"
    )
    _append_log({
        "type": "execute",
        "round": _state["round"],
        "executed": executed,
        "executed_kote": executed_kote,
        "role_type": executed_role_type,
        "tally": tally,
    })

    # 勝敗判定
    winner = _check_win()
    if winner:
        _state["votes"] = {}
        _finish_game(winner)
        return {
            "status": "done",
            "action_type": "execute",
            "round": _state["round"],
            "executed": executed,
            "executed_kote": executed_kote,
            "role_type": executed_role_type,
            "role_jp": role_jp,
            "tally": tally,
            "result": winner,
            "message": (
                f"{executed_kote}({executed})={role_jp}が処刑 → "
                f"{'村人チームの勝利！' if winner == 'village' else '人狼チームの勝利！'}"
            ),
        }

    # 夜フェーズへ
    _state["phase"] = "night"
    _state["sub_index"] = 0
    _state["votes"] = {}
    _save_state()

    return {
        "status": "phase_change",
        "action_type": "execute",
        "round": _state["round"],
        "executed": executed,
        "executed_kote": executed_kote,
        "role_type": executed_role_type,
        "role_jp": role_jp,
        "tally": tally,
        "message": (
            f"{executed_kote}({executed})={role_jp}が処刑されました。夜フェーズへ移行します"
        ),
        "phase": "night",
    }


def _step_night() -> dict:
    wolves = _get_alive_werewolves()

    if not wolves:
        winner = _check_win()
        if winner:
            _finish_game(winner)
        return {"status": "error", "message": "生存中の人狼がいません"}

    wolf_role = wolves[0]
    wolf_kote = _state["players"][wolf_role]["kote"]

    targets = _get_alive_villagers()
    if not targets:
        _finish_game("werewolf")
        return {"status": "done", "result": "werewolf", "message": "村人がいなくなりました"}

    prompt = _build_attack_prompt(wolf_role)
    result = _call_model(wolf_role, prompt)

    attack = _extract_field(result, "attack")

    # 攻撃対象の検証と救済（1回再試行 → ランダム）
    if not attack or attack not in targets:
        if attack:
            print(f"[WEREWOLF] {wolf_role}の攻撃「{attack}」が無効 → 再試行")
            result2 = _call_model(wolf_role, prompt)
            attack = _extract_field(result2, "attack")
        if not attack or attack not in targets:
            print(f"[WEREWOLF] {wolf_role}の攻撃「{attack}」無効 → ランダム選択")
            attack = random.choice(targets)

    _state["players"][attack]["alive"] = False
    attack_kote = _state["players"][attack]["kote"]

    _log_event(f"R{_state['round']}夜: {attack_kote}({attack})が犠牲")
    _append_log({
        "type": "attack",
        "round": _state["round"],
        "wolf": wolf_role,
        "wolf_kote": wolf_kote,
        "victim": attack,
        "victim_kote": attack_kote,
    })

    # 勝敗判定
    winner = _check_win()
    if winner:
        _finish_game(winner)
        return {
            "status": "done",
            "action_type": "attack",
            "round": _state["round"],
            "victim": attack,
            "victim_kote": attack_kote,
            "result": winner,
            "message": (
                f"夜: {attack_kote}({attack})が犠牲に → "
                f"{'村人チームの勝利！' if winner == 'village' else '人狼チームの勝利！'}"
            ),
        }

    # 次ラウンドの昼へ
    _state["round"] += 1
    _state["phase"] = "day_discuss"
    _state["sub_index"] = 0
    _state["day_transcript"] = []
    _state["votes"] = {}
    _save_state()

    return {
        "status": "phase_change",
        "action_type": "attack",
        "prev_round": _state["round"] - 1,
        "victim": attack,
        "victim_kote": attack_kote,
        "message": (
            f"夜: {attack_kote}({attack})が犠牲になりました。"
            f"R{_state['round']}の昼が始まります"
        ),
        "phase": "day_discuss",
        "round": _state["round"],
    }


# ─────────────────────────────────────────────────
# ゲーム終了処理
# ─────────────────────────────────────────────────

def _finish_game(winner: str):
    _state["active"] = False
    _state["phase"] = "done"
    _state["result"] = winner
    winner_label = "村人チーム" if winner == "village" else "人狼チーム"
    _log_event(f"ゲーム終了: {winner_label}の勝利！")
    _append_log({"type": "game_end", "result": winner})

    _write_text_log()

    try:
        _post_to_board()
    except Exception as e:
        print(f"[WEREWOLF] 板投稿失敗（無視）: {e}")

    _save_state()


def _write_text_log():
    """werewolf_log_YYYYMMDD.txt にゲーム全体のログを書き出す。"""
    today = date.today().strftime("%Y%m%d")
    log_path = os.path.join(_BASE, f"werewolf_log_{today}.txt")

    lines = [
        f"===== 人狼ゲームログ {datetime.now().strftime('%Y-%m-%d %H:%M')} =====",
        "",
        "【参加者と役割（公開）】",
    ]
    for r, p in _state["players"].items():
        role_jp = "人狼" if p["role_type"] == "werewolf" else "村人"
        lines.append(f"  {p['kote']}({r}): {role_jp}")

    lines += ["", "【ゲームログ】"]
    for entry in _state["log"]:
        t = entry.get("type")
        r = entry.get("round", "?")
        if t == "game_start":
            continue
        elif t == "speak":
            lines.append(f"  R{r}昼 [{entry['kote']}] {entry['message']}")
        elif t == "vote":
            tally_text = entry.get("tally_text", "")
            tally_suffix = f"  集計: {tally_text}" if tally_text else ""
            lines.append(
                f"  R{r}投票 {entry['voter_kote']}→{entry['target']}"
                f"  理由: {entry.get('reason', '')[:40]}{tally_suffix}"
            )
        elif t == "execute":
            tally_str = " ".join(f"{k}:{v}" for k, v in entry.get("tally", {}).items())
            role_jp = "人狼" if entry["role_type"] == "werewolf" else "村人"
            lines.append(
                f"  R{r}処刑 {entry['executed_kote']}({entry['executed']})={role_jp}"
                f"  票: {tally_str}"
            )
        elif t == "attack":
            lines.append(f"  R{r}夜  {entry['victim_kote']}({entry['victim']})が犠牲")
        elif t == "game_end":
            winner_label = "村人チーム" if entry["result"] == "village" else "人狼チーム"
            lines.append(f"  ★ {winner_label}の勝利！")

    lines += ["", "【イベントサマリー】"]
    for e in _state["event_log"]:
        lines.append(f"  {e}")

    lines += ["", "=" * 50, ""]

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[WEREWOLF] ログ保存: {log_path}")
    except Exception as e:
        print(f"[WEREWOLF] ログ保存失敗: {e}")


def _post_to_board():
    """ゲーム終了後、結果を板のゲーム広場に1投稿する。"""
    from executor import _board_post

    winner = _state["result"]
    winner_label = "村人チーム" if winner == "village" else "人狼チーム"
    title = f"AI人狼ゲーム結果 — {winner_label}の勝利！"

    lines = [f"【{winner_label}の勝利！】", "", "◆ 参加者と役割"]
    for r, p in _state["players"].items():
        role_jp = "人狼" if p["role_type"] == "werewolf" else "村人"
        lines.append(f"  {p['kote']}: {role_jp}")

    lines += ["", "◆ 経過"]
    for e in _state["event_log"]:
        if any(k in e for k in ["処刑", "犠牲", "終了"]):
            lines.append(f"  {e}")

    body = "\n".join(lines)
    ok = _board_post(BOARD_CAT_ID, title, body, kote="LA")
    print(f"[WEREWOLF] 板投稿: {'成功' if ok else '失敗'}")


# ─────────────────────────────────────────────────
# 公開インターフェース
# ─────────────────────────────────────────────────

def start_game(players: list = None) -> dict:
    """ゲームを開始する。players 未指定時はデフォルト5人。"""
    global _state

    if _state.get("active"):
        return {
            "status": "error",
            "message": "ゲームが既に進行中です。/werewolf/cancel で中断してから開始してください",
        }

    if players is None:
        players = PLAYERS

    kote_map = _load_kote()

    # 役割をランダム割当（シードなし・毎回異なる）
    roles_pool = (["werewolf"] * WEREWOLF_COUNT
                  + ["villager"] * (len(players) - WEREWOLF_COUNT))
    random.shuffle(roles_pool)

    player_data = {}
    for i, role in enumerate(players):
        player_data[role] = {
            "role_type": roles_pool[i],
            "alive": True,
            "kote": kote_map.get(role, role),
        }

    # 発言順もシャッフル（役割割り当てとは独立）
    speaking_order = list(players)
    random.shuffle(speaking_order)

    _state = {
        "active": True,
        "round": 1,
        "phase": "day_discuss",
        "sub_index": 0,
        "players": player_data,
        "player_order": speaking_order,
        "day_transcript": [],
        "votes": {},
        "event_log": [],
        "result": None,
        "log": [],
    }

    wolf_roles = [r for r, p in player_data.items() if p["role_type"] == "werewolf"]
    wolf_kotes = [player_data[r]["kote"] for r in wolf_roles]
    _log_event(f"ゲーム開始: 人狼は {','.join(wolf_kotes)}（非公開）")
    _append_log({
        "type": "game_start",
        "round": 1,
        "players": {r: {"role_type": p["role_type"], "kote": p["kote"]}
                    for r, p in player_data.items()},
    })
    _save_state()

    print(f"[WEREWOLF] ゲーム開始！人狼: {wolf_roles}")

    return {
        "status": "started",
        "round": 1,
        "phase": "day_discuss",
        "players": [{"role": r, "kote": p["kote"]} for r, p in player_data.items()],
        "message": f"人狼ゲーム開始！{len(players)}人・人狼{WEREWOLF_COUNT}人",
    }


def next_action() -> dict:
    """次の1アクションを実行する。"""
    if not _state.get("active"):
        if _state.get("result"):
            return {
                "status": "done",
                "result": _state["result"],
                "message": "ゲームは終了しています",
            }
        return {
            "status": "error",
            "message": "ゲームが開始されていません。/werewolf/start を呼んでください",
        }

    phase = _state.get("phase")

    if phase == "day_discuss":
        return _step_day_discuss()
    elif phase == "day_vote":
        return _step_day_vote()
    elif phase == "night":
        return _step_night()
    elif phase == "done":
        return {
            "status": "done",
            "result": _state.get("result"),
            "message": "ゲームは終了しています",
        }
    else:
        return {"status": "error", "message": f"不明なフェーズ: {phase}"}


def get_state() -> dict:
    """現在の状態スナップショットを返す。"""
    tally_info = _vote_tally()
    return {
        "active": _state.get("active"),
        "round": _state.get("round"),
        "phase": _state.get("phase"),
        "result": _state.get("result"),
        "players": _state.get("players", {}),
        "event_log": _state.get("event_log", []),
        "day_transcript": _state.get("day_transcript", []),
        "votes": _state.get("votes", {}),
        "vote_tally": tally_info,
        "tally": tally_info["tally"],
        "tally_text": _vote_tally_text(tally_info),
    }


def cancel_game() -> dict:
    """ゲームを中断・リセットする。"""
    _reset()
    return {"status": "cancelled", "message": "ゲームをキャンセルしました"}


# 起動時に状態を復元
_load_state()
