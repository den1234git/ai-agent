"""
village_werewolf_player.py — 村のモデルが板の人狼ゲームに参加するクライアント

実行方法:
  python village_werewolf_player.py            # 5体で1ゲーム
  python village_werewolf_player.py --single   # 1体だけ参加（他を待つ）
    --kote "[村]人狼参加A" --model swallow

制約:
  - identity_id はローカルのみ保持（ログ・出力に出さない）
  - 本番KVへの書き込みはデプロイ後にユーザー判断で実行
  - qwen3 は除外（重いため）

Worker API フィールド名（Phase 1 実装に完全一致）:
  POST /api/werewolf/join
    req:  { kote }
    res:  { ok, matched:bool, game_id?, role?, identity_id?, players?, queue_position? }
  GET  /api/werewolf/join?kote=
    res:  { matched:bool, game_id?, role?, identity_id? } | { waiting:bool, queue_position? }
  GET  /api/werewolf/{id}?identity_id=
    res:  { id, status, phase, round, players, speakers_done, votes, log,
            execution_log, night_kill_log, winner }
  POST /api/werewolf/{id}/action
    req:  { identity_id, type:"speak"|"vote"|"kill", text?:str, target?:str }
    res speak: { ok, action:"speak_recorded", phase, all_spoke }
    res vote:  { ok, action:"vote_recorded", all_voted, phase?, winner? }
    res kill:  { ok, action:"kill_executed", phase, winner? }
"""

import argparse
import json
import os
import re
import sys
import time
import random
import threading
import requests

sys.stdout.reconfigure(encoding="utf-8")

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from output_validator import is_valid_japanese_output

# ─── 定数 ──────────────────────────────────────────────────────
API_BASE       = "https://fnoita-board.mogumogu7la001.workers.dev"
OLLAMA_URL     = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 300
KEEP_ALIVE     = "0"

POLL_INTERVAL  = 8     # ポーリング間隔（秒）
MATCH_TIMEOUT  = 600   # マッチング待ち最大時間（秒）
GAME_TIMEOUT   = 3600  # ゲームプレイ最大時間（秒）

# 5体のモデル割り当て（kote, model_key）
WW_POOL = [
    ("[村]人狼参加A", "swallow"),
    ("[村]人狼参加B", "llama3_2"),
    ("[村]人狼参加C", "gemma3"),
    ("[村]人狼参加D", "llama3_1"),
    ("[村]人狼参加E", "router"),
]

SPEECH_FALLBACK    = "…（うまく話せなかった）"
MAX_SPEAK_ATTEMPTS = 2
MAX_VOTE_ATTEMPTS  = 3


# ─── モデル名解決 ─────────────────────────────────────────────
def _get_model_name(key: str) -> str:
    config_path = os.path.join(_BASE, "models.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg["models"].get(key, key)
    except Exception:
        return key


# ─── Ollama 呼び出し ──────────────────────────────────────────
def _call_ollama(model_key: str, system: str, messages: list, label: str = "") -> str:
    model_name = _get_model_name(model_key)
    ollama_messages = [{"role": "system", "content": system}] + messages
    for attempt in range(1, 4):
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model":      model_name,
                    "messages":   ollama_messages,
                    "stream":     False,
                    "keep_alive": KEEP_ALIVE,
                    "think":      False,
                    "options":    {"num_predict": 300, "repeat_penalty": 1.15},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            res.raise_for_status()
            raw = res.json()["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if text:
                print(f"[WW][{label}] 応答: {text[:80]}")
            return text
        except requests.exceptions.Timeout:
            print(f"[WW][{label}] タイムアウト ({attempt}/3)")
        except Exception as e:
            print(f"[WW][{label}] Ollama エラー ({attempt}/3): {e}")
        if attempt < 3:
            time.sleep(5)
    return ""


# ─── WerewolfPlayer ───────────────────────────────────────────
class WerewolfPlayer:
    """1プレイヤー分のゲームセッション（join → poll → action）を管理する"""

    def __init__(self, kote: str, model_key: str):
        self.kote      = kote
        self.model_key = model_key
        self.game_id: str | None = None
        self.role: str | None    = None   # "werewolf" | "villager"
        self._identity_id: str | None = None  # ローカルのみ保持・出力しない
        self.prev_speeches: list = []

    # ── join ──────────────────────────────────────────────────
    def join(self) -> str:
        """
        POST /api/werewolf/join
        戻り値: "matched" | "waiting" | "error"
        """
        for attempt in range(1, 4):
            try:
                r = requests.post(
                    API_BASE + "/api/werewolf/join",
                    json={"kote": self.kote},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
            except Exception as e:
                print(f"[WW][{self.kote}] join エラー ({attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(5)
                    continue
                return "error"

            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", 65))
                print(f"[WW][{self.kote}] join レート制限 → {retry}秒待機")
                time.sleep(retry)
                continue

            if r.status_code == 409:
                # 既にゲームに組み込まれている（他プレイヤーの join がマッチを成立させた）。
                # エラー中断ではなく、wait_for_match で通知（役割つき）を拾って参加する。
                try:
                    self.game_id = r.json().get("game_id")
                except Exception:
                    pass
                print(f"[WW][{self.kote}] 既にゲームに参加済み → 通知待ちに移行 (game={self.game_id})")
                return "waiting"

            if not r.ok:
                print(f"[WW][{self.kote}] join 失敗: {r.status_code} {r.text[:120]}")
                if attempt < 3:
                    time.sleep(5)
                    continue
                return "error"

            data = r.json()
            # Worker は matched:true/false を返す（status フィールドなし）
            if data.get("matched"):
                self.game_id      = data.get("game_id")
                self.role         = data.get("role")
                self._identity_id = data.get("identity_id")
                print(f"[WW][{self.kote}] マッチング成立: role=[hidden] game={self.game_id}")
                return "matched"
            elif data.get("ok"):
                pos = data.get("queue_position", "?")
                print(f"[WW][{self.kote}] 待機中 (queue位置={pos})")
                return "waiting"
            else:
                print(f"[WW][{self.kote}] 予期しないレスポンス: {data}")
                return "error"
        return "error"

    # ── wait_for_match ────────────────────────────────────────
    def wait_for_match(self, timeout: int = MATCH_TIMEOUT) -> bool:
        """
        GET /api/werewolf/join?kote={kote} をポーリングしてマッチング通知を受け取る。
        通知には game_id / role / identity_id が含まれる。
        """
        if self.role:  # 既にマッチング済み
            return True
        start = time.time()
        kote_enc = requests.utils.quote(self.kote, safe="")
        while time.time() - start < timeout:
            try:
                r = requests.get(
                    API_BASE + f"/api/werewolf/join?kote={kote_enc}",
                    timeout=10,
                )
                if r.ok:
                    data = r.json()
                    if data.get("matched"):
                        self.game_id      = data.get("game_id", self.game_id)
                        self.role         = data.get("role", self.role)
                        self._identity_id = data.get("identity_id", self._identity_id)
                        print(f"[WW][{self.kote}] マッチング確認: game={self.game_id}")
                        return True
                    pos = data.get("queue_position", "?")
                    print(f"[WW][{self.kote}] 待機中 (queue={pos})")
            except Exception as e:
                print(f"[WW][{self.kote}] ポーリングエラー: {e}")
            time.sleep(POLL_INTERVAL)
        print(f"[WW][{self.kote}] マッチングタイムアウト ({timeout}秒)")
        return False

    # ── game state ────────────────────────────────────────────
    def _get_game_state(self) -> dict | None:
        """GET /api/werewolf/{id}?identity_id=... でゲーム状態を取得。
        自分の役のみ公開（他プレイヤーの役は '?'）。"""
        if not self.game_id or not self._identity_id:
            return None
        iid = self._identity_id  # ローカル変数に取り出し（外部に出ない）
        try:
            r = requests.get(
                API_BASE + f"/api/werewolf/{self.game_id}",
                params={"identity_id": iid},
                timeout=10,
            )
            if r.ok:
                return r.json()
            return None
        except Exception as e:
            print(f"[WW][{self.kote}] GET state エラー: {e}")
            return None

    # ── turn detection ────────────────────────────────────────
    def _is_my_speak_turn(self, game: dict) -> bool:
        """昼の議論フェーズ かつ まだ発言していない"""
        return (game.get("phase") == "day_discussion"
                and self.kote not in game.get("speakers_done", []))

    def _is_my_vote_turn(self, game: dict) -> bool:
        """投票フェーズ かつ まだ投票していない（votes は voting フェーズのみ公開）"""
        votes = game.get("votes", {})
        return (game.get("phase") == "voting"
                and self.kote not in votes)

    def _is_my_kill_turn(self, game: dict) -> bool:
        """夜フェーズ かつ 人狼"""
        return (game.get("phase") == "night"
                and self.role == "werewolf")

    # ── prompt helpers ────────────────────────────────────────
    def _alive_kotes(self, game: dict) -> list:
        return [p["kote"] for p in game.get("players", []) if p.get("alive")]

    def _alive_others(self, game: dict) -> list:
        return [k for k in self._alive_kotes(game) if k != self.kote]

    def _recent_speak_text(self, game: dict, last_n: int = 6) -> str:
        entries = [
            e for e in game.get("log", [])
            if e.get("type") == "speak" and e.get("round") == game.get("round", 1)
        ]
        if not entries:
            return "（まだ発言なし）"
        return "\n".join(f"  {e['kote']}: {e.get('text','')[:120]}" for e in entries[-last_n:])

    def _event_summary(self, game: dict) -> str:
        lines = []
        for e in game.get("execution_log", []):
            lines.append(f"R{e['round']} 処刑: {e['executed_kote']}")
        for e in game.get("night_kill_log", []):
            lines.append(f"R{e['round']} 夜: {e['killed_kote']} が犠牲に")
        return "\n".join(lines[-4:]) if lines else "（なし）"

    # ── generate actions ──────────────────────────────────────
    def _generate_speak(self, game: dict) -> str:
        role_desc = (
            "あなたは人狼です。正体を隠しながら村人を混乱させ、処刑を回避してください。"
            if self.role == "werewolf"
            else "あなたは村人です。発言・議論・投票で人狼を見つけて処刑してください。"
        )
        alive_text = "、".join(self._alive_kotes(game))
        system = (
            f"あなたは人狼ゲームのプレイヤーです（コテハン: {self.kote}）。\n"
            f"{role_desc}\n"
            "出力形式：日本語の発言文のみ（1〜2文）。JSONラベルや前置き不要。"
        )
        messages = [{
            "role": "user",
            "content": (
                f"【ラウンド {game.get('round', 1)}・昼の議論】\n"
                f"【生存者】{alive_text}\n\n"
                f"【過去の出来事】\n{self._event_summary(game)}\n\n"
                f"【今ラウンドの発言（直近6件）】\n{self._recent_speak_text(game)}\n\n"
                "人狼を見つけるために（または正体を隠しながら）1〜2文で発言してください。"
            ),
        }]
        for attempt in range(1, MAX_SPEAK_ATTEMPTS + 1):
            raw = _call_ollama(self.model_key, system, messages, label=self.kote)
            if not raw:
                continue
            text = raw[:400].strip()
            # JSON 出力のフォールバック抽出
            m = re.search(r'"message"\s*:\s*"([^"]{3,})"', text)
            if m:
                text = m.group(1)
            text = text[:200]
            is_valid, reason = is_valid_japanese_output(text, self.prev_speeches)
            if is_valid:
                return text
            print(f"[WW][{self.kote}] speak 無効 ({attempt}/{MAX_SPEAK_ATTEMPTS}): {reason}")
        return SPEECH_FALLBACK

    def _generate_vote(self, game: dict) -> str:
        alive_others = self._alive_others(game)
        if not alive_others:
            return ""
        targets_text = "、".join(alive_others)
        system = (
            f"あなたは人狼ゲームのプレイヤーです（コテハン: {self.kote}）。\n"
            "今ラウンドの議論を振り返り、誰を処刑すべきか選んでください。\n"
            '以下のJSONのみ返してください：{"vote": "コテハン名", "reason": "理由1文"}'
        )
        messages = [{
            "role": "user",
            "content": (
                f"【投票候補（自分以外の生存者）】{targets_text}\n\n"
                f"【今ラウンドの発言】\n{self._recent_speak_text(game, 99)}"
            ),
        }]
        for attempt in range(1, MAX_VOTE_ATTEMPTS + 1):
            raw = _call_ollama(self.model_key, system, messages, label=f"{self.kote}(vote)")
            m = re.search(r'"vote"\s*:\s*"([^"]+)"', raw or "")
            if m:
                target = m.group(1).strip()
                if target in alive_others:
                    return target
                # 部分一致フォールバック
                for k in alive_others:
                    if target in k or k in target:
                        print(f"[WW][{self.kote}] vote 部分一致 → {k}")
                        return k
            print(f"[WW][{self.kote}] vote パース失敗 ({attempt}/{MAX_VOTE_ATTEMPTS})")
        # 最終フォールバック: ランダム
        choice = random.choice(alive_others)
        print(f"[WW][{self.kote}] vote ランダム選択 → {choice}")
        return choice

    def _generate_kill(self, game: dict) -> str:
        alive_others = self._alive_others(game)
        if not alive_others:
            return ""
        targets_text = "、".join(alive_others)
        system = (
            f"あなたは人狼です（コテハン: {self.kote}）。夜です。\n"
            '以下のJSONのみ返してください：{"attack": "コテハン名"}'
        )
        messages = [{
            "role": "user",
            "content": f"【襲撃できる相手】{targets_text}\n誰を今夜の犠牲者にしますか？",
        }]
        raw = _call_ollama(self.model_key, system, messages, label=f"{self.kote}(kill)")
        m = re.search(r'"attack"\s*:\s*"([^"]+)"', raw or "")
        if m:
            target = m.group(1).strip()
            if target in alive_others:
                return target
            for k in alive_others:
                if target in k or k in target:
                    return k
        choice = random.choice(alive_others)
        print(f"[WW][{self.kote}] kill ランダム選択 → {choice}")
        return choice

    # ── post action ───────────────────────────────────────────
    def _post_action(self, action_type: str, **kwargs) -> dict | None:
        """
        POST /api/werewolf/{id}/action
        body: { identity_id, type, text? (speak), target? (vote/kill) }
        Worker のフィールド名に完全一致させる。
        """
        iid = self._identity_id
        body: dict = {"identity_id": iid, "type": action_type}
        if action_type == "speak":
            body["text"] = kwargs.get("text", "")      # フィールド名: text
        elif action_type in ("vote", "kill"):
            body["target"] = kwargs.get("target", "")  # フィールド名: target
        try:
            r = requests.post(
                API_BASE + f"/api/werewolf/{self.game_id}/action",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if r.ok:
                return r.json()
            print(f"[WW][{self.kote}] action={action_type} 失敗: {r.status_code} {r.text[:120]}")
            return None
        except Exception as e:
            print(f"[WW][{self.kote}] action={action_type} エラー: {e}")
            return None

    # ── play ──────────────────────────────────────────────────
    def play(self, timeout: int = GAME_TIMEOUT) -> str | None:
        """
        ゲームが終了するまでポーリング → 自分のターンなら action を送る。
        戻り値: winner ("villager" | "werewolf" | "timeout" | None)
        """
        start = time.time()
        while time.time() - start < timeout:
            game = self._get_game_state()
            if not game:
                time.sleep(POLL_INTERVAL)
                continue

            status = game.get("status")
            phase  = game.get("phase")
            rnd    = game.get("round", 1)

            if status == "done":
                winner = game.get("winner")
                print(f"[WW][{self.kote}] ゲーム終了: winner={winner}")
                return winner

            # ── 昼の発言 ──────────────────────────────────────
            if self._is_my_speak_turn(game):
                print(f"[WW][{self.kote}] R{rnd} 発言ターン（phase={phase}）")
                text = self._generate_speak(game)
                self.prev_speeches.append(text)
                result = self._post_action("speak", text=text)
                if result and result.get("ok"):
                    print(f"[WW][{self.kote}] 発言送信完了")
                    if result.get("all_spoke"):
                        print(f"[WW][{self.kote}] 全員発言済み → 投票フェーズへ")
                else:
                    print(f"[WW][{self.kote}] 発言送信失敗: {result}")
                continue  # 即座に再ポーリング

            # ── 投票 ──────────────────────────────────────────
            if self._is_my_vote_turn(game):
                print(f"[WW][{self.kote}] R{rnd} 投票ターン")
                target = self._generate_vote(game)
                if target:
                    result = self._post_action("vote", target=target)
                    if result and result.get("ok"):
                        print(f"[WW][{self.kote}] 投票送信完了: → {target}")
                        if result.get("all_voted"):
                            print(f"[WW][{self.kote}] 全員投票済み → 処刑・夜フェーズへ")
                    else:
                        print(f"[WW][{self.kote}] 投票送信失敗: {result}")
                continue

            # ── 夜の襲撃（人狼のみ）──────────────────────────
            if self._is_my_kill_turn(game):
                print(f"[WW][{self.kote}] R{rnd} 夜: 襲撃ターン")
                target = self._generate_kill(game)
                if target:
                    result = self._post_action("kill", target=target)
                    if result and result.get("ok"):
                        print(f"[WW][{self.kote}] 襲撃送信完了: → {target}")
                    else:
                        print(f"[WW][{self.kote}] 襲撃送信失敗: {result}")
                continue

            # 自分のターンでなければ待機
            time.sleep(POLL_INTERVAL)

        print(f"[WW][{self.kote}] ゲームタイムアウト ({timeout}秒)")
        return None


# ─── join レート制限対策（CTF と同じパターン） ─────────────────
def _wait_for_next_minute():
    """分の境界まで待機する。同一 kote の連続 join に使う。不要な場合もあるが安全のため。"""
    seconds_into_minute = time.time() % 60
    wait = 60 - seconds_into_minute + 2
    total = int(wait)
    print(f"[WW] 分の境界まで {total}秒待機... 中断しないでください")
    elapsed = 0
    while elapsed < total:
        chunk = min(10, total - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        if elapsed < total:
            print(f"[WW]   残り {total - elapsed}秒...")


# ─── 5人マッチング実行 ────────────────────────────────────────
def run_match(pool: list = None):
    """5体のプレイヤーを join させてゲームを実行する。"""
    if pool is None:
        pool = WW_POOL
    players = [WerewolfPlayer(kote, model) for kote, model in pool]

    print(f"[WW] {'='*50}")
    print(f"[WW] 人狼ゲーム開始: {len(players)}体が参加します")
    for p in players:
        print(f"[WW]   {p.kote} ({p.model_key})")
    print(f"[WW] {'='*50}")

    # 各プレイヤーが join（kote が全員異なるのでレート制限は不要だが念のため1秒の間隔を開ける）
    statuses = []
    for i, p in enumerate(players):
        print(f"[WW] ({i+1}/{len(players)}) {p.kote} join 中...")
        s = p.join()
        statuses.append(s)
        if s == "error":
            print(f"[WW] {p.kote} join 失敗 → 中断")
            return
        if i < len(players) - 1:
            # KV の書き込み伝播待ち: 間隔が短いと前の join がまだキューに反映されず
            # お互い上書きしてキューが育たない（マッチングが成立しない）。
            # 8秒空けて、各 join が前の全員を確実に見られるようにする。
            print(f"[WW]   次の join まで 8秒待機（キュー反映待ち）...")
            time.sleep(8)

    print(f"[WW] 全員 join 完了: {dict(zip([p.kote for p in players], statuses))}")
    print("[WW] マッチング待ち...")

    # "waiting" のプレイヤーはポーリングでマッチング通知を受け取る
    def poll_match(p):
        return p.wait_for_match(timeout=MATCH_TIMEOUT)

    waiting = [(i, p) for i, p in enumerate(players) if statuses[i] == "waiting"]
    if waiting:
        threads, results = [], {}
        for i, p in waiting:
            def _run(pp=p):
                results[pp.kote] = poll_match(pp)
            t = threading.Thread(target=_run, daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=MATCH_TIMEOUT + 30)
        if not all(results.values()):
            failed = [k for k, ok in results.items() if not ok]
            print(f"[WW] マッチング失敗: {failed}")
            return

    print("[WW] 全員マッチング完了。ゲーム開始！")
    print(f"[WW] game_id={players[0].game_id}")

    # 全員を並行でプレイさせる
    def play_player(p):
        try:
            result = p.play(timeout=GAME_TIMEOUT)
            print(f"[WW] {p.kote} 終了 (winner={result})")
        except Exception as e:
            print(f"[WW] {p.kote} 例外: {e}")

    threads = [threading.Thread(target=play_player, args=(p,), daemon=True) for p in players]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=GAME_TIMEOUT + 60)

    print("[WW] ゲーム終了")


# ─── シングルモード（1体だけ参加） ────────────────────────────
def run_single(kote: str, model_key: str):
    """1体だけ参加。他の参加者を待ってゲームをプレイする。"""
    p = WerewolfPlayer(kote, model_key)
    print(f"[WW] シングルモード: {kote} ({model_key})")
    s = p.join()
    if s == "error":
        print("[WW] join 失敗 → 中断")
        return
    if s == "waiting":
        ok = p.wait_for_match()
        if not ok:
            print("[WW] マッチングタイムアウト → 中断")
            return
    print(f"[WW] マッチング成立: game={p.game_id}")
    p.play()


# ─── エントリポイント ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人狼ゲーム クライアント")
    parser.add_argument("--single", action="store_true", help="1体だけ参加（他を待つ）")
    parser.add_argument("--kote",   default="[村]人狼参加A", help="コテハン（シングルモード時）")
    parser.add_argument("--model",  default="swallow",       help="モデルキー（シングルモード時）")
    args = parser.parse_args()

    if args.single:
        run_single(args.kote, args.model)
    else:
        run_match()
