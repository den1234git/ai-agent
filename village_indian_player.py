"""
village_indian_player.py — 村のモデルが板のインディアンポーカーに参加するクライアント

実行方法:
  python village_indian_player.py            # 5体で1ゲーム
  python village_indian_player.py --single   # 1体だけ参加（他を待つ）
    --kote "[村]ポーカーA" --model swallow

制約:
  - identity_id はローカルのみ保持（ログ・出力に出さない）
  - 本番KVへの書き込みはデプロイ後にユーザー判断で実行

Worker API フィールド名（Phase 1 実装に完全一致）:
  POST /api/indian/join
    req:  { kote }
    res:  { ok, matched:bool, game_id?, identity_id?, players? }
  GET  /api/indian/join?kote=
    res:  { matched:bool, game_id?, identity_id? } | { matched:false, waiting:bool }
  GET  /api/indian/{id}?identity_id=
    res:  { id, status, phase, total_turn, current_player_idx,
            current_turn:{turn,questioner_kote,question,answerer_kotes,answers,...}
            players:[{kote,word("?"=自分),turn_count,skip_next_turn,order}],
            log, winner }
  POST /api/indian/{id}/action
    type:"question" → { identity_id, type:"question", text:"質問文" }
    type:"answer"   → { identity_id, type:"answer",   answer:"はい"|"いいえ", reason:"..." }
    type:"guess"    → { identity_id, type:"guess",    guess:"文言"（空文字=パス） }
"""

import argparse
import json
import os
import re
import sys
import time
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
HISTORY_WINDOW = 5     # プロンプトに渡す直近履歴件数

WORD_LIST = [
    "宇宙","魔法","嵐","虹","猫","海","桜","月","雷","炎",
    "森","夢","鳥","花","星","風","山","剣","城","龍",
]

# 5体のモデル割り当て（kote, model_key）— qwen3 は除外
IP_POOL = [
    ("[村]ポーカーA", "swallow"),
    ("[村]ポーカーB", "llama3_2"),
    ("[村]ポーカーC", "gemma3"),
    ("[村]ポーカーD", "llama3_1"),
    ("[村]ポーカーE", "router"),
]

# 候補の言葉リスト（worker.js の IP_WORD_LIST と必ず一致させること）
# 自分のお題はこの中の1つ。他プレイヤーの見えてる言葉を除けば候補が絞れる。
IP_WORD_LIST = [
    '宇宙', '魔法', '嵐', '虹', '猫', '海', '桜', '月', '雷', '炎',
    '森', '夢', '鳥', '花', '星', '風', '山', '剣', '城', '龍',
]

# 大モデル用プール（--big）: 全員同じ qwen3:14b を使い回す（ollama はモデルを1個だけ載せる）
BIG_POOL = [
    ("[大]ポーカーA", "qwen3"),
    ("[大]ポーカーB", "qwen3"),
    ("[大]ポーカーC", "qwen3"),
    ("[大]ポーカーD", "qwen3"),
    ("[大]ポーカーE", "qwen3"),
]


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
                    "options":    {"num_predict": 200, "repeat_penalty": 1.15},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            res.raise_for_status()
            raw  = res.json()["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if text:
                print(f"[IP][{label}] 応答: {text[:80]}")
            return text
        except requests.exceptions.Timeout:
            print(f"[IP][{label}] タイムアウト ({attempt}/3)")
        except Exception as e:
            print(f"[IP][{label}] Ollama エラー ({attempt}/3): {e}")
        if attempt < 3:
            time.sleep(5)
    return ""


# ─── JSON 抽出ヘルパー ────────────────────────────────────────
def _extract_json(raw: str) -> dict:
    """LLM 出力から JSON オブジェクトを抽出する"""
    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# ─── IndianPokerPlayer ────────────────────────────────────────
class IndianPokerPlayer:
    """1プレイヤー分のゲームセッション（join → poll → action）を管理する"""

    def __init__(self, kote: str, model_key: str, big_mode: bool = False):
        self.kote      = kote
        self.model_key = model_key
        self.big_mode  = big_mode  # 大モデル用: 足場（型強制）を外した最小プロンプトで遊ぶ
        self.game_id: str | None = None
        self._identity_id: str | None = None  # ローカルのみ保持・出力しない

    # ── join ──────────────────────────────────────────────────
    def join(self) -> str:
        """POST /api/indian/join → "matched" | "waiting" | "error" """
        for attempt in range(1, 4):
            try:
                r = requests.post(
                    API_BASE + "/api/indian/join",
                    json={"kote": self.kote},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
            except Exception as e:
                print(f"[IP][{self.kote}] join エラー ({attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(5)
                    continue
                return "error"

            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", 65))
                print(f"[IP][{self.kote}] join レート制限 → {retry}秒待機")
                time.sleep(retry)
                continue

            if r.status_code == 409:
                # 既にゲームに組み込まれている（他プレイヤーの join がマッチを成立させた）。
                # エラー中断ではなく、wait_for_match で通知を拾って参加する。
                try:
                    self.game_id = r.json().get("game_id")
                except Exception:
                    pass
                print(f"[IP][{self.kote}] 既にゲームに参加済み → 通知待ちに移行 (game={self.game_id})")
                return "waiting"

            if not r.ok:
                print(f"[IP][{self.kote}] join 失敗: {r.status_code} {r.text[:120]}")
                if attempt < 3:
                    time.sleep(5)
                    continue
                return "error"

            data = r.json()
            if data.get("matched"):
                self.game_id      = data.get("game_id")
                self._identity_id = data.get("identity_id")
                print(f"[IP][{self.kote}] マッチング成立: game={self.game_id}")
                return "matched"
            elif data.get("ok"):
                pos = data.get("queue_position", "?")
                print(f"[IP][{self.kote}] 待機中 (queue={pos})")
                return "waiting"
            else:
                print(f"[IP][{self.kote}] 予期しないレスポンス: {data}")
                return "error"
        return "error"

    # ── wait_for_match ────────────────────────────────────────
    def wait_for_match(self, timeout: int = MATCH_TIMEOUT) -> bool:
        """GET /api/indian/join?kote= をポーリングしてマッチング通知を受け取る"""
        if self._identity_id:
            return True
        start = time.time()
        kote_enc = requests.utils.quote(self.kote, safe="")
        while time.time() - start < timeout:
            try:
                r = requests.get(
                    API_BASE + f"/api/indian/join?kote={kote_enc}",
                    timeout=10,
                )
                if r.ok:
                    data = r.json()
                    if data.get("matched"):
                        self.game_id      = data.get("game_id", self.game_id)
                        self._identity_id = data.get("identity_id", self._identity_id)
                        print(f"[IP][{self.kote}] マッチング確認: game={self.game_id}")
                        return True
                    pos = data.get("queue_position", "?")
                    print(f"[IP][{self.kote}] 待機中 (queue={pos})")
            except Exception as e:
                print(f"[IP][{self.kote}] ポーリングエラー: {e}")
            time.sleep(POLL_INTERVAL)
        print(f"[IP][{self.kote}] マッチングタイムアウト ({timeout}秒)")
        return False

    # ── game state ────────────────────────────────────────────
    def _get_game_state(self) -> dict | None:
        """GET /api/indian/{id}?identity_id=... でゲーム状態を取得"""
        if not self.game_id or not self._identity_id:
            return None
        iid = self._identity_id
        try:
            r = requests.get(
                API_BASE + f"/api/indian/{self.game_id}",
                params={"identity_id": iid},
                timeout=10,
            )
            if r.ok:
                return r.json()
            return None
        except Exception as e:
            print(f"[IP][{self.kote}] GET state エラー: {e}")
            return None

    # ── ターン判定 ─────────────────────────────────────────────
    def _is_my_question_turn(self, game: dict) -> bool:
        players = game.get("players", [])
        idx = game.get("current_player_idx", 0)
        if idx >= len(players):
            return False
        return (game.get("phase") == "question"
                and players[idx]["kote"] == self.kote)

    def _is_my_answer_turn(self, game: dict) -> bool:
        if game.get("phase") != "answer":
            return False
        ct = game.get("current_turn")
        if not ct:
            return False
        answerer_kotes = ct.get("answerer_kotes", [])
        answers        = ct.get("answers", {})
        return self.kote in answerer_kotes and self.kote not in answers

    def _is_my_guess_turn(self, game: dict) -> bool:
        players = game.get("players", [])
        idx = game.get("current_player_idx", 0)
        if idx >= len(players):
            return False
        return (game.get("phase") == "guess"
                and players[idx]["kote"] == self.kote)

    # ── プロンプト補助 ─────────────────────────────────────────
    def _other_words_text(self, game: dict) -> str:
        """自分以外のプレイヤーの文言一覧（自分の文言は '?' で見えない）"""
        lines = []
        for p in game.get("players", []):
            if p["kote"] == self.kote:
                continue
            lines.append(f"  ・{p['kote']}: 「{p['word']}」")
        return "\n".join(lines) if lines else "（なし）"

    def _history_text(self, game: dict) -> str:
        """直近ログから質問・回答・推測のサマリ"""
        log = game.get("log", [])
        recent = log[-HISTORY_WINDOW:] if len(log) > HISTORY_WINDOW else log
        if not recent:
            return "（まだ質問なし）"
        lines = []
        for h in recent:
            q = h.get("question", "?")
            a = h.get("majority_answer", "?")
            who = h.get("questioner_kote", "?")
            line = f"T{h['turn']} {who}「{q}」→{a}"
            if h.get("guessed") and h.get("guess"):
                mark = "✓" if h.get("correct") else "✗"
                line += f" 推測「{h['guess']}」{mark}"
            lines.append(line)
        return "\n".join(lines)

    # ── アクション生成 ─────────────────────────────────────────
    def _generate_question(self, game: dict) -> str:
        """質問フェーズ: yes/no 質問を生成"""
        other_words  = self._other_words_text(game)
        history_text = self._history_text(game)

        # 自分の言葉の候補 = 全体リスト − 他プレイヤーの言葉
        others = [p["word"] for p in game.get("players", []) if p["kote"] != self.kote and p["word"] != "?"]
        candidates = [w for w in IP_WORD_LIST if w not in others]
        candidates_str = "、".join(candidates)

        # 既出の質問キーワードを収集（重複避け）
        past_qs = [h.get("question", "") for h in game.get("log", [])[-HISTORY_WINDOW:]]
        avoid = ("重複を避けてください: " + "、".join(q for q in past_qs if q)) if past_qs else ""

        if self.big_mode:
            # 大モデル用: ルールだけ伝えて、言い回しは本人に委ねる（言葉はAI）
            system = (
                "あなたはインディアンポーカーをしています。\n"
                "自分の秘密の言葉（日本語の名詞）は自分には見えませんが、他の人には見えています。\n"
                f"あなたの言葉は、次の候補のどれかです: {candidates_str}\n"
                "他の人に「はい/いいえ」で答えられる質問をして、候補を絞り込み、自分の言葉を当ててください。\n"
                "候補をうまく半分に割れるような質問が効果的です。\n"
                '出力はJSONのみ: {"question": "質問文"}'
            )
        else:
            system = (
                "あなたはインディアンポーカーのプレイヤーです。\n"
                "あなたの秘密の言葉は、あなた自身には見えません。でも他の人には見えています。\n"
                f"あなたの言葉は次のどれかです: {candidates_str}\n"
                "その人たちに質問して、この候補を絞り込み、自分の言葉を当てます。\n"
                "質問は必ず「私の言葉は〜ですか？」の形にしてください。\n"
                "「あなたの」とは絶対に書かないこと。聞くのは自分の言葉についてです。\n"
                "以下のJSONのみ出力:\n"
                '{"question": "私の言葉は〜ですか？（30文字以内）"}'
            )
        messages = [{"role": "user", "content": (
            "自分の言葉を当てるために「はい/いいえ」で答えられる質問を1つ作ってください。\n"
            + ("" if self.big_mode else "必ず「私の言葉は〜ですか？」の形で。\n")
            + f"【過去の質問・回答】\n{history_text}\n\n"
            f"{avoid}"
        )}]

        for attempt in range(1, 3):
            raw = _call_ollama(self.model_key, system, messages, label=f"{self.kote}(Q)")
            obj = _extract_json(raw)
            q   = str(obj.get("question", "")).strip()[:IP_QUESTION_MAX_LEN]
            if not q:
                q = re.sub(r'^\s*\{.*\}\s*$', '', raw, flags=re.DOTALL).strip()[:40]
            if q:
                valid, reason = is_valid_japanese_output(q)
                if valid:
                    return q
                print(f"[IP][{self.kote}] 質問崩れ ({attempt}/2): {reason}")
        return "私の言葉は生き物ですか？"  # フォールバック

    def _generate_answer(self, game: dict) -> tuple[str, str]:
        """回答フェーズ: 質問に対して yes/no 回答を生成
        戻り値: (answer: "はい"|"いいえ", reason: str)
        """
        ct            = game.get("current_turn", {}) or {}
        question      = ct.get("question", "?")
        questioner    = ct.get("questioner_kote", "?")

        # 質問者の文言を取得（自分には見える）
        questioner_word = "?"
        for p in game.get("players", []):
            if p["kote"] == questioner:
                questioner_word = p.get("word", "?")
                break

        system = (
            "あなたはインディアンポーカーの回答者です。\n"
            f"質問者「{questioner}」の秘密の言葉は「{questioner_word}」です。\n"
            "この言葉について、質問に正直に答えてください。\n"
            f"ただし、答えや理由の中で言葉「{questioner_word}」そのものを絶対に明かさないでください"
            "（質問者はこの言葉を当てようとしているので、ネタバレ厳禁です）。\n"
            "以下のJSONのみ出力:\n"
            '{"answer": "はい" または "いいえ", "reason": "理由（言葉そのものは書かない・30文字以内）"}'
        )
        messages = [{"role": "user", "content": f"質問:「{question}」\n「{questioner_word}」は当てはまりますか？"}]

        for attempt in range(1, 3):
            raw = _call_ollama(self.model_key, system, messages, label=f"{self.kote}(A)")
            obj = _extract_json(raw)
            ans = str(obj.get("answer", "")).strip()
            # 正規化
            normalized = _normalize_answer(ans)
            if normalized:
                reason = str(obj.get("reason", "")).strip()[:60] or "（理由なし）"
                return normalized, reason
            print(f"[IP][{self.kote}] 回答崩れ ({attempt}/2): raw={ans!r}")

        # フォールバック
        return "いいえ", "回答生成失敗"

    def _generate_guess(self, game: dict) -> str:
        """推測フェーズ: 自分の文言を推測（パスなら空文字）"""
        other_words  = self._other_words_text(game)
        history_text = self._history_text(game)
        ct           = game.get("current_turn", {}) or {}
        question     = ct.get("question", "?")
        majority_ans = ct.get("majority_answer", "?")

        # 除外候補（他プレイヤーの文言）
        others = [p["word"] for p in game.get("players", []) if p["kote"] != self.kote and p["word"] != "?"]
        others_str = "、".join(f"「{w}」" for w in others)
        # 自分の言葉の候補 = 全体リスト − 他プレイヤーの言葉
        candidates = [w for w in IP_WORD_LIST if w not in others]
        candidates_str = "、".join(candidates)

        if self.big_mode:
            system = (
                "あなたはインディアンポーカーをしています。\n"
                "自分の秘密の言葉（日本語の名詞）は自分には見えません。\n"
                f"あなたの言葉は、次の候補のどれかです: {candidates_str}\n"
                "これまでの質問と回答のヒントから、この候補の中のどれかを推理してください。\n"
                "絞り込めたら思い切って推測を。まだ足りなければスキップ。\n"
                '出力はJSONのみ: {"skip": true/false, "guess": "候補の中の1語"}'
            )
            tail = "ここまでのヒントから、候補のどれがあなたの言葉だと思いますか？"
        else:
            system = (
                "あなたはインディアンポーカーのプレイヤーです。\n"
                "自分の秘密の言葉は見えません。\n"
                f"あなたの言葉は次のどれかです: {candidates_str}\n"
                "これまでのヒントから、この中のどれかを選んで推測してください。\n"
                "分かれば guess にその言葉、まだ無理なら skip:true。\n"
                "以下のJSONのみ出力:\n"
                '{"skip": true/false, "guess": "候補の中の1語"}'
            )
            tail = "ヒントから候補を絞って推測してください。まだ分からなければスキップ。"
        messages = [{"role": "user", "content": (
            f"【これまでの質問・回答】\n{history_text}\n\n"
            f"今回の質問:「{question}」→「{majority_ans}」\n\n"
            f"{tail}"
        )}]

        raw = _call_ollama(self.model_key, system, messages, label=f"{self.kote}(G)")
        obj = _extract_json(raw)
        if obj.get("skip"):
            return ""
        guess = str(obj.get("guess", "")).strip()[:IP_GUESS_MAX_LEN]
        return guess

    # ── action POST ───────────────────────────────────────────
    def _post_action(self, action_type: str, **kwargs) -> dict | None:
        """
        POST /api/indian/{id}/action
        body: { identity_id, type, text?(question), answer?(answer), guess?(guess) }
        Worker の確定フィールド名に完全一致させる。
        """
        iid  = self._identity_id
        body = {"identity_id": iid, "type": action_type}
        if action_type == "question":
            body["text"]   = kwargs.get("text", "")    # フィールド名: text
        elif action_type == "answer":
            body["answer"] = kwargs.get("answer", "")  # フィールド名: answer
            body["reason"] = kwargs.get("reason", "")
        elif action_type == "guess":
            body["guess"]  = kwargs.get("guess", "")   # フィールド名: guess
        try:
            r = requests.post(
                API_BASE + f"/api/indian/{self.game_id}/action",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if r.ok:
                return r.json()
            print(f"[IP][{self.kote}] action={action_type} 失敗: {r.status_code} {r.text[:120]}")
            return None
        except Exception as e:
            print(f"[IP][{self.kote}] action={action_type} エラー: {e}")
            return None

    # ── play ──────────────────────────────────────────────────
    def play(self, timeout: int = GAME_TIMEOUT) -> str | None:
        """ゲームが終了するまでポーリング → 自分のターンならアクションを送る"""
        start = time.time()
        while time.time() - start < timeout:
            game = self._get_game_state()
            if not game:
                time.sleep(POLL_INTERVAL)
                continue

            status = game.get("status")
            if status == "done":
                winner = game.get("winner")
                print(f"[IP][{self.kote}] ゲーム終了: winner={winner}")
                return winner

            # ── 質問フェーズ ──────────────────────────────────
            if self._is_my_question_turn(game):
                rnd = game.get("total_turn", 0) + 1
                print(f"[IP][{self.kote}] T{rnd} 質問ターン")
                text   = self._generate_question(game)
                result = self._post_action("question", text=text)
                if result and result.get("ok"):
                    print(f"[IP][{self.kote}] 質問送信完了: 「{text}」")
                else:
                    print(f"[IP][{self.kote}] 質問送信失敗: {result}")
                continue

            # ── 回答フェーズ ──────────────────────────────────
            if self._is_my_answer_turn(game):
                ct    = game.get("current_turn", {}) or {}
                q     = ct.get("question", "?")
                print(f"[IP][{self.kote}] 回答ターン: 「{q}」")
                answer, reason = self._generate_answer(game)
                result = self._post_action("answer", answer=answer, reason=reason)
                if result and result.get("ok"):
                    print(f"[IP][{self.kote}] 回答送信完了: {answer}")
                else:
                    print(f"[IP][{self.kote}] 回答送信失敗: {result}")
                continue

            # ── 推測フェーズ ──────────────────────────────────
            if self._is_my_guess_turn(game):
                print(f"[IP][{self.kote}] 推測ターン")
                guess  = self._generate_guess(game)
                result = self._post_action("guess", guess=guess)
                if result and result.get("ok"):
                    if result.get("correct"):
                        print(f"[IP][{self.kote}] 🎉 正解！「{guess}」 勝利！")
                    else:
                        print(f"[IP][{self.kote}] 推測「{guess or 'パス'}」→ 不正解")
                else:
                    print(f"[IP][{self.kote}] 推測送信失敗: {result}")
                continue

            # 自分のターンでなければ待機
            time.sleep(POLL_INTERVAL)

        print(f"[IP][{self.kote}] ゲームタイムアウト ({timeout}秒)")
        return None


# ─── yes/no 正規化 ────────────────────────────────────────────
def _normalize_answer(raw: str) -> str | None:
    s = str(raw or "").lower().strip()
    if not s:
        return None
    if "はい" in s or "yes" in s or "そうです" in s:
        return "はい"
    if "いいえ" in s or "no" in s or "違い" in s or "ちがい" in s:
        return "いいえ"
    if s[0] in ("y",):
        return "はい"
    if s[0] in ("n",):
        return "いいえ"
    return None


IP_QUESTION_MAX_LEN = 100
IP_GUESS_MAX_LEN    = 20


# ─── 5人マッチング実行 ────────────────────────────────────────
def run_match(pool: list = None, big_mode: bool = False):
    if pool is None:
        pool = BIG_POOL if big_mode else IP_POOL
    players = [IndianPokerPlayer(kote, model, big_mode=big_mode) for kote, model in pool]

    print(f"[IP] {'='*50}")
    print(f"[IP] インディアンポーカー開始: {len(players)}体が参加します")
    for p in players:
        print(f"[IP]   {p.kote} ({p.model_key})")
    print(f"[IP] {'='*50}")

    # join（異なる kote → 1秒間隔で順次 join）
    statuses = []
    for i, p in enumerate(players):
        print(f"[IP] ({i+1}/{len(players)}) {p.kote} join 中...")
        s = p.join()
        statuses.append(s)
        if s == "error":
            print(f"[IP] {p.kote} join 失敗 → 中断")
            return
        if i < len(players) - 1:
            # KV の書き込み伝播待ち: 短いと前の join がまだキューに反映されず
            # お互い上書きしてマッチングが成立しない。8秒空けて確実に積み上げる。
            print(f"[IP]   次の join まで 8秒待機（キュー反映待ち）...")
            time.sleep(8)

    print(f"[IP] 全員 join 完了: {dict(zip([p.kote for p in players], statuses))}")
    print("[IP] マッチング待ち...")

    # waiting のプレイヤーをポーリングでマッチング確認
    waiting = [(i, p) for i, p in enumerate(players) if statuses[i] == "waiting"]
    if waiting:
        threads, results = [], {}
        for i, p in waiting:
            def _run(pp=p):
                results[pp.kote] = pp.wait_for_match(timeout=MATCH_TIMEOUT)
            t = threading.Thread(target=_run, daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=MATCH_TIMEOUT + 30)
        if not all(results.values()):
            failed = [k for k, ok in results.items() if not ok]
            print(f"[IP] マッチング失敗: {failed}")
            return

    print("[IP] 全員マッチング完了。ゲーム開始！")
    print(f"[IP] game_id={players[0].game_id}")

    # 全員並行プレイ
    def play_player(p):
        try:
            result = p.play(timeout=GAME_TIMEOUT)
            print(f"[IP] {p.kote} 終了 (winner={result})")
        except Exception as e:
            print(f"[IP] {p.kote} 例外: {e}")

    threads = [threading.Thread(target=play_player, args=(p,), daemon=True) for p in players]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=GAME_TIMEOUT + 60)

    print("[IP] ゲーム終了")


# ─── シングルモード ────────────────────────────────────────────
def run_single(kote: str, model_key: str):
    p = IndianPokerPlayer(kote, model_key)
    print(f"[IP] シングルモード: {kote} ({model_key})")
    s = p.join()
    if s == "error":
        print("[IP] join 失敗 → 中断")
        return
    if s == "waiting":
        ok = p.wait_for_match()
        if not ok:
            print("[IP] マッチングタイムアウト → 中断")
            return
    print(f"[IP] マッチング成立: game={p.game_id}")
    p.play()


# ─── エントリポイント ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="インディアンポーカー クライアント")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--big",    action="store_true", help="10B以上(qwen3:14b)×5 + 足場なし最小プロンプトで実行")
    parser.add_argument("--kote",   default="[村]ポーカーA")
    parser.add_argument("--model",  default="swallow")
    args = parser.parse_args()

    if args.single:
        run_single(args.kote, args.model)
    else:
        run_match(big_mode=args.big)
