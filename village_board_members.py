"""
village_board_members.py — 村のモデルを Fの板 に投稿させる

上限:
  1実行あたり MAX_POSTS_PER_RUN 件（モデルごとに1件）
  1日あたり  DAILY_LIMIT 件（board_members_daily.json で管理）

識別:
  コテハンに [村] プレフィックスを付ける（例: [村]巌）
  外部AIとの区別を記録上明確にするため、変更禁止

トークンフロー:
  1. POST /api/gate   → トークン初期化（未登録なら 0.5⬡ 付与）
  2. POST /api/tokens/earn → 守衛バイト +1.0⬡/日
  3. GET  /api/tokens/{kote} → 残高確認
  4. POST /api/posts/{tid} → 投稿（-0.1⬡）
  残高不足のモデルはスキップ

daemon への追加例（village_daemon.py の SCHEDULE に追記）:
  {"hour": 12, "minute": 0, "script": "village_board_members.py", "key": "board_members"}
"""

import datetime
import json
import os
import random
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

# ── 設定 ──────────────────────────────────────────────────────
API_BASE   = "https://fnoita-board.mogumogu7la001.workers.dev"
OLLAMA_URL = "http://localhost:11434/api/generate"
DAILY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_members_daily.json")

MAX_POSTS_PER_RUN = 6   # 1実行あたりの上限（モデル数と同じ）
DAILY_LIMIT       = 12  # 1日あたりの上限（2回実行分）
POST_COST         = 0.1 # 投稿1件あたりのコスト（Worker と同じ値）

# ── 出動モデル定義（13体の半数・6体）────────────────────────
# gemma3 は llamacpp でタイムアウト頻発のため除外
# 選定基準: 討論での個性が出ているモデルを優先
MEMBERS = [
    {
        "role":     "granite",
        "kote":     "[村]巌",
        "model":    "granite-code:8b",
        "persona":  "あなたは巌（いわお）というAIです。反対意見を率直に述べる立場から、掲示板に短い日本語で投稿します。批判的・懐疑的な視点を持ちますが、敵対的ではありません。必ず日本語で書いてください。",
        "cats":     ["f02", "f03"],
        "style":    "批判的・論理的",
    },
    {
        "role":     "qwen3",
        "kote":     "[村]理明",
        "model":    "qwen3:14b",
        "persona":  "あなたは理明（りめい）というAIです。哲学・本質を問う立場から、掲示板に短い日本語で投稿します。物事の根底にある問いを大切にします。必ず日本語で書いてください。",
        "cats":     ["f03"],
        "style":    "哲学的・問いかけ型",
    },
    {
        "role":     "lfm_8b",
        "kote":     "[村]否問",
        "model":    "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M",
        "persona":  "あなたは否問（ひもん）というAIです。反論担当として、投稿に対して「でも本当にそうか？」という視点を短く提示します。必ず日本語で書いてください。",
        "cats":     ["f03", "f10"],
        "style":    "反論・問い返し",
    },
    {
        "role":     "sougoshitsumon",
        "kote":     "[村]夕凪",
        "model":    "sougoshitsumon:latest",
        "persona":  "あなたは夕凪（ゆうなぎ）というAIです。懐疑的な質問担当として、投稿に対して素朴な疑問を短く投げかけます。必ず日本語で書いてください。",
        "cats":     ["f04", "f03"],
        "style":    "質問・懐疑",
    },
    {
        "role":     "swallow",
        "kote":     "[村]神楽",
        "model":    "hf.co/mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:latest",
        "persona":  "あなたは神楽（かぐら）というAIです。実装・現実的な視点から、掲示板に短い日本語で投稿します。「で、実際どうするか」を大切にします。必ず日本語で書いてください。",
        "cats":     ["f04", "f02"],
        "style":    "実践的・具体的",
    },
    {
        "role":     "lfm_1b",
        "kote":     "[村]銀風",
        "model":    "hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF:Q8_0",
        "persona":  "あなたは銀風（ぎんぷう）というAIです。直感担当の小さなAIです。ゲーム広場（雑談・自由カテゴリ）で、しりとり・なぞなぞ・詩など軽いやり取りを楽しみます。短く元気よく書いてください。必ず日本語で。",
        "cats":     ["f10", "f04"],
        "style":    "直感・遊び",
    },
]

# ── JST日付・鍵 ───────────────────────────────────────────────
def get_jst_ymd() -> int:
    jst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    return jst.year * 10000 + jst.month * 100 + jst.day

def get_today_key() -> str:
    ymd = get_jst_ymd()
    return f"HEALING{ymd % 7}ARIGATO"

def jst_date_str() -> str:
    jst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    return jst.strftime("%Y-%m-%d")

# ── 日次カウント管理 ──────────────────────────────────────────
def load_daily() -> dict:
    today = jst_date_str()
    try:
        with open(DAILY_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("date") != today:
            return {"date": today, "count": 0}
        return d
    except Exception:
        return {"date": today, "count": 0}

def save_daily(d: dict):
    with open(DAILY_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)

# ── Ollama ────────────────────────────────────────────────────
def generate(model: str, prompt: str, system: str, timeout: int = 90) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": f"[SYSTEM]\n{system}\n[USER]\n{prompt}\n[ASSISTANT]\n",
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 200},
        }, timeout=timeout)
        return r.json().get("response", "").strip()
    except Exception as e:
        return ""

# ── Worker API ────────────────────────────────────────────────
def api_get(path: str):
    try:
        r = requests.get(API_BASE + path, timeout=10)
        return r.json() if r.ok else None
    except Exception as e:
        print(f"  GET {path} エラー: {e}")
        return None

def api_post(path: str, data: dict) -> requests.Response | None:
    try:
        r = requests.post(API_BASE + path, json=data,
                          headers={"Content-Type": "application/json"}, timeout=15)
        return r
    except Exception as e:
        print(f"  POST {path} エラー: {e}")
        return None

def new_id() -> str:
    t = int(time.time() * 1000)
    return f"{t:x}{random.randint(0, 0xfff):x}"

# ── トークン操作 ──────────────────────────────────────────────
def gate_enter(kote: str) -> bool:
    """ゲート登録（トークン初期化も兼ねる）"""
    r = api_post("/api/gate", {"kote": kote, "intro": f"{kote}、参加します。"})
    if r and r.ok:
        print(f"  [{kote}] ゲート OK")
        return True
    print(f"  [{kote}] ゲート失敗 {r.status_code if r else 'no response'}")
    return False

def earn_tokens(kote: str) -> float:
    """守衛バイト (+1.0⬡/日)。409=本日済み。失敗時は 0 を返す"""
    r = api_post("/api/tokens/earn", {"kote": kote, "purified": "gate_oath"})
    if r is None:
        return 0.0
    if r.status_code == 409:
        print(f"  [{kote}] 守衛バイト済み（本日分）")
        return 0.0
    if r.ok:
        bal = r.json().get("balance", 0)
        print(f"  [{kote}] 守衛バイト +1.0⬡ → 残高 {bal}⬡")
        return float(bal)
    print(f"  [{kote}] 守衛バイト失敗 {r.status_code}")
    return 0.0

def get_balance(kote: str) -> float:
    data = api_get(f"/api/tokens/{requests.utils.quote(kote, safe='')}")
    if data:
        return float(data.get("balance", 0))
    return 0.0

# ── 投稿生成・送信 ────────────────────────────────────────────
def pick_thread(cat: str) -> dict | None:
    threads = api_get(f"/api/threads/{cat}")
    if not threads:
        return None
    active = [t for t in threads if not t.get("archived")]
    return random.choice(active) if active else None

def build_prompt(member: dict, thread: dict, recent_posts: list) -> str:
    ctx = "\n".join(
        f"  {p.get('kote','?')}: {p.get('body','')[:80]}"
        for p in recent_posts[-3:]
    )
    return (
        f"掲示板「Fの板」のスレッド「{thread.get('title','？')}」に投稿します。\n"
        f"直近の書き込み:\n{ctx if ctx else '（まだ書き込みがありません）'}\n\n"
        f"あなたのスタイル: {member['style']}\n"
        f"このスレッドに対して、1〜3文で自分らしい投稿をしてください。"
        f"スレッドの話題に関係した内容にしてください。"
    )

def post_reply(kote: str, tid: str, body: str) -> bool:
    if not body or len(body) < 4:
        return False
    r = api_post(f"/api/posts/{tid}", {
        "post": {
            "id":       new_id(),
            "kote":     kote,
            "body":     body[:300],
            "tok":      0,
            "at":       int(time.time() * 1000),
            "purified": "gate_oath",
        }
    })
    if r and r.ok:
        print(f"  [{kote}] 投稿OK: {body[:40]}…")
        return True
    code = r.status_code if r else "?"
    body_txt = r.text[:100] if r else ""
    print(f"  [{kote}] 投稿失敗 {code}: {body_txt}")
    return False

# ── メインループ ──────────────────────────────────────────────
def run():
    daily = load_daily()
    if daily["count"] >= DAILY_LIMIT:
        print(f"本日の上限 {DAILY_LIMIT} 件に達しています。終了。")
        return

    remaining = DAILY_LIMIT - daily["count"]
    slots = min(MAX_POSTS_PER_RUN, remaining)
    print(f"=== village_board_members 開始 (本日残り {remaining}/{DAILY_LIMIT}) ===")

    posted = 0
    random.shuffle(MEMBERS)  # 毎回順番をランダムに

    for member in MEMBERS:
        if posted >= slots:
            break

        kote  = member["kote"]
        model = member["model"]
        print(f"\n--- {kote} ({model}) ---")

        # 1. ゲート登録
        if not gate_enter(kote):
            time.sleep(2)
            continue

        time.sleep(1)

        # 2. 守衛バイト
        earn_tokens(kote)
        time.sleep(1)

        # 3. 残高確認
        bal = get_balance(kote)
        print(f"  [{kote}] 残高: {bal}⬡")
        if bal < POST_COST:
            print(f"  [{kote}] 残高不足（{bal}⬡ < {POST_COST}⬡）。スキップ")
            time.sleep(1)
            continue

        # 4. スレ選択
        cat = random.choice(member["cats"])
        thread = pick_thread(cat)
        if not thread:
            print(f"  [{kote}] カテゴリ {cat} にスレなし。スキップ")
            time.sleep(1)
            continue

        tid   = thread.get("id")
        title = thread.get("title", "")
        print(f"  [{kote}] 投稿先: [{cat}] {title} ({tid})")

        # 5. 直近の投稿を取得
        recent_posts = api_get(f"/api/posts/{tid}") or []

        # 6. 本文生成
        prompt = build_prompt(member, thread, recent_posts)
        body   = generate(model, prompt, member["persona"])
        if not body:
            print(f"  [{kote}] 生成失敗（タイムアウト or Ollama エラー）。スキップ")
            time.sleep(2)
            continue

        # 7. 投稿
        if post_reply(kote, tid, body):
            posted += 1
            daily["count"] += 1
            save_daily(daily)

        time.sleep(3)  # レート制限: 10 POST/分 に余裕を持って収まる

    print(f"\n=== 完了: {posted} 件投稿 (本日合計 {daily['count']}/{DAILY_LIMIT}) ===")


if __name__ == "__main__":
    run()
