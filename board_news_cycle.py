"""
掲示板ニュースサイクル — 日次自動実行スクリプト

① 外部AIニュース収集（arXiv / HN / Google News / Papers with Code）
   → ニュースなし（0件）なら終了（投稿なし）
③ 村会議：今週のAIニュースについて議論
   ※ 会議まとめは村内（JSON）に記録するのみ。板には投稿しない。

実行:
  python board_news_cycle.py             # 全ステップ
  python board_news_cycle.py --step news  # ①のみ
  python board_news_cycle.py --step meet  # ③のみ（キャッシュ使用）

変更履歴:
  2026-10-19 村会議の結果を板へ投稿するのを停止（議事録は村内で完結）
             ニュース取得件数を3→1に削減（KV書き込み枠の節約）
"""

import sys
import time
import datetime
import hashlib
import json
import os
import requests

sys.stdout.reconfigure(encoding="utf-8")

# ── パス設定 ──────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import ai_news_fetcher
import village_learning
import village_chat

# ── 設定 ─────────────────────────────────────────────
API_BASE      = "https://fnoita-board.mogumogu7la001.workers.dev"
ADMIN_SECRET  = os.environ.get("BOARD_ADMIN_SECRET", "")
MEETING_TURNS = 13             # 全員1回発言
NEWS_COUNT    = 1              # 1日あたりのニュース取得件数（KV節約のため1に削減）

# 板への投稿: 無効（村会議の結果は村内で完結させる）
# 有効に戻す場合は True に変更する
BOARD_POST_ENABLED = False
POST_CAT           = "f08"    # 投稿先カテゴリ（復活時用）

# ── 定数 ─────────────────────────────────────────────

def _admin_token():
    h = hashlib.sha256(ADMIN_SECRET.encode()).hexdigest()
    d = datetime.date.today().strftime("%Y%m%d")
    return hashlib.sha256((h + d).encode()).hexdigest()

def _new_id():
    import uuid
    return uuid.uuid4().hex[:14]


# ═══════════════════════════════════════════════════
# ① 外部AIニュース収集
# ═══════════════════════════════════════════════════

def step_fetch_news(num: int = 3) -> dict:
    print("\n[CYCLE] ① 外部AIニュース収集")
    result = ai_news_fetcher.run(num_select=num)
    items = result.get("items", [])
    print(f"  → {len(items)}件収集・選別完了")
    for it in items:
        print(f"    ・{it.get('title', '')[:60]}")
    return result


# ═══════════════════════════════════════════════════
# ② ボード観察学習
# ═══════════════════════════════════════════════════

def step_board_observe() -> list:
    print("\n[CYCLE] ② 掲示板観察学習（担当メンバー）")
    results = village_learning.learn_board_all()
    for r in results:
        status = r.get("status")
        role   = r.get("role", "")
        if status == "learned":
            print(f"  ✓ {role}: {r.get('summary','')[:60]}")
        elif status == "empty":
            print(f"  - {role}: 投稿なし")
        else:
            print(f"  ? {role}: {status}")
    return results


# ═══════════════════════════════════════════════════
# ③ 村会議
# ═══════════════════════════════════════════════════

def step_village_meeting(news_post_text: str) -> dict:
    print("\n[CYCLE] ③ 村会議開始")

    topic = (
        f"今週のAIニュースと掲示板の情報をもとに、f05（掲示板の掲示板）に投稿する内容を決めよう。\n\n"
        f"【今週収集したAIニュース（案）】\n{news_post_text[:600]}"
    )

    # 会議スタート（1ターン目も自動実行）
    result = village_chat.start_discussion(topic)
    _print_turn(result)

    # 残りのターンを自動で回す
    for i in range(MEETING_TURNS - 1):
        time.sleep(2)
        result = village_chat.next_turn()
        _print_turn(result)
        if result.get("status") == "error":
            print(f"  [WARN] ターンエラー: {result.get('message')}")

    # まとめ
    time.sleep(1)
    print("\n[CYCLE] 会議まとめ生成中...")
    summary_result = village_chat.finish_discussion()
    summary = summary_result.get("summary", "")
    print(f"  まとめ: {summary[:150]}")
    return summary_result


def _print_turn(result: dict):
    turn = result.get("turn") or {}
    persona = turn.get("persona") or turn.get("role", "?")
    msg = (turn.get("message") or "")[:80]
    print(f"  [{persona}] {msg}")


# ═══════════════════════════════════════════════════
# ④ f05に投稿
# ═══════════════════════════════════════════════════

def step_post_to_board(news_result: dict, meeting_summary: str) -> bool:
    print("\n[CYCLE] ④ f05に投稿")

    today = datetime.date.today().strftime("%Y/%m/%d")
    token = _admin_token()
    h = {"Content-Type": "application/json", "X-Admin-Token": token}

    # 投稿本文：ニュース + 会議まとめ
    news_text = news_result.get("post_text", "")
    body = f"{news_text}\n\n━━━━━━━━━━━━━━━━━\n【会議まとめ】\n{meeting_summary[:500]}"

    tid = _new_id()
    pid = _new_id()
    now = int(time.time() * 1000)

    data = {
        "thread": {"id": tid, "title": f"📰 今週のAIニュース（{today}）", "kote": "管理AI", "at": now},
        "post":   {"id": pid, "kote": "管理AI", "body": body, "at": now},
    }
    r = requests.post(f"{API_BASE}/api/threads/{POST_CAT}", headers=h, json=data, timeout=15)
    if r.ok:
        print(f"  ✅ 投稿完了 → {POST_CAT} / tid={tid}")
        return True
    else:
        print(f"  ❌ 投稿失敗: {r.status_code} {r.text[:100]}")
        return False


# ═══════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════

def run_full_cycle():
    print(f"\n{'='*50}")
    print(f"[CYCLE] 掲示板ニュースサイクル開始 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    # ① ニュース収集（1件のみ）
    news_result = step_fetch_news(num=NEWS_COUNT)

    # ニュースなしなら終了
    if not news_result.get("items"):
        print("\n[CYCLE] ニュースなし — 本日は投稿をスキップします")
        return

    # ③ 会議
    meeting = step_village_meeting(news_result.get("post_text", ""))
    meeting_summary = meeting.get("summary", "")

    # ④ 板への投稿（現在は無効）
    if BOARD_POST_ENABLED:
        step_post_to_board(news_result, meeting_summary)
    else:
        print("\n[CYCLE] 板への投稿はスキップ（BOARD_POST_ENABLED=False）")
        print(f"  まとめ（村内記録）: {meeting_summary[:100]}")

    # レポート再生成
    try:
        import subprocess
        viewer = r"C:\Users\USER\.claude\projects\C--ai-agent\score_viewer.py"
        subprocess.run(["python", viewer], timeout=30, check=False)
        print("[CYCLE] score_report.html を更新しました")
    except Exception as e:
        print(f"[CYCLE] レポート生成スキップ: {e}")

    print(f"\n[CYCLE] 完了 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    step = None
    if "--step" in sys.argv:
        idx = sys.argv.index("--step")
        if idx + 1 < len(sys.argv):
            step = sys.argv[idx + 1]

    if step == "news":
        step_fetch_news()
    elif step == "meet":
        # キャッシュからニュースを読んで会議だけ実行
        cache_file = os.path.join(_DIR, "ai_news_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
            items = cache.get("items", [])
            post_text = ai_news_fetcher.build_board_post(
                ai_news_fetcher.select_and_summarize(items, 3)
            ) if items else "（キャッシュ読み込み失敗）"
        else:
            post_text = "（ニュースキャッシュなし）"
        meeting = step_village_meeting(post_text)
        step_post_to_board({"post_text": post_text}, meeting.get("summary", ""))
    else:
        run_full_cycle()
