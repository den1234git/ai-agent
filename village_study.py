"""
村メンバー個別学習スクリプト（自由学習・休み時間）

1日1体がローテーションで、自分でテーマを選んで自由に学習する。
会議が始まったら中断して参加。会議終了後に再開。
学習状況は study_status.json に書き出し（UIから参照可能）。

実行:
  python village_study.py          # 今日の担当が学習
  python village_study.py --role granite  # 指定メンバーで実行
"""

import sys
import os
import json
import datetime
import time

sys.stdout.reconfigure(encoding="utf-8")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_DIR, ".env"))

import urllib.request
import urllib.parse
import village_learning
import memory_store
from ai import ask_ai


# ── マルチソース検索 ──────────────────────────────
def _fetch_duckduckgo(query: str) -> str:
    """DuckDuckGo Instant Answer API（無料・APIキー不要）"""
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VillageStudyBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"][:400])
        for rt in data.get("RelatedTopics", [])[:3]:
            text = rt.get("Text", "")
            if text:
                parts.append(text[:150])
        return "\n".join(parts)
    except Exception as e:
        print(f"  [DDG] 失敗: {e}")
        return ""


def _fetch_semantic_scholar(query: str) -> str:
    """Semantic Scholar API（無料・月1万回）"""
    try:
        params = urllib.parse.urlencode({"query": query, "limit": 3, "fields": "title,abstract"})
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VillageStudyBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        parts = []
        for p in data.get("data", [])[:2]:
            title = p.get("title", "")
            abstract = (p.get("abstract") or "")[:200]
            if title:
                parts.append(f"[論文] {title}: {abstract}")
        return "\n".join(parts)
    except Exception as e:
        print(f"  [S2] 失敗: {e}")
        return ""


def _fetch_openalex(query: str) -> str:
    """OpenAlex API（完全無料・無制限）"""
    try:
        params = urllib.parse.urlencode({"search": query, "per-page": 3,
                                          "select": "title,abstract_inverted_index"})
        url = f"https://api.openalex.org/works?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VillageStudyBot/1.0 (mailto:study@village.ai)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        parts = []
        for w in data.get("results", [])[:2]:
            title = w.get("title", "")
            if title:
                parts.append(f"[OpenAlex] {title}")
        return "\n".join(parts)
    except Exception as e:
        print(f"  [OA] 失敗: {e}")
        return ""


def _fetch_tavily(query: str) -> str:
    """Tavily Web検索（月1000回、arXivで取れない場合の補助）"""
    try:
        from web_search import tavily_search
        result = tavily_search(query, max_results=3)
        if not result.get("available"):
            return ""
        parts = []
        if result.get("answer"):
            parts.append(result["answer"][:200])
        for r in result.get("results", [])[:2]:
            parts.append(f"[Web] {r.get('title','')}: {r.get('content','')[:150]}")
        return "\n".join(parts)
    except Exception as e:
        print(f"  [Tavily] 失敗: {e}")
        return ""


def fetch_all_sources(topic: str) -> tuple[str, str]:
    """
    全ソースから情報を収集して結合する。
    返値: (combined_text, source_url)
    """
    print(f"  🔍 検索中: {topic}")
    arxiv_text, arxiv_url = village_learning._fetch_arxiv(topic)
    time.sleep(0.5)
    wiki_text = village_learning._fetch_wikipedia(topic)
    time.sleep(0.5)
    ddg_text = _fetch_duckduckgo(topic)
    time.sleep(0.5)
    s2_text = _fetch_semantic_scholar(topic)
    time.sleep(0.5)
    oa_text = _fetch_openalex(topic)

    # arXivで十分な情報が取れなかった場合のみTavilyを使う
    if not arxiv_text.strip():
        time.sleep(0.5)
        tavily_text = _fetch_tavily(topic)
    else:
        tavily_text = ""

    combined = "\n".join(filter(None, [arxiv_text, wiki_text, ddg_text, s2_text, oa_text, tavily_text]))
    return combined[:2000], arxiv_url

# ローテーション管理ファイル
_ROTATION_FILE  = os.path.join(_DIR, "study_rotation.json")
_STATUS_FILE    = os.path.join(_DIR, "study_status.json")

ALL_ROLES = [
    "router", "llama3_2", "granite", "swallow", "lfm_jp",
    "llama3_1", "qwen3", "qwen_coder_14b", "sougoshitsumon",
    "gemma3", "lfm_8b", "qwen_coder_7b", "lfm_1b"
]

TOPICS_PER_SESSION = 20


# ── ローテーション ──────────────────────────────
def _load_rotation() -> dict:
    if os.path.exists(_ROTATION_FILE):
        with open(_ROTATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"index": 0, "last_date": ""}


def _save_rotation(data: dict):
    with open(_ROTATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_today_member(offset: int = 0) -> str:
    rot = _load_rotation()
    today = datetime.date.today().isoformat()
    if rot["last_date"] != today:
        rot["index"] = (rot["index"] + 1) % len(ALL_ROLES)
        rot["last_date"] = today
        _save_rotation(rot)
    idx = (rot["index"] + offset) % len(ALL_ROLES)
    return ALL_ROLES[idx]


# ── ステータス管理 ──────────────────────────────
def _write_status(role: str, topic: str, count: int, total: int, state: str = "studying"):
    """UIから参照できるステータスファイルに現在状況を書き込む。"""
    data = {
        "state":    state,   # studying / meeting / done / idle
        "role":     role,
        "topic":    topic,
        "count":    count,
        "total":    total,
        "updated":  datetime.datetime.now().isoformat(),
    }
    try:
        with open(_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _clear_status():
    _write_status("", "", 0, 0, state="idle")


# ── 自由テーマ選択 ──────────────────────────────
def _pick_topic(role: str, learned_titles: list[str]) -> str:
    """
    そのメンバーに「今日何を調べたい？」と聞いて、テーマを自分で選ばせる。
    """
    recent = "、".join(learned_titles[-5:]) if learned_titles else "（なし）"
    prompt = f"""あなたは今、自由な休み時間です。最近学んだこと：{recent}

arXivやWikipediaで調べられる「今日学びたいテーマ」を1つ選んでください。
何でもOKです。あなたが本当に興味を持っていることを選んでください。

以下のJSONのみ返してください：
{{"topic": "テーマ名（英語または日本語、30文字以内）", "reason": "選んだ理由（20文字以内）"}}"""
    try:
        result = ask_ai(prompt, model=role)
        topic = (result.get("topic") or "").strip()
        reason = (result.get("reason") or "").strip()
        if topic:
            print(f"  📚 {role} が選んだテーマ: 「{topic}」 — {reason}")
            return topic
    except Exception as e:
        print(f"  [WARN] テーマ選択失敗: {e}")
    # フォールバック: ランダムなarXiv分野から
    import random
    fallbacks = ["AI safety", "quantum computing", "neuroscience", "climate change",
                 "robotics", "materials science", "economics", "linguistics"]
    return random.choice(fallbacks)


# ── 会議チェック ──────────────────────────────
def _is_meeting_active() -> bool:
    try:
        import village_chat
        return village_chat._state.get("active", False)
    except Exception:
        return False


def _wait_for_meeting_end(role: str, topic: str, count: int, total: int):
    """会議中は待機。UIには「会議参加中」を表示。"""
    _write_status(role, topic, count, total, state="meeting")
    print(f"\n[STUDY] 会議が始まりました。{role} は一時中断して参加します...")
    while _is_meeting_active():
        time.sleep(5)
    print(f"[STUDY] 会議終了。学習を再開します。")
    _write_status(role, topic, count, total, state="studying")


# ── メイン学習ループ ──────────────────────────────
def run_study(role: str, topics: int = TOPICS_PER_SESSION):
    print(f"\n{'='*50}")
    print(f"[STUDY] 自由学習タイム開始")
    print(f"  担当: {role}")
    print(f"  日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    learned_titles: list[str] = []
    learned = stored = 0

    for i in range(topics):
        # 会議中なら待機
        if _is_meeting_active():
            _wait_for_meeting_end(role, "（待機中）", i, topics)

        # テーマを自分で選ぶ
        topic = _pick_topic(role, learned_titles)
        _write_status(role, topic, i + 1, topics, state="studying")
        print(f"\n[STUDY] ({i+1}/{topics}) {role} → 「{topic}」を調べ中...")

        try:
            # 全ソースから取得（arXiv・Wikipedia・DuckDuckGo・S2・OpenAlex・Tavily）
            raw, arxiv_url = fetch_all_sources(topic)

            if not raw.strip():
                print(f"  - 情報が取得できませんでした")
                continue

            summary = village_learning._summarize(role, topic, raw)
            if not summary:
                summary = raw[:150]

            learned_titles.append(topic)
            learned += 1
            print(f"  ✓ 要約: {summary[:60]}")

            # ChromaDB に保存（掲示板には投稿しない）
            doc_id = memory_store.save_village_knowledge(
                role=role,
                title=topic,
                summary=summary,
                source_url=arxiv_url,
                doc_type="free_study",
            )
            stored += 1
            print(f"  → ChromaDB保存: {doc_id}")

        except Exception as e:
            print(f"  ✗ エラー: {e}")

        time.sleep(2)

    _write_status(role, "（完了）", topics, topics, state="done")
    print(f"\n{'='*50}")
    print(f"[STUDY] 自由学習 完了")
    print(f"  学習: {learned} / {topics}")
    print(f"  ChromaDB保存: {stored} 件")
    print(f"{'='*50}")

    return {"role": role, "learned": learned, "stored": stored}


MEMBERS_PER_DAY = 4  # 1日に学習するメンバー数

if __name__ == "__main__":
    role = None
    if "--role" in sys.argv:
        idx = sys.argv.index("--role")
        if idx + 1 < len(sys.argv):
            role = sys.argv[idx + 1]

    if role:
        if role not in ALL_ROLES:
            print(f"[STUDY] 不明なメンバー: {role}")
            sys.exit(1)
        run_study(role)
    else:
        for i in range(MEMBERS_PER_DAY):
            r = get_today_member(offset=i)
            print(f"[STUDY] {i+1}/{MEMBERS_PER_DAY} 担当: {r}")
            run_study(r)
            if i < MEMBERS_PER_DAY - 1:
                print("[STUDY] 次のメンバーに移行...")
                time.sleep(5)
