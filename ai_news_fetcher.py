"""
AIニュース収集エンジン。

4ソースから最新AI関連ニュースを取得し、
swallowに選別・要約させて掲示板投稿用テキストを生成する。

ソース：
  1. arXiv RSS       - 最新AI論文（cs.AI / cs.LG）
  2. Hacker News API - 技術者目線のホット話題
  3. Google News RSS - 日本語AIニュース
  4. Papers with Code - 実装付き論文

APIキー不要・無料。
"""

import json
import time
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os

from ai import ask_ai

_BASE = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_BASE, "ai_news_cache.json")

ARXIV_RSS_URL   = "https://rss.arxiv.org/rss/cs.AI"
HN_API_URL      = "https://hn.algolia.com/api/v1/search"
GNEWS_RSS_URL   = "https://news.google.com/rss/search?q=AI%20%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD%20LLM&hl=ja&gl=JP&ceid=JP:ja"
PWC_API_URL     = "https://paperswithcode.com/api/v1/papers/"
TIMEOUT         = 20
HEADERS         = {"User-Agent": "FnoItaNewsBot/1.0 (bulletin board news aggregator)"}


# ─────────────────────────────────────────
# ① arXiv RSS（論文）
# ─────────────────────────────────────────

def fetch_arxiv(max_items: int = 3) -> list[dict]:
    try:
        req = urllib.request.Request(ARXIV_RSS_URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
        root = ET.fromstring(raw)
        ns = {"": "http://www.w3.org/2005/Atom"}
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            desc  = (item.findtext("description") or "").strip()[:200]
            if title and link:
                items.append({"source": "arXiv", "title": title, "url": link, "summary": desc})
        return items
    except Exception as e:
        print(f"[NEWS] arXiv取得失敗: {e}")
        return []


# ─────────────────────────────────────────
# ② Hacker News（業界ニュース）
# ─────────────────────────────────────────

def fetch_hackernews(max_items: int = 3) -> list[dict]:
    try:
        params = urllib.parse.urlencode({
            "query": "AI LLM machine learning",
            "tags": "story",
            "hitsPerPage": max_items * 2,
            "numericFilters": f"created_at_i>{int(time.time()) - 7 * 86400}",  # 直近7日
        })
        url = f"{HN_API_URL}?{params}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = []
        for hit in data.get("hits", [])[:max_items]:
            title = (hit.get("title") or "").strip()
            url   = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            pts   = hit.get("points", 0)
            if title:
                items.append({"source": "Hacker News", "title": title, "url": url,
                              "summary": f"スコア: {pts}pt"})
        return items
    except Exception as e:
        print(f"[NEWS] HN取得失敗: {e}")
        return []


# ─────────────────────────────────────────
# ③ Google News RSS（日本語ニュース）
# ─────────────────────────────────────────

def fetch_google_news(max_items: int = 3) -> list[dict]:
    try:
        req = urllib.request.Request(GNEWS_RSS_URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
        root = ET.fromstring(raw)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title  = (item.findtext("title") or "").strip()
            link   = (item.findtext("link") or "").strip()
            source = (item.findtext("source") or "Google News").strip()
            if title and link:
                items.append({"source": f"Google News ({source})", "title": title,
                              "url": link, "summary": ""})
        return items
    except Exception as e:
        print(f"[NEWS] Google News取得失敗: {e}")
        return []


# ─────────────────────────────────────────
# ④ Papers with Code（実装付き論文）
# ─────────────────────────────────────────

def fetch_papers_with_code(max_items: int = 3) -> list[dict]:
    """arXiv cs.LG（機械学習）RSSで代替。PwC APIはHTML返却のため使用不可。"""
    try:
        params = urllib.parse.urlencode({
            "search_query": "cat:cs.LG",
            "max_results": max_items,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"https://export.arxiv.org/api/query?{params}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = []
        for e in root.findall("atom:entry", ns)[:max_items]:
            title   = (e.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            link_el = e.find("atom:id", ns)
            link    = (link_el.text or "").strip() if link_el is not None else ""
            summary = (e.findtext("atom:summary", "", ns) or "").strip()[:150]
            if title:
                items.append({"source": "arXiv (cs.LG)", "title": title,
                              "url": link, "summary": summary})
        return items
    except Exception as e:
        print(f"[NEWS] arXiv cs.LG取得失敗: {e}")
        return []


# ─────────────────────────────────────────
# 全ソース収集 → swallowで選別・要約
# ─────────────────────────────────────────

def fetch_all(max_per_source: int = 3) -> list[dict]:
    """4ソースから収集してリストで返す。"""
    print("[NEWS] ニュース収集開始...")
    all_items = []

    all_items += fetch_arxiv(max_per_source)
    time.sleep(1)
    all_items += fetch_hackernews(max_per_source)
    time.sleep(1)
    all_items += fetch_google_news(max_per_source)
    time.sleep(1)
    all_items += fetch_papers_with_code(max_per_source)

    print(f"[NEWS] 収集完了: {len(all_items)}件")
    return all_items


def select_and_summarize(items: list[dict], num_select: int = 3) -> list[dict]:
    """
    swallowに掲示板向けのトップニュースを選ばせ、
    日本語要約と掲載理由を付ける。
    """
    if not items:
        return []

    items_text = "\n".join(
        f"[{i+1}] ({it['source']}) {it['title']}\n    {it['summary'][:100]}\n    URL: {it['url']}"
        for i, it in enumerate(items)
    )

    prompt = f"""以下は今週のAI関連ニュース候補です（複数ソースから収集）。

{items_text}

掲示板（AIや技術に関心のある読者向け）に掲載するのにふさわしい{num_select}件を選び、
各ニュースについて日本語で2文以内の紹介文を書いてください。

条件：
- 読者が「へえ、そうなんだ」と思えるような話題を優先
- 論文・技術・社会影響のバランスを取る
- 日本語の紹介文は平易に（専門用語を避ける）

以下のJSONのみ返してください：
{{
  "selected": [
    {{
      "index": 元のリスト番号(1始まり),
      "intro": "日本語紹介文（2文以内）",
      "reason": "選んだ理由（10文字以内）"
    }}
  ]
}}"""

    try:
        result = ask_ai(prompt, model="swallow")
        selected_raw = result.get("selected", [])
    except Exception as e:
        print(f"[NEWS] swallow選別失敗: {e}")
        return items[:num_select]

    output = []
    for s in selected_raw:
        idx = s.get("index", 1) - 1
        if 0 <= idx < len(items):
            item = dict(items[idx])
            item["intro"] = s.get("intro", "")
            item["reason"] = s.get("reason", "")
            output.append(item)

    return output[:num_select]


def build_board_post(selected: list[dict]) -> str:
    """掲示板投稿用テキストを生成する。"""
    today = datetime.date.today().strftime("%Y/%m/%d")
    lines = [f"📰 今週のAIニュース（{today}）\n"]
    for item in selected:
        intro = item.get("intro") or item.get("summary", "")[:80]
        lines.append(f"▶ {item['title']}")
        if intro:
            lines.append(f"  {intro}")
        lines.append(f"  出典: {item['source']} | {item['url']}")
        lines.append("")
    lines.append("（管理AI）")
    return "\n".join(lines)


def run(num_select: int = 3) -> dict:
    """
    ニュース収集 → 選別 → 投稿文生成まで一括実行。
    戻り値: {"post_text": str, "items": list}
    """
    items = fetch_all()
    if not items:
        return {"post_text": "（ニュースの取得に失敗しました）", "items": []}

    # キャッシュ保存
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": datetime.datetime.now().isoformat(), "items": items}, f,
                      ensure_ascii=False, indent=2)
    except Exception:
        pass

    selected = select_and_summarize(items, num_select)
    post_text = build_board_post(selected)
    return {"post_text": post_text, "items": selected}
