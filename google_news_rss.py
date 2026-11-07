"""
google_news_rss.py — Google News RSS による最新ニュース取得

状況変化の材料として使う。RSS なので APIキー不要・無料。
ただし Google の利用制限がある（過剰なアクセスはブロックされる）ので
1回の予想で 2〜3 クエリまでにする。

使い方:
    from google_news_rss import fetch_news
    result = fetch_news("2026 midterm election Democrats House")
    # → {"ok": True, "items": [...], "summary": "...", "retrieved_at": "..."}
"""
from __future__ import annotations

import datetime
import re
import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote_plus

_UA = "Mozilla/5.0 (compatible; LAVillage/1.0; yosouya-bot)"
_TIMEOUT = 15
_BASE_URL = "https://news.google.com/rss/search"


def fetch_news(query: str, max_items: int = 8, lang: str = "en", country: str = "US") -> dict:
    """
    Google News RSS を取得してニュース一覧を返す。

    戻り値:
      {
        "ok": bool,
        "query": str,
        "items": [{"title", "link", "pubDate", "source", "description"}],
        "summary": str,       # 村人に配る材料テキスト
        "retrieved_at": str,
        "error": str | None,
      }
    """
    params = urlencode({
        "q": query,
        "hl": lang,
        "gl": country,
        "ceid": f"{country}:{lang}",
    })
    url = f"{_BASE_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            xml_bytes = resp.read()
    except Exception as e:
        return {"ok": False, "query": query, "items": [], "summary": "",
                "retrieved_at": _now(), "error": str(e)}

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        return {"ok": False, "query": query, "items": [], "summary": "",
                "retrieved_at": _now(), "error": f"XML parse error: {e}"}

    ns = {"media": "http://search.yahoo.com/mrss/"}
    items: list[dict] = []
    channel = root.find("channel")
    if channel is None:
        return {"ok": True, "query": query, "items": [], "summary": "(記事なし)",
                "retrieved_at": _now(), "error": None}

    for item_el in channel.findall("item")[:max_items]:
        title = _text(item_el, "title")
        link = _text(item_el, "link")
        pub_date = _text(item_el, "pubDate")
        source_el = item_el.find("source")
        source = source_el.text if source_el is not None else ""
        desc_raw = _text(item_el, "description")
        desc = _strip_html(desc_raw)[:200] if desc_raw else ""
        items.append({
            "title": title,
            "link": link,
            "pubDate": pub_date,
            "source": source,
            "description": desc,
        })

    summary = _build_summary(query, items)
    return {
        "ok": True,
        "query": query,
        "items": items,
        "summary": summary,
        "retrieved_at": _now(),
        "error": None,
    }


def _text(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _build_summary(query: str, items: list[dict]) -> str:
    if not items:
        return f"（「{query}」の最新ニュースは見つかりませんでした）"
    lines = [f"【Google News: {query}】"]
    for i, it in enumerate(items, 1):
        line = f"[{i}] {it['title']}"
        if it["pubDate"]:
            line += f"  ({it['pubDate'][:16]})"
        if it["source"]:
            line += f"  — {it['source']}"
        lines.append(line)
        if it["description"]:
            lines.append(f"    {it['description']}")
    return "\n".join(lines)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ── 選挙向けクエリ推定 ─────────────────────────────────────────────

def build_election_queries(question: str) -> list[str]:
    """
    問いの文章から Google News 検索クエリを 1〜2 個生成する。
    """
    q = question
    queries: list[str] = []

    if "中間選挙" in q or "midterm" in q.lower():
        import re as _re
        year_m = _re.search(r"(20\d\d)", q)
        year = year_m.group(1) if year_m else "2026"
        queries.append(f"{year} midterm election results forecast")
        if "下院" in q or "house" in q.lower():
            queries.append(f"{year} House Democrats Republicans control poll")
        elif "上院" in q or "senate" in q.lower():
            queries.append(f"{year} Senate election results latest")
    elif "大統領" in q or "presidential" in q.lower():
        import re as _re
        year_m = _re.search(r"(20\d\d)", q)
        year = year_m.group(1) if year_m else "2028"
        queries.append(f"{year} presidential election poll latest")
    else:
        # 一般クエリ：問い文字列をそのまま短縮
        queries.append(q[:80])

    return queries[:2]


# ── 直接実行でデモ ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "2026 midterm election Democrats House"
    r = fetch_news(query)
    print(f"ok={r['ok']}, items={len(r['items'])}")
    if r["ok"]:
        print(r["summary"])
    else:
        print("ERROR:", r["error"])
