"""
fetch_election_materials.py — 選挙関連情報の多ソース収集モジュール
=================================================================
以下のソースから選挙関連データを収集し、予想材料テキストを生成する。

  1. FiveThirtyEight  — 米選挙予測モデル（スクレイピング）
  2. Bluesky検索     — 選挙キーワードでSNS世論取得
  3. NHK選挙         — 日本選挙の情勢（RSS）

各ソースは独立して失敗可能（失敗時はスキップしてログ出力）。
"""
from __future__ import annotations

import re
import json
import time
import datetime
import urllib.request
import urllib.parse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "election-materials-fetcher/1.0 (contact: local-agent)"

BSKY_SEARCH_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
NHK_SENKYO_RSS  = "https://www.nhk.or.jp/senkyo/rss/news.xml"
FIVETHIRTYEIGHT_BASE = "https://projects.fivethirtyeight.com/"


# ── FiveThirtyEight ──────────────────────────────────────────────

def fetch_fivethirtyeight(race_path: str = "2024-elections/") -> dict:
    """
    FiveThirtyEight 選挙予測ページから各党の勝率を取得。
    race_path: "2024-elections/", "2026-senate-forecast/" 等

    失敗時: {"ok": False, "error": str}
    成功時: {"ok": True, "source": "FiveThirtyEight", "url": str, "snippets": [str]}
    """
    url = f"{FIVETHIRTYEIGHT_BASE}{race_path}"
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": USER_AGENT,
                                  "Accept-Language": "en-US,en;q=0.9"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"[election] FiveThirtyEight スキップ: {e}")
        return {"ok": False, "error": str(e)[:100]}

    snippets: list[str] = []
    try:
        # 予測確率テーブルを探す
        for el in soup.select(".forecast-box, .probability-box, "
                               "[class*='forecast'], [class*='winner'], "
                               "[class*='prob'], .house-calls, .senate-calls"):
            txt = el.get_text(separator=" ", strip=True)
            if txt and len(txt) > 5:
                snippets.append(txt[:200])
        # フォールバック: <strong>や<b>で%が含まれる要素
        if not snippets:
            for el in soup.find_all(["strong", "b", "span"]):
                txt = el.get_text(strip=True)
                if re.search(r"\d+%", txt) and len(txt) < 100:
                    snippets.append(txt)
    except Exception as e:
        print(f"[election] FiveThirtyEight パース失敗: {e}")

    return {
        "ok":      True,
        "source":  "FiveThirtyEight",
        "url":     url,
        "snippets": snippets[:10],
    }


# ── Bluesky 選挙SNS ──────────────────────────────────────────────

def fetch_bluesky_election(keywords: list[str]) -> dict:
    """
    Bluesky公開APIで選挙キーワードを検索してSNS世論を取得。

    keywords: ["2026 election", "2026 midterms", "選挙", ...]
    失敗時: {"ok": False, "error": str}
    成功時: {"ok": True, "total": int, "posts": [{keyword, text, at, handle}]}
    """
    all_posts: list[dict] = []
    total = 0
    errors: list[str] = []

    for kw in keywords:
        try:
            r = requests.get(
                BSKY_SEARCH_URL,
                params={"q": kw, "limit": 10, "sort": "latest"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=8,
            )
            if r.status_code == 200:
                posts = r.json().get("posts", [])
                total += len(posts)
                for p in posts[:3]:
                    rec = p.get("record", {})
                    all_posts.append({
                        "keyword": kw,
                        "text":    rec.get("text", "")[:150],
                        "at":      rec.get("createdAt", "")[:16],
                        "handle":  p.get("author", {}).get("handle", ""),
                    })
            else:
                errors.append(f"{kw}: HTTP {r.status_code}")
            time.sleep(0.3)
        except Exception as e:
            errors.append(f"{kw}: {str(e)[:50]}")
            print(f"[election] Bluesky '{kw}' スキップ: {e}")

    return {
        "ok":       True,
        "source":   "Bluesky",
        "total":    total,
        "posts":    all_posts[:15],
        "keywords": keywords,
        "errors":   errors,
    }


# ── NHK選挙 RSS ──────────────────────────────────────────────────

def fetch_nhk_senkyo() -> dict:
    """
    NHK選挙RSSから最新の選挙関連ニュースを取得。
    失敗時: {"ok": False, "error": str}
    成功時: {"ok": True, "source": "NHK選挙", "items": [{title, link, pubDate}]}
    """
    try:
        req = urllib.request.Request(
            NHK_SENKYO_RSS,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            rss_text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[election] NHK選挙RSS スキップ: {e}")
        return {"ok": False, "error": str(e)[:100]}

    items: list[dict] = []
    try:
        # 簡易XMLパース（lxml不要）
        for item_m in re.finditer(r"<item>(.*?)</item>", rss_text, re.DOTALL):
            item_text = item_m.group(1)
            title   = re.search(r"<title>(.*?)</title>", item_text)
            link    = re.search(r"<link>(.*?)</link>", item_text)
            pubdate = re.search(r"<pubDate>(.*?)</pubDate>", item_text)
            desc    = re.search(r"<description>(.*?)</description>", item_text)
            items.append({
                "title":   (title.group(1) if title else "").strip()[:100],
                "link":    (link.group(1) if link else "").strip(),
                "pubDate": (pubdate.group(1) if pubdate else "").strip()[:30],
                "desc":    (desc.group(1) if desc else "").strip()[:150],
            })
    except Exception as e:
        print(f"[election] NHK RSS パース失敗: {e}")
        return {"ok": False, "error": f"パース失敗: {e}"}

    return {
        "ok":     True,
        "source": "NHK選挙",
        "url":    NHK_SENKYO_RSS,
        "items":  items[:20],
    }


# ── 統合取得 ─────────────────────────────────────────────────────

def fetch_election_materials(
    keywords_en: list[str] | None = None,
    keywords_ja: list[str] | None = None,
    fivethirtyeight_path: str | None = None,
    include_nhk: bool = True,
) -> dict:
    """
    選挙関連の全ソースから情報を収集して統合辞書を返す。

    keywords_en: Bluesky英語検索キーワード（例: ["2026 election", "midterm"]）
    keywords_ja: Bluesky日本語検索キーワード（例: ["参院選", "選挙"]）
    fivethirtyeight_path: FiveThirtyEightのパス（省略時=最新の選挙）
    include_nhk: NHK選挙RSSを取得するか（日本選挙のみ有用）

    戻り値:
      {
        "ok": True,
        "fetched_at": str,
        "fivethirtyeight": {...},
        "bluesky": {...},
        "nhk": {...},          # include_nhk=True の場合のみ
        "summary_text": str,   # 村人への材料テキスト
      }
    """
    result: dict = {
        "ok":         True,
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # 1) FiveThirtyEight
    path = fivethirtyeight_path or "2026-elections/"
    result["fivethirtyeight"] = fetch_fivethirtyeight(path)

    # 2) Bluesky
    all_kws = (keywords_en or []) + (keywords_ja or [])
    if not all_kws:
        all_kws = ["2026 election", "2026 midterms", "Democrat Republican"]
    result["bluesky"] = fetch_bluesky_election(all_kws)

    # 3) NHK選挙
    if include_nhk:
        result["nhk"] = fetch_nhk_senkyo()

    # 4) 材料テキスト生成
    lines: list[str] = ["【選挙情勢 追加材料】", ""]

    fte = result["fivethirtyeight"]
    if fte.get("ok") and fte.get("snippets"):
        lines.append("■ FiveThirtyEight 予測")
        for s in fte["snippets"][:5]:
            lines.append(f"  {s}")
        lines.append("")

    bsky = result["bluesky"]
    if bsky.get("ok") and bsky.get("posts"):
        lines.append(f"■ SNS世論（Bluesky, {bsky['total']}件）")
        for p in bsky["posts"][:6]:
            lines.append(f"  [{p['keyword']}] {p['text'][:80]}")
        lines.append("")

    nhk = result.get("nhk", {})
    if nhk.get("ok") and nhk.get("items"):
        lines.append("■ NHK選挙ニュース")
        for item in nhk["items"][:5]:
            lines.append(f"  {item['pubDate'][:10]} {item['title']}")
        lines.append("")

    result["summary_text"] = "\n".join(lines)
    return result


# ── 直接実行でデモ ──────────────────────────────────────────────
if __name__ == "__main__":
    print("選挙材料を取得中...")
    r = fetch_election_materials(
        keywords_en=["2026 election", "2026 midterms"],
        keywords_ja=["選挙", "参院選"],
        include_nhk=True,
    )
    print(r["summary_text"])
    print(f"FiveThirtyEight: {'OK' if r['fivethirtyeight'].get('ok') else 'NG'}")
    print(f"Bluesky: {r['bluesky'].get('total', 0)}件")
    if "nhk" in r:
        print(f"NHK: {len(r['nhk'].get('items', []))}件")
