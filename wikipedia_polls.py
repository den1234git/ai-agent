"""
wikipedia_polls.py — Wikipedia 世論調査まとめ表の取得

選挙予想の材料として使う。Wikipedia の選挙ページには
各レースの世論調査が表になっており、調査会社・実施期間・サンプル数まで載っている。
無料、APIキー不要。

使い方:
    from wikipedia_polls import fetch_polls
    result = fetch_polls("2026 United States House of Representatives elections")
    # → {"ok": True, "rows": [...], "source_url": "...", "retrieved_at": "..."}
"""
from __future__ import annotations

import datetime
import re
import urllib.request
from urllib.parse import quote

# Wikipedia の HTML から table を取り出すのに標準ライブラリだけで済ます。
# BeautifulSoup がある環境なら使うが、なければ正規表現で代用。
try:
    from bs4 import BeautifulSoup as _BS
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_BASE = "https://en.wikipedia.org/wiki/"
_UA = "Mozilla/5.0 (compatible; LAVillage/1.0; yosouya-bot)"
_TIMEOUT = 20


def _fetch_html(title: str) -> tuple[str, str]:
    """Wikipedia ページの HTML と URL を返す。"""
    encoded = quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/wiki/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return html, url


def _parse_tables_bs4(html: str) -> list[list[list[str]]]:
    """BS4 で wikitable を全部抽出する。"""
    soup = _BS(html, "html.parser")
    tables: list[list[list[str]]] = []
    for tbl in soup.find_all("table", class_=re.compile(r"wikitable")):
        rows: list[list[str]] = []
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _parse_tables_regex(html: str) -> list[list[list[str]]]:
    """BS4 がない場合のフォールバック。wikitable を正規表現で抽出する（粗い）。"""
    # <table class="wikitable...">...</table> を取り出す
    tbl_re = re.compile(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    row_re = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_re = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    tag_re = re.compile(r'<[^>]+>')

    def strip(s: str) -> str:
        return tag_re.sub("", s).strip()

    tables: list[list[list[str]]] = []
    for tbl_m in tbl_re.finditer(html):
        rows: list[list[str]] = []
        for row_m in row_re.finditer(tbl_m.group(1)):
            cells = [strip(c.group(1)) for c in cell_re.finditer(row_m.group(1))]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _tables_to_text(tables: list[list[list[str]]], max_tables: int = 3, max_rows: int = 30) -> str:
    """テーブルを人間が読めるテキストに変換する。"""
    parts: list[str] = []
    for i, tbl in enumerate(tables[:max_tables]):
        header = tbl[0] if tbl else []
        lines = [" | ".join(header)]
        lines.append("-" * min(80, sum(len(h) + 3 for h in header)))
        for row in tbl[1:max_rows]:
            lines.append(" | ".join(row))
        parts.append(f"[表 {i+1}]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def fetch_polls(page_title: str, max_tables: int = 3) -> dict:
    """
    Wikipedia ページから世論調査表を取得してテキストにまとめる。

    戻り値:
      {
        "ok": bool,
        "source_url": str,
        "page_title": str,
        "tables_found": int,
        "text": str,          # 村人に配る材料テキスト
        "retrieved_at": str,
        "error": str | None,
      }
    """
    try:
        html, url = _fetch_html(page_title)
    except Exception as e:
        return {"ok": False, "source_url": "", "page_title": page_title,
                "tables_found": 0, "text": "", "retrieved_at": _now(), "error": str(e)}

    try:
        tables = _parse_tables_bs4(html) if _HAS_BS4 else _parse_tables_regex(html)
    except Exception as e:
        return {"ok": False, "source_url": url, "page_title": page_title,
                "tables_found": 0, "text": "", "retrieved_at": _now(), "error": f"パース失敗: {e}"}

    if not tables:
        return {"ok": True, "source_url": url, "page_title": page_title,
                "tables_found": 0, "text": f"（{page_title} に世論調査表が見つかりませんでした）",
                "retrieved_at": _now(), "error": None}

    text = _tables_to_text(tables, max_tables=max_tables)
    return {
        "ok": True,
        "source_url": url,
        "page_title": page_title,
        "tables_found": len(tables),
        "text": text,
        "retrieved_at": _now(),
        "error": None,
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ── 選挙ページタイトルの推定 ─────────────────────────────────────

def guess_election_page(question: str) -> str | None:
    """
    問いの文字列から Wikipedia の選挙ページタイトルを推定する。
    分からなければ None を返す。
    """
    q = question.lower()
    if "中間選挙" in question or "midterm" in q:
        year_m = re.search(r"(20\d\d)", question)
        year = year_m.group(1) if year_m else "2026"
        if "下院" in question or "house" in q:
            return f"{year} United States House of Representatives elections"
        if "上院" in question or "senate" in q:
            return f"{year} United States Senate elections"
        return f"{year} United States midterm elections"
    if "大統領" in question or "presidential" in q:
        year_m = re.search(r"(20\d\d)", question)
        year = year_m.group(1) if year_m else "2024"
        return f"{year} United States presidential election"
    return None


# ── 直接実行でデモ ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    title = sys.argv[1] if len(sys.argv) > 1 else \
        "2026 United States House of Representatives elections"
    r = fetch_polls(title)
    print(f"ok={r['ok']}, tables={r['tables_found']}, url={r['source_url']}")
    if r["ok"]:
        print(r["text"][:2000])
    else:
        print("ERROR:", r["error"])
