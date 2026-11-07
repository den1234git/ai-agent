"""
polymarket.py — Polymarket Gamma API からマーケットデータを取得

【用途】採点時の比較相手。村人の材料には入れない。
【認証】不要。Gamma API は完全公開。
【レート制限】公式ドキュメントに明記なし。過剰アクセスを避けるため
            1予想につき最大2リクエストに収める。

Gamma API ベースURL: https://gamma-api.polymarket.com
主なエンドポイント:
  GET /markets?keyword=<query>&limit=<n>   マーケット一覧（キーワード検索）
  GET /events?keyword=<query>&limit=<n>    イベント一覧

使い方:
    from polymarket import search_markets
    r = search_markets("Democrats House majority 2026")
    # → {"ok": True, "markets": [...], "summary": "...", "retrieved_at": "..."}
"""
from __future__ import annotations

import datetime
import json
import urllib.request
import urllib.parse

_BASE = "https://gamma-api.polymarket.com"
_UA = "Mozilla/5.0 (compatible; LAVillage/1.0; yosouya-scoring)"
_TIMEOUT = 15


def _get(path: str, params: dict | None = None) -> dict:
    url = _BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_markets(query: str, limit: int = 5) -> dict:
    """
    キーワードで Polymarket マーケットを検索し、
    現在の Yes 価格（確率）と出来高を返す。

    戻り値:
      {
        "ok": bool,
        "query": str,
        "markets": [
          {
            "id": str,
            "question": str,
            "yes_price": float,       # 0.0〜1.0（Yesの価格 = 確率の市場評価）
            "yes_pct": float,         # yes_price * 100
            "volume_usd": float,      # 出来高（ドル）
            "end_date": str,
            "url": str,
          }
        ],
        "summary": str,               # 採点メモ用テキスト
        "retrieved_at": str,
        "error": str | None,
      }
    """
    try:
        data = _get("/markets", {"keyword": query, "limit": limit,
                                 "active": "true", "closed": "false"})
    except Exception as e:
        return {"ok": False, "query": query, "markets": [], "summary": "",
                "retrieved_at": _now(), "error": str(e)}

    markets: list[dict] = []
    raw_list = data if isinstance(data, list) else data.get("markets", [])

    for m in raw_list:
        # outcomePrices は JSON 文字列として入っていることがある
        outcome_prices = m.get("outcomePrices") or []
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                outcome_prices = []

        yes_price: float | None = None
        if outcome_prices and len(outcome_prices) >= 1:
            try:
                yes_price = float(outcome_prices[0])
            except (ValueError, TypeError):
                pass

        # volume は文字列の場合もある
        vol = m.get("volumeNum") or m.get("volume") or 0
        try:
            vol = float(vol)
        except (ValueError, TypeError):
            vol = 0.0

        slug = m.get("slug", "")
        url = f"https://polymarket.com/event/{slug}" if slug else ""

        markets.append({
            "id":         str(m.get("id", "")),
            "question":   m.get("question", ""),
            "yes_price":  yes_price,
            "yes_pct":    round(yes_price * 100, 1) if yes_price is not None else None,
            "volume_usd": round(vol, 0),
            "end_date":   m.get("endDate", ""),
            "url":        url,
        })

    summary = _build_summary(query, markets)
    return {"ok": True, "query": query, "markets": markets,
            "summary": summary, "retrieved_at": _now(), "error": None}


def _build_summary(query: str, markets: list[dict]) -> str:
    if not markets:
        return f"（Polymarket: 「{query}」に該当マーケットなし）"
    lines = [f"【Polymarket 比較（採点用）: {query}】"]
    for m in markets:
        pct = f"{m['yes_pct']}%" if m["yes_pct"] is not None else "価格不明"
        vol = f"${m['volume_usd']:,.0f}" if m["volume_usd"] else ""
        line = f"  {m['question'][:80]}  → Yes {pct}"
        if vol:
            line += f"  出来高 {vol}"
        lines.append(line)
    return "\n".join(lines)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ── 問いから検索クエリを推定 ─────────────────────────────────────

def build_query(question: str) -> str:
    """日本語の問いから Polymarket 検索クエリを作る。"""
    q = question
    if "中間選挙" in q or "midterm" in q.lower():
        import re
        year_m = re.search(r"(20\d\d)", q)
        year = year_m.group(1) if year_m else "2026"
        if "下院" in q or "house" in q.lower():
            return f"Democrats House majority {year}"
        if "上院" in q or "senate" in q.lower():
            return f"Democrats Senate majority {year}"
        return f"US midterm election {year}"
    if "大統領" in q or "presidential" in q.lower():
        import re
        year_m = re.search(r"(20\d\d)", q)
        year = year_m.group(1) if year_m else "2028"
        return f"US presidential election {year}"
    # フォールバック：英数字以外を除いて最初の60文字
    import re
    return re.sub(r"[^\w\s]", " ", q)[:60].strip()


# ── 直接実行でデモ ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Democrats House majority 2026"
    r = search_markets(q)
    print(f"ok={r['ok']}, markets={len(r['markets'])}")
    if r["ok"]:
        print(r["summary"])
    else:
        print("ERROR:", r["error"])
