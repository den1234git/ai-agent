"""
kalshi.py — Kalshi 公開 API からマーケットデータを取得

【用途】採点時の比較相手。村人の材料には入れない。
【認証】不要。マーケットデータの読み取りは公開エンドポイント。
        取引・発注には認証が必要だが、ここでは読み取りのみ。
【規約】規約・レート制限の範囲内で使用。

API ベースURL: https://trading-api.kalshi.com/trade-api/v2
主なエンドポイント:
  GET /markets?limit=<n>&keyword=<q>&status=open
  GET /events?limit=<n>&keyword=<q>&status=open

Kalshi の特徴:
  - CFTC 規制を受けた米国正式取引所
  - `settled` / `closed` ステータスが API で取得できるため判定・採点に使いやすい
  - `result` フィールドが "yes"/"no"/"" で確定後に入る → 自動採点に利用可能

使い方:
    from kalshi import search_markets
    r = search_markets("Democrats House majority 2026")
    # → {"ok": True, "markets": [...], "summary": "...", "retrieved_at": "..."}
"""
from __future__ import annotations

import datetime
import json
import urllib.request
import urllib.parse

_BASE = "https://trading-api.kalshi.com/trade-api/v2"
_UA = "Mozilla/5.0 (compatible; LAVillage/1.0; yosouya-scoring)"
_TIMEOUT = 15


def _get(path: str, params: dict | None = None) -> dict:
    url = _BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_markets(query: str, limit: int = 5) -> dict:
    """
    キーワードで Kalshi マーケットを検索し、
    現在の Yes 価格（確率）・出来高・決済状態を返す。

    戻り値:
      {
        "ok": bool,
        "query": str,
        "markets": [
          {
            "ticker": str,
            "title": str,
            "yes_price": float | None,    # セント単位 → 0〜1 に変換
            "yes_pct": float | None,
            "volume": int,                # コントラクト出来高
            "status": str,                # "open" / "closed" / "settled"
            "result": str | None,         # "yes" / "no" / None（未確定）
            "close_time": str,
            "url": str,
          }
        ],
        "summary": str,
        "retrieved_at": str,
        "error": str | None,
      }
    """
    try:
        data = _get("/markets", {"limit": limit, "status": "open",
                                  "keyword": query})
    except Exception as e:
        return {"ok": False, "query": query, "markets": [], "summary": "",
                "retrieved_at": _now(), "error": str(e)}

    raw_list = data.get("markets", [])
    markets: list[dict] = []

    for m in raw_list:
        # yes_ask / yes_bid の中点を確率として使う（セント単位: 0〜100）
        yes_ask = m.get("yes_ask")
        yes_bid = m.get("yes_bid")
        yes_price: float | None = None
        if yes_ask is not None and yes_bid is not None:
            try:
                yes_price = (float(yes_ask) + float(yes_bid)) / 2 / 100
            except (ValueError, TypeError):
                pass
        elif yes_ask is not None:
            try:
                yes_price = float(yes_ask) / 100
            except (ValueError, TypeError):
                pass

        ticker = m.get("ticker", "")
        url = f"https://kalshi.com/markets/{ticker}" if ticker else ""

        markets.append({
            "ticker":     ticker,
            "title":      m.get("title", ""),
            "yes_price":  yes_price,
            "yes_pct":    round(yes_price * 100, 1) if yes_price is not None else None,
            "volume":     m.get("volume", 0),
            "status":     m.get("status", ""),
            "result":     m.get("result"),     # "yes"/"no"/None
            "close_time": m.get("close_time", ""),
            "url":        url,
        })

    summary = _build_summary(query, markets)
    return {"ok": True, "query": query, "markets": markets,
            "summary": summary, "retrieved_at": _now(), "error": None}


def get_settled_result(ticker: str) -> dict:
    """
    特定の ticker の決済結果を返す（自動採点用）。

    戻り値:
      {
        "ok": bool,
        "ticker": str,
        "status": str,         # "settled" なら result が確定
        "result": str | None,  # "yes" / "no" / None
        "retrieved_at": str,
        "error": str | None,
      }
    """
    try:
        data = _get(f"/markets/{ticker}")
        m = data.get("market", data)
        return {
            "ok": True,
            "ticker":       ticker,
            "status":       m.get("status", ""),
            "result":       m.get("result"),
            "retrieved_at": _now(),
            "error":        None,
        }
    except Exception as e:
        return {"ok": False, "ticker": ticker, "status": "", "result": None,
                "retrieved_at": _now(), "error": str(e)}


def _build_summary(query: str, markets: list[dict]) -> str:
    if not markets:
        return f"（Kalshi: 「{query}」に該当マーケットなし）"
    lines = [f"【Kalshi 比較（採点用）: {query}】"]
    for m in markets:
        pct = f"{m['yes_pct']}%" if m["yes_pct"] is not None else "価格不明"
        result_str = ""
        if m["result"]:
            result_str = f"  ★確定: {m['result'].upper()}"
        elif m["status"] == "settled":
            result_str = "  ★決済済"
        line = f"  {m['title'][:80]}  → Yes {pct}{result_str}"
        lines.append(line)
    return "\n".join(lines)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ── 問いから検索クエリを推定（polymarket.py と共通） ─────────────

def build_query(question: str) -> str:
    from polymarket import build_query as _bq
    return _bq(question)


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
