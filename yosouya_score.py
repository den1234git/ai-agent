"""
yosouya_score.py — 予想屋さん 採点パイプライン

【役割】
  1. 台帳 (ledger.jsonl) の未採点エントリを探す
     条件: deadline が今日以前 かつ resolved が null
  2. Kalshi の決済結果を確認
     → settled + result が "yes"/"no" なら自動採点
  3. 各村人の Brier スコアを計算して台帳に書き戻す
     Brier = (確率/100 − 結果)²  小さいほど良い
  4. 市場（Polymarket / Kalshi）の Brier スコアも並べて記録

【使い方】
  from yosouya_score import check_and_score
  results = check_and_score()          # 自動判定できたものをまとめて採点

  from yosouya_score import score_entry
  score_entry("y_20261108120000", outcome=1)  # 手動で結果を指定して採点
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib

import yosouya as _y


# ── Brier スコア計算 ─────────────────────────────────────────────

def brier(prob_pct: float | None, outcome: int) -> float | None:
    """
    Brier スコアを返す。prob_pct は 0〜100 の整数/小数。
    outcome は 1（Yes/起きた） または 0（No/起きなかった）。
    """
    if prob_pct is None:
        return None
    p = max(0.0, min(100.0, float(prob_pct))) / 100.0
    return round((p - outcome) ** 2, 4)


# ── 台帳の読み書き ───────────────────────────────────────────────

def _read_ledger() -> list[dict]:
    path = pathlib.Path(_y.LEDGER_PATH)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _write_ledger(entries: list[dict]) -> None:
    path = pathlib.Path(_y.LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _today() -> str:
    return datetime.date.today().isoformat()


# ── 自動判定: Kalshi の決済ステータスを確認 ───────────────────────

def _auto_detect_outcome(entry: dict) -> int | None:
    """
    Kalshi の決済結果から outcome (1/0) を返す。
    まだ決済されていない場合は None を返す。
    """
    from kalshi import get_settled_result

    market_ref = entry.get("market_ref") or {}
    kalshi_markets = market_ref.get("kalshi") or []

    for m in kalshi_markets:
        ticker = m.get("ticker") or ""
        if not ticker:
            continue
        r = get_settled_result(ticker)
        if r.get("ok") and r.get("status") == "settled" and r.get("result") in ("yes", "no"):
            outcome = 1 if r["result"] == "yes" else 0
            print(f"[score] Kalshi '{ticker}' → {r['result']} (outcome={outcome})")
            return outcome

    return None


# ── 採点コア ─────────────────────────────────────────────────────

def score_entry(entry_id: str, outcome: int, source: str = "manual") -> dict:
    """
    特定エントリを採点して台帳に書き戻す。

    outcome : 1 = Yes（起きた）/ 0 = No（起きなかった）
    source  : "auto_kalshi" / "auto_polymarket" / "manual"

    戻り値:
      {
        "ok": bool,
        "id": str,
        "outcome": int,
        "median_brier": float | None,
        "market_brier": {"polymarket": float | None, "kalshi": float | None},
        "villager_briers": [{"kote": str, "brier": float | None}],
        "error": str | None,
      }
    """
    entries = _read_ledger()
    target_idx: int | None = None
    for i, e in enumerate(entries):
        if e.get("id") == entry_id:
            target_idx = i
            break

    if target_idx is None:
        return {"ok": False, "id": entry_id, "error": "エントリが見つかりません"}

    entry = entries[target_idx]

    # ── 各村人の Brier スコア ────────────────────────────────────
    villager_briers: list[dict] = []
    for est in entry.get("estimates", []):
        b = brier(est.get("prob"), outcome)
        villager_briers.append({"kote": est.get("kote", "?"), "brier": b})

    valid_briers = [v["brier"] for v in villager_briers if v["brier"] is not None]
    median_b = round(sum(valid_briers) / len(valid_briers), 4) if valid_briers else None

    # ── 市場の Brier スコア ──────────────────────────────────────
    market_ref = entry.get("market_ref") or {}
    pm_brier: float | None = None
    ka_brier: float | None = None

    pm_markets = market_ref.get("polymarket") or []
    if pm_markets:
        # 最初のマーケットの yes_pct で代表させる
        pm_pct = pm_markets[0].get("yes_pct")
        pm_brier = brier(pm_pct, outcome)

    ka_markets = market_ref.get("kalshi") or []
    if ka_markets:
        ka_pct = ka_markets[0].get("yes_pct")
        ka_brier = brier(ka_pct, outcome)

    # ── 台帳エントリを更新 ────────────────────────────────────────
    entry["resolved"] = outcome
    entry["resolved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    entry["resolved_source"] = source
    entry["brier_avg"] = median_b         # 村の平均 Brier（個人スコアは下記）
    entry["villager_briers"] = villager_briers
    entry["market_brier"] = {"polymarket": pm_brier, "kalshi": ka_brier}

    entries[target_idx] = entry
    _write_ledger(entries)

    print(f"[score] 採点完了: {entry_id} outcome={outcome} "
          f"avg_brier={median_b} pm={pm_brier} ka={ka_brier}")

    return {
        "ok":             True,
        "id":             entry_id,
        "outcome":        outcome,
        "median_brier":   median_b,
        "market_brier":   {"polymarket": pm_brier, "kalshi": ka_brier},
        "villager_briers": villager_briers,
        "error":          None,
    }


# ── 自動採点パイプライン ─────────────────────────────────────────

def check_and_score() -> dict:
    """
    台帳を全スキャンして採点できるエントリを自動採点する。

    判定条件:
      - deadline が今日以前（空の場合はスキップ）
      - resolved が null
      - Kalshi の決済結果が "yes" or "no" で確定している

    戻り値:
      {
        "checked":  int,           # スキャンした件数
        "scored":   int,           # 採点できた件数
        "skipped":  int,           # 判定できなかった件数（未決済 or deadline 未来）
        "results":  [score_entry の戻り値],
      }
    """
    entries = _read_ledger()
    today = _today()

    checked = 0
    scored = 0
    skipped = 0
    results: list[dict] = []

    for entry in entries:
        # 既に採点済み
        if entry.get("resolved") is not None:
            continue

        # deadline が未設定 or 未来
        deadline = entry.get("deadline", "")
        if not deadline or deadline > today:
            continue

        checked += 1
        entry_id = entry.get("id", "")
        print(f"[score] 判定確認: {entry_id} deadline={deadline} question={entry.get('question', '')[:40]}")

        outcome = _auto_detect_outcome(entry)
        if outcome is None:
            print(f"[score] スキップ（Kalshi 未決済）: {entry_id}")
            skipped += 1
            continue

        result = score_entry(entry_id, outcome, source="auto_kalshi")
        results.append(result)
        if result["ok"]:
            scored += 1
        else:
            skipped += 1

    return {
        "checked": checked,
        "scored":  scored,
        "skipped": skipped,
        "results": results,
    }


# ── 直接実行でデモ ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "score":
        # python yosouya_score.py score <entry_id> <outcome:0|1>
        eid = sys.argv[2]
        out = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        r = score_entry(eid, out)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        # 自動採点チェック
        r = check_and_score()
        print(json.dumps(r, ensure_ascii=False, indent=2))
