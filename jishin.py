"""
jishin.py — 余震予測モジュール
=================================
工程:
  ① 気象庁API から最近の地震リストを取得
  ② 指定した本震の余震シーケンスを抽出
  ③ 大森公式（参考値）+ 村人の独立見積もり
  ④ 「次の24h以内にM○以上の余震が来る確率」を出力
  ⑤ jishin_ledger.jsonl に記録

台帳は keiba・yosouya とは別ファイル。
"""
from __future__ import annotations

import os
import re
import json
import math
import time
import datetime
import statistics
import concurrent.futures

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

LEDGER_PATH = os.environ.get(
    "JISHIN_LEDGER",
    r"C:\Users\USER\workspace\yosouya\jishin_ledger.jsonl"
)

JMA_LIST_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"

DEFAULT_MEMBERS = [
    {"kote": "[震]巌",   "model": "deepseek-r1:32b"},
    {"kote": "[震]理明", "model": "gemma3:4b"},
    {"kote": "[震]否問", "model": "gemma3:4b"},
    {"kote": "[震]夕凪", "model": "gemma3:4b"},
    {"kote": "[震]神楽", "model": "gemma3:4b"},
    {"kote": "[震]銀風", "model": "gemma3:4b"},
]

_HEAVY_MODELS = {"deepseek-r1:32b", "deepseek-r1:70b"}

_SYSTEM_JISHIN = """\
あなたは地震の専門家です。本震情報と余震履歴を分析し、以下の形式のみで回答してください。
余計な説明は不要です。数字と所定フォーマットだけ出力してください。

【出力形式（必須）】
M3以上: NN%
M4以上: NN%
M5以上: NN%
最大震度予測: 震度N

大森公式: 余震数は本震後の時間に対して減衰（p≈1.1）。本震Mが大きいほど余震が多く長続きする。\
"""


# ── 気象庁APIデータ取得 ──────────────────────────────────────────

def _parse_cod(cod: str) -> tuple[float | None, float | None, float | None]:
    """"+35.6+139.8-10000/" → (lat, lon, depth_km)"""
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)([+-]\d+)", cod or "")
    if not m:
        return None, None, None
    lat   = float(m.group(1))
    lon   = float(m.group(2))
    depth = abs(float(m.group(3))) / 1000  # m → km
    return lat, lon, depth


def fetch_recent_quakes(limit: int = 200) -> list[dict]:
    """
    気象庁APIから最新の地震リストを取得してパース。
    戻り値: [{eid, at, mag, depth_km, lat, lon, area, maxi, ttl}, ...]
    """
    try:
        r = requests.get(JMA_LIST_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        items = r.json()
    except Exception as e:
        return []

    quakes = []
    for item in items[:limit]:
        # 震源・震度情報のみ（マグニチュードがある行）
        mag_str = item.get("mag", "")
        if not mag_str:
            continue
        try:
            mag = float(mag_str)
        except ValueError:
            continue

        lat, lon, depth = _parse_cod(item.get("cod", ""))
        quakes.append({
            "eid":      item.get("eid", ""),
            "at":       item.get("at", ""),
            "mag":      mag,
            "depth_km": depth,
            "lat":      lat,
            "lon":      lon,
            "area":     item.get("anm", ""),
            "maxi":     item.get("maxi", ""),
        })
    return quakes


def fetch_aftershock_sequence(mainshock_eid: str | None = None,
                               min_mag: float = 2.0,
                               hours_back: int = 72) -> dict:
    """
    本震のeidを指定して、その後の余震シーケンスを取得。
    eidがNoneの場合は最新の最大地震を本震とみなす。

    戻り値:
      {
        "ok": bool,
        "mainshock": {...},
        "aftershocks": [{...}, ...],
        "n_aftershocks": int,
        "hours_analyzed": int,
      }
    """
    all_quakes = fetch_recent_quakes(500)
    if not all_quakes:
        return {"ok": False, "error": "気象庁APIからデータを取得できませんでした"}

    # 本震を特定
    if mainshock_eid:
        mainshock = next((q for q in all_quakes if q["eid"] == mainshock_eid), None)
        if not mainshock:
            return {"ok": False, "error": f"eid={mainshock_eid} の地震が見つかりません"}
    else:
        # 直近のM4以上で最大のものを本震とみなす
        candidates = [q for q in all_quakes if q["mag"] >= 4.0]
        if not candidates:
            candidates = all_quakes
        mainshock = max(candidates, key=lambda q: q["mag"])

    # 本震より後の地震を余震候補とする
    main_time = datetime.datetime.fromisoformat(mainshock["at"])
    cutoff    = main_time + datetime.timedelta(hours=hours_back)
    now       = datetime.datetime.now(tz=main_time.tzinfo)

    aftershocks = []
    for q in all_quakes:
        if q["eid"] == mainshock["eid"]:
            continue
        try:
            qt = datetime.datetime.fromisoformat(q["at"])
        except Exception:
            continue
        if main_time < qt <= min(cutoff, now) and q["mag"] >= min_mag:
            # 本震から緯度経度が近いもの（2度以内）を余震とみなす
            if mainshock["lat"] and mainshock["lon"] and q["lat"] and q["lon"]:
                dlat = abs(q["lat"] - mainshock["lat"])
                dlon = abs(q["lon"] - mainshock["lon"])
                if dlat > 2.0 or dlon > 2.0:
                    continue
            hours_after = (qt - main_time).total_seconds() / 3600
            aftershocks.append({**q, "hours_after": round(hours_after, 2)})

    aftershocks.sort(key=lambda x: x["at"])

    return {
        "ok":            True,
        "mainshock":     mainshock,
        "aftershocks":   aftershocks,
        "n_aftershocks": len(aftershocks),
        "hours_analyzed": min(hours_back, (now - main_time).total_seconds() / 3600),
    }


# ── 大森公式による参考確率 ──────────────────────────────────────

def omori_reference(mainshock_mag: float, hours_elapsed: float,
                    n_observed: int) -> dict:
    """
    大森公式の簡易推定値。
    次の24時間で発生する余震数の推定と確率（参考値）。
    """
    # 本震M→期待余震数の経験則: log10(N) ≈ M - 1.0 (Bath's law 近似)
    expected_total = 10 ** (mainshock_mag - 1.0)
    p = 1.1  # decay exponent
    c = 0.1  # hours

    # 経過時間から次の24hの期待余震数（Omori積分）
    if hours_elapsed < 1:
        hours_elapsed = 1.0
    t0 = hours_elapsed + c
    t1 = hours_elapsed + 24 + c

    if p == 1.0:
        ratio = math.log(t1 / t0)
    else:
        ratio = (t0 ** (1 - p) - t1 ** (1 - p)) / (p - 1)

    # 観測数で補正
    scale = n_observed / max(expected_total * (1 - (c / (c + hours_elapsed))), 0.1)
    expected_next24 = ratio * expected_total * scale

    # ポアソン過程として確率計算
    def poisson_prob(lam: float, k_min: int) -> float:
        if lam <= 0:
            return 0.0
        p0 = math.exp(-lam)
        return round((1 - p0) * 100, 1)

    return {
        "expected_next24h": round(expected_next24, 1),
        "prob_m3_plus":  poisson_prob(expected_next24 * 1.0, 1),
        "prob_m4_plus":  poisson_prob(expected_next24 * 0.1, 1),
        "prob_m5_plus":  poisson_prob(expected_next24 * 0.01, 1),
    }


# ── 材料テキスト生成 ─────────────────────────────────────────────

def _build_jishin_summary(seq: dict) -> str:
    ms = seq["mainshock"]
    afs = seq["aftershocks"]
    ref = omori_reference(ms["mag"], seq["hours_analyzed"], seq["n_aftershocks"])

    lines = [
        "【本震情報】",
        f"発生時刻: {ms['at']}",
        f"震源地: {ms['area']}",
        f"マグニチュード: M{ms['mag']}",
        f"深さ: {ms['depth_km']}km" if ms.get("depth_km") else "深さ: 不明",
        f"最大震度: {ms['maxi']}" if ms.get("maxi") else "",
        "",
        f"【余震履歴（本震後{seq['hours_analyzed']:.1f}時間、M2以上）】",
        f"余震数: {seq['n_aftershocks']}回",
    ]

    if afs:
        lines.append("時間(h後) | M   | 震源地")
        lines.append("─" * 40)
        for a in afs[-20:]:  # 直近20件
            lines.append(
                f"{a['hours_after']:>7.1f}h  | M{a['mag']:<4} | {a['area']}"
            )

    lines += [
        "",
        "【大森公式による参考値（機械的推定）】",
        f"次の24h期待余震数: {ref['expected_next24h']}回",
        f"M3以上の余震確率（参考）: {ref['prob_m3_plus']}%",
        f"M4以上の余震確率（参考）: {ref['prob_m4_plus']}%",
        f"M5以上の余震確率（参考）: {ref['prob_m5_plus']}%",
    ]
    return "\n".join(l for l in lines if l is not None)


# ── 村人への問い合わせ ──────────────────────────────────────────

def _ask_one_jishin(member: dict, summary: str, timeout: float = 90.0) -> dict:
    kote  = member.get("kote", "?")
    model = member.get("model", "")

    prefix = "M3以上: "
    completion_prompt = (
        f"{_SYSTEM_JISHIN}\n\n"
        f"{summary}\n\n"
        f"{prefix}"
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": completion_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=timeout,
        )
        resp = (r.json().get("response", "") or "").strip()
        text = prefix + resp
    except Exception as e:
        return {"kote": kote, "model": model, "probs": {}, "ok": False, "error": str(e)}

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    print(f"[jishin] {kote} raw(150)={text[:150].replace(chr(10),' ')}")
    probs, shindo = _parse_jishin_response(text)
    return {
        "kote":   kote,
        "model":  model,
        "probs":  probs,
        "shindo": shindo,
        "raw":    text[:600],
        "ok":     bool(probs),
    }


def _parse_jishin_response(text: str) -> tuple[dict, str]:
    """
    "M3以上: NN%", "M4以上: NN%", "M5以上: NN%",
    "最大震度予測: 震度N" をパース。
    """
    probs: dict[str, float] = {}
    for key in ["M3以上", "M4以上", "M5以上"]:
        m = re.search(rf"{key}\s*[:：]\s*(\d+(?:\.\d+)?)\s*[%％]", text)
        if m:
            probs[key] = float(m.group(1))

    shindo = ""
    m = re.search(r"最大震度予測\s*[:：]\s*震度\s*([\d.]+)", text)
    if m:
        shindo = m.group(1)

    return probs, shindo


def _aggregate_jishin(all_results: list[dict]) -> dict:
    """確率の平均と震度の最頻値を集計"""
    keys = ["M3以上", "M4以上", "M5以上"]
    probs_by_key: dict[str, list[float]] = {k: [] for k in keys}
    shindo_votes: dict[str, int] = {}

    for res in all_results:
        if not res.get("ok"):
            continue
        for k in keys:
            if k in res.get("probs", {}):
                probs_by_key[k].append(res["probs"][k])
        s = res.get("shindo", "")
        if s:
            shindo_votes[s] = shindo_votes.get(s, 0) + 1

    avg_probs = {
        k: round(statistics.mean(v), 1) if v else None
        for k, v in probs_by_key.items()
    }
    best_shindo = max(shindo_votes, key=shindo_votes.get) if shindo_votes else None

    return {
        "avg_probs":    avg_probs,
        "shindo":       best_shindo,
        "shindo_votes": shindo_votes,
        "n_responded":  sum(1 for r in all_results if r.get("ok")),
    }


# ── 台帳 ───────────────────────────────────────────────────────

def _append_ledger(entry: dict) -> str:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return LEDGER_PATH


# ── 過去予測の自動照合 ──────────────────────────────────────────

def resolve_past(limit: int = 50) -> dict:
    """
    jishin_ledger.jsonl の resolved=null なエントリを読み込み、
    気象庁APIから実際の余震データを再取得して resolved フィールドを埋める。

    対象: 本震から24時間以上経過したエントリのみ（まだ24h経っていないものはスキップ）

    resolved フィールドの形式:
      {
        "resolved_at": "ISO8601",
        "hours_window": 24,
        "actual_aftershocks": {"M3以上": int, "M4以上": int, "M5以上": int},
        "max_mag": float,
        "max_shindo": str | null,
        "hit_m3": bool,
        "hit_m4": bool,
        "hit_m5": bool,
      }

    戻り値: {"ok": True, "n_resolved": int, "n_skipped": int, "entries": [...]}
    """
    import pathlib

    path = pathlib.Path(LEDGER_PATH)
    if not path.exists():
        return {"ok": False, "error": "台帳が見つかりません"}

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            entries.append({"_raw": line})

    targets = [
        e for e in entries
        if isinstance(e, dict) and e.get("resolved") is None and e.get("mainshock")
    ]
    targets = targets[:limit]

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    n_resolved = 0
    n_skipped  = 0
    result_rows = []

    for entry in targets:
        eid  = entry.get("id", "?")
        ms   = entry.get("mainshock", {})
        ms_at_str = ms.get("at", "")
        try:
            ms_at = datetime.datetime.fromisoformat(ms_at_str)
            if ms_at.tzinfo is None:
                ms_at = ms_at.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            n_skipped += 1
            result_rows.append({"id": eid, "ok": False, "error": "本震日時パース失敗"})
            continue

        hours_elapsed = (now - ms_at).total_seconds() / 3600
        if hours_elapsed < 24:
            n_skipped += 1
            result_rows.append({
                "id": eid, "ok": False,
                "error": f"本震から{hours_elapsed:.1f}h経過（24h未満のためスキップ）",
            })
            continue

        print(f"[jishin] resolve: {eid} M{ms.get('mag','?')} {ms.get('area','')} "
              f"({hours_elapsed:.0f}h経過)", flush=True)

        mainshock_eid_val = ms.get("eid")
        seq = fetch_aftershock_sequence(mainshock_eid_val, min_mag=2.0, hours_back=24)
        time.sleep(0.5)

        if not seq.get("ok"):
            n_skipped += 1
            result_rows.append({"id": eid, "ok": False,
                                 "error": seq.get("error", "データ取得失敗")})
            continue

        aftershocks = seq.get("aftershocks", [])
        actual_m3 = sum(1 for a in aftershocks if a["mag"] >= 3.0)
        actual_m4 = sum(1 for a in aftershocks if a["mag"] >= 4.0)
        actual_m5 = sum(1 for a in aftershocks if a["mag"] >= 5.0)
        max_mag   = max((a["mag"] for a in aftershocks), default=0.0)
        max_shindo_raw = max(
            (a.get("maxi", "") for a in aftershocks if a.get("maxi")),
            default=None,
        )

        # 予想確率を取得（avg_probs から）
        avg_probs = entry.get("aggregated", {}).get("avg_probs", {})
        prob_m3 = avg_probs.get("M3以上", 0) or 0
        prob_m4 = avg_probs.get("M4以上", 0) or 0
        prob_m5 = avg_probs.get("M5以上", 0) or 0

        resolved = {
            "resolved_at":  now.isoformat(timespec="seconds"),
            "hours_window": 24,
            "actual_aftershocks": {
                "M3以上": actual_m3,
                "M4以上": actual_m4,
                "M5以上": actual_m5,
            },
            "max_mag":    round(max_mag, 1),
            "max_shindo": max_shindo_raw,
            "hit_m3": bool(prob_m3 > 50 and actual_m3 > 0) or bool(prob_m3 <= 50 and actual_m3 == 0),
            "hit_m4": bool(prob_m4 > 50 and actual_m4 > 0) or bool(prob_m4 <= 50 and actual_m4 == 0),
            "hit_m5": bool(prob_m5 > 50 and actual_m5 > 0) or bool(prob_m5 <= 50 and actual_m5 == 0),
        }
        entry["resolved"] = resolved
        n_resolved += 1
        result_rows.append({
            "id": eid, "ok": True,
            "actual": resolved["actual_aftershocks"],
            "max_mag": resolved["max_mag"],
            "hit_m3": resolved["hit_m3"],
            "hit_m4": resolved["hit_m4"],
            "hit_m5": resolved["hit_m5"],
        })
        print(f"[jishin]   → M3+:{actual_m3} M4+:{actual_m4} M5+:{actual_m5} "
              f"hit_m3={resolved['hit_m3']}")

    # 台帳を書き戻す
    updated_lines = []
    for e in entries:
        if isinstance(e, dict) and "_raw" not in e:
            updated_lines.append(json.dumps(e, ensure_ascii=False))
        elif isinstance(e, dict) and "_raw" in e:
            updated_lines.append(e["_raw"])
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(f"[jishin] resolve_past 完了: 解決={n_resolved} スキップ={n_skipped}")
    return {
        "ok":         True,
        "n_resolved": n_resolved,
        "n_skipped":  n_skipped,
        "entries":    result_rows,
    }


# ── メイン実行関数 ──────────────────────────────────────────────

def run(mainshock_eid: str | None = None, members: list | None = None) -> dict:
    """
    余震予測を実行。

    mainshock_eid: 気象庁のeid（例: "20261101152834"）。
                   Noneの場合は最新の最大地震を本震とみなす。
    """
    members = members or DEFAULT_MEMBERS

    # ── 1. 余震シーケンス取得 ──────────────────────────────────
    print(f"[jishin] 余震シーケンス取得: eid={mainshock_eid or '最新最大'}")
    seq = fetch_aftershock_sequence(mainshock_eid)
    if not seq.get("ok"):
        return {"ok": False, "error": seq.get("error", "データ取得失敗")}

    ms = seq["mainshock"]
    print(f"[jishin] 本震: M{ms['mag']} {ms['area']} {ms['at'][:16]}")
    print(f"[jishin] 余震数: {seq['n_aftershocks']}回 ({seq['hours_analyzed']:.1f}h分)")

    # ── 2. 大森公式参考値 ─────────────────────────────────────
    ref = omori_reference(ms["mag"], seq["hours_analyzed"], seq["n_aftershocks"])
    print(f"[jishin] 大森参考: 次24h期待{ref['expected_next24h']}回 "
          f"M3+:{ref['prob_m3_plus']}% M4+:{ref['prob_m4_plus']}%")

    # ── 3. 材料テキスト生成 ───────────────────────────────────
    summary = _build_jishin_summary(seq)

    # ── 4. 独立見積もり ───────────────────────────────────────
    light_members = [m for m in members if m.get("model") not in _HEAVY_MODELS]
    heavy_members = [m for m in members if m.get("model") in _HEAVY_MODELS]
    all_results: list[dict] = []

    if light_members:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(light_members)) as ex:
            futs = {ex.submit(_ask_one_jishin, m, summary, 90.0): m for m in light_members}
            for fut in concurrent.futures.as_completed(futs, timeout=120):
                res = fut.result()
                all_results.append(res)
                ok_str = "OK" if res.get("ok") else "NG"
                print(f"[jishin] {res['kote']} {ok_str}")

    for m in heavy_members:
        print(f"[jishin] 重量モデル: {m['kote']} ({m['model']})")
        res = _ask_one_jishin(m, summary, 600.0)
        all_results.append(res)
        ok_str = "OK" if res.get("ok") else "NG"
        print(f"[jishin] {res['kote']} {ok_str}")

    # ── 5. 集計 ───────────────────────────────────────────────
    agg = _aggregate_jishin(all_results)
    print(f"\n【村の予想】")
    for k, v in agg["avg_probs"].items():
        print(f"  {k}: {v}%")
    if agg.get("shindo"):
        print(f"  最大震度予測: 震度{agg['shindo']}")

    # ── 6. 台帳に記録 ─────────────────────────────────────────
    entry = {
        "id":           "j_" + time.strftime("%Y%m%d%H%M%S"),
        "at":           datetime.datetime.now().isoformat(timespec="seconds"),
        "mainshock":    ms,
        "n_aftershocks": seq["n_aftershocks"],
        "hours_analyzed": seq["hours_analyzed"],
        "omori_ref":    ref,
        "member_results": [
            {"kote": r["kote"], "model": r["model"],
             "ok": r.get("ok", False), "probs": r.get("probs", {}),
             "shindo": r.get("shindo", "")}
            for r in all_results
        ],
        "aggregated":   agg,
        "resolved":     None,  # 24h後に実際の最大余震を記録
    }
    ledger_path = _append_ledger(entry)
    print(f"[jishin] 台帳に記録: {ledger_path}")

    return {
        "ok":          True,
        "id":          entry["id"],
        "mainshock":   ms,
        "n_aftershocks": seq["n_aftershocks"],
        "omori_ref":   ref,
        "aggregated":  agg,
        "members":     all_results,
        "ledger":      ledger_path,
    }


# ── 前兆モニター ────────────────────────────────────────────────

NOAA_KP_URL   = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
BSKY_SEARCH_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

# NICT GEONET TEC（旧 swc.nict.go.jp は廃止 → www2.nict.go.jp へ移転）
# リアルタイム版: 10分間隔、最新6時間 (候補1)
_NICT_TEC_BASE1 = "https://www2.nict.go.jp/spe/gps/REALTIME_GEONET"
# 準リアルタイム版: 約3時間遅延 (候補2)、6時間GIFアニメも提供
_NICT_TEC_BASE2 = "https://aer-nc-web.nict.go.jp/GPS/QR_GEONET"
_NICT_TEC_GIF   = f"{_NICT_TEC_BASE2}/Latest_6hours_Amap.gif"


def _nict_tec_latest_url() -> tuple[str, str]:
    """最新フレームのJPEG URLを構築して返す (url, timestamp_str)。"""
    now_jst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    # 10分単位に切り捨て
    minute = (now_jst.minute // 10) * 10
    dt = now_jst.replace(minute=minute, second=0, microsecond=0)
    doy = dt.timetuple().tm_yday  # 通年通算日
    hh  = dt.strftime("%H")
    mm  = dt.strftime("%M")
    mmdd = dt.strftime("%m%d")
    yyyy = dt.strftime("%Y")
    fname = f"{hh}_{mm}_{mmdd}_{yyyy}_Amap.jpg"
    url = f"{_NICT_TEC_BASE1}/AMAP/{yyyy}/{doy:03d}/{fname}"
    ts  = dt.strftime("%Y-%m-%d %H:%M JST")
    return url, ts

BSKY_KEYWORDS = ["地震雲", "地震前兆", "地鳴り", "地震の前兆", "earthquake cloud"]

def _kp_level(kp: float) -> str:
    if kp >= 8: return "激しい嵐"
    if kp >= 6: return "強い嵐"
    if kp >= 5: return "中程度の嵐"
    if kp >= 4: return "活発"
    if kp >= 2: return "穏やか"
    return "静穏"

def _kp_color(kp: float) -> str:
    if kp >= 6: return "danger"
    if kp >= 4: return "warn"
    return "normal"

def fetch_precursors() -> dict:
    """
    地震前兆関連の環境データを取得。
    - Kpインデックス（地磁気活動）: NOAA
    - 電離層TEC: NICT (失敗時はスキップ)
    """
    result: dict = {"ok": True, "fetched_at": datetime.datetime.now().isoformat(timespec="seconds")}

    # ── Kpインデックス (NOAA) ───────────────────────────────────
    try:
        r = requests.get(NOAA_KP_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        rows = r.json()
        # 先頭行がヘッダー: ["time_tag", "kp", "kp_fraction", "a_running", "station_count"]
        # 最新3時間分（直近5件）を取る
        data_rows = [row for row in rows[1:] if row[1] not in (None, "")]
        recent = data_rows[-5:] if len(data_rows) >= 5 else data_rows
        kp_values = []
        for row in recent:
            try:
                kp_values.append({"time": row[0], "kp": float(row[1])})
            except (TypeError, ValueError):
                pass
        current_kp = kp_values[-1]["kp"] if kp_values else None
        result["kp"] = {
            "current": current_kp,
            "level":   _kp_level(current_kp) if current_kp is not None else "不明",
            "color":   _kp_color(current_kp) if current_kp is not None else "normal",
            "history": kp_values,
            "source":  "NOAA SWPC",
        }
    except Exception as e:
        result["kp"] = {"error": str(e)}

    # ── NICT 電離層TEC（www2.nict.go.jp リアルタイム版） ──────────
    try:
        img_url, ts = _nict_tec_latest_url()
        hdr = {"User-Agent": "Mozilla/5.0", "Referer": _NICT_TEC_BASE1 + "/"}
        r = requests.head(img_url, timeout=8, headers=hdr, allow_redirects=True)
        if r.status_code == 200:
            result["tec"] = {
                "ok":      True,
                "img_url": img_url,
                "gif_url": _NICT_TEC_GIF,
                "latest":  ts,
                "source":  "NICT GEONET リアルタイム（www2.nict.go.jp）",
                "note":    "10分間隔・最新6時間",
            }
        else:
            # フォールバック: 準リアルタイム版 GIF（確定版）
            r2 = requests.head(_NICT_TEC_GIF, timeout=8, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
            if r2.status_code == 200:
                result["tec"] = {
                    "ok":      True,
                    "gif_url": _NICT_TEC_GIF,
                    "latest":  ts,
                    "source":  "NICT GEONET 準リアルタイム（aer-nc-web.nict.go.jp）",
                    "note":    "約3時間遅延・6時間GIF",
                }
            else:
                result["tec"] = {"ok": False, "note": f"TEC HTTP {r.status_code} / {r2.status_code}"}
    except Exception as e:
        result["tec"] = {"ok": False, "error": str(e)[:80], "note": "NICT TEC取得失敗（任意）"}

    # ── Bluesky 前兆キーワード検索 ─────────────────────────────
    bsky_results: list[dict] = []
    bsky_total = 0
    try:
        for kw in BSKY_KEYWORDS:
            r = requests.get(
                BSKY_SEARCH_URL,
                params={"q": kw, "limit": 10, "sort": "latest"},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=8,
            )
            if r.status_code == 200:
                posts = r.json().get("posts", [])
                bsky_total += len(posts)
                for p in posts[:3]:
                    rec = p.get("record", {})
                    bsky_results.append({
                        "keyword": kw,
                        "text":    rec.get("text", "")[:100],
                        "at":      rec.get("createdAt", "")[:16],
                        "handle":  p.get("author", {}).get("handle", ""),
                    })
        result["bluesky"] = {
            "ok":      True,
            "total":   bsky_total,
            "posts":   bsky_results[:10],
            "keywords": BSKY_KEYWORDS,
        }
    except Exception as e:
        result["bluesky"] = {"ok": False, "error": str(e)}

    # ── Hi-net 高感度地震観測網（防災科研）────────────────────────
    # Hi-net はログイン不要のCSV公開データ: 最新震源リスト
    HINET_URL = "https://www.hinet.bosai.go.jp/REGS/download/catalog/his_eqlist.csv"
    try:
        r = requests.get(HINET_URL, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://www.hinet.bosai.go.jp/"})
        # CSV: Origin Time, Lat, Lon, Dep, M, Region
        lines = r.text.splitlines()
        hinet_quakes: list[dict] = []
        for line in lines[1:21]:  # 先頭20件
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                hinet_quakes.append({
                    "at":     parts[0].strip(),
                    "lat":    float(parts[1]) if parts[1].strip() else None,
                    "lon":    float(parts[2]) if parts[2].strip() else None,
                    "depth":  float(parts[3]) if parts[3].strip() else None,
                    "mag":    float(parts[4]) if parts[4].strip() else None,
                    "region": parts[5].strip() if len(parts) > 5 else "",
                })
            except (ValueError, IndexError):
                continue
        result["hinet"] = {
            "ok":     True,
            "count":  len(hinet_quakes),
            "quakes": hinet_quakes,
            "source": "Hi-net（防災科研 高感度地震観測網）",
        }
    except Exception as e:
        result["hinet"] = {"ok": False, "error": str(e)[:80],
                            "note": "Hi-net CSV取得失敗（任意）"}
        print(f"[jishin] Hi-net スキップ: {e}")

    # ── tenki.jp 地震情報 ─────────────────────────────────────────
    TENKI_EQ_URL = "https://tenki.jp/bousai/earthquake/"
    try:
        r = requests.get(TENKI_EQ_URL, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "utf-8"
        from bs4 import BeautifulSoup as _BS
        soup = _BS(r.text, "html.parser")
        tenki_quakes: list[dict] = []
        # 震度別地震一覧テーブル: .recent-eq-list, .earthquake-list
        for item in soup.select(".earthquake-list li, .recent-eq-list li, "
                                 ".quake-list-item, article.quake")[:15]:
            txt = item.get_text(separator=" ", strip=True)
            if not txt:
                continue
            tenki_quakes.append({"text": txt[:100]})
        # フォールバック: table行
        if not tenki_quakes:
            for tr in soup.select("table tr")[1:16]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 3:
                    tenki_quakes.append({"row": tds[:6]})
        result["tenki"] = {
            "ok":     True,
            "count":  len(tenki_quakes),
            "quakes": tenki_quakes,
            "source": "tenki.jp 地震情報",
            "url":    TENKI_EQ_URL,
        }
    except Exception as e:
        result["tenki"] = {"ok": False, "error": str(e)[:80],
                            "note": "tenki.jp取得失敗（任意）"}
        print(f"[jishin] tenki.jp スキップ: {e}")

    # ── 総合判定 ───────────────────────────────────────────────
    kp = (result.get("kp") or {}).get("current")
    alerts = []
    if kp is not None and kp >= 5:
        alerts.append(f"地磁気嵐 Kp={kp} ({_kp_level(kp)})")
    bsky_total_val = (result.get("bluesky") or {}).get("total", 0)
    if bsky_total_val >= 15:
        alerts.append(f"SNS前兆報告 {bsky_total_val}件（Bluesky）")
    hinet_m4 = sum(
        1 for q in result.get("hinet", {}).get("quakes", [])
        if isinstance(q.get("mag"), float) and q["mag"] >= 4.0
    )
    if hinet_m4 >= 3:
        alerts.append(f"Hi-net M4+地震 {hinet_m4}件（高感度観測網）")
    result["alerts"] = alerts
    result["alert_level"] = "high" if alerts else "normal"

    return result


# ── GEONET 電子基準点 ────────────────────────────────────────────

# 主要電子基準点（国土地理院GEONET、地震活動域を中心に約60点）
# (点名, 点ID, 緯度, 経度)
_GEONET_STATIONS = [
    ("稚内",     "0010",  45.416,  141.683),
    ("旭川",     "0020",  43.771,  142.365),
    ("札幌",     "0030",  43.062,  141.329),
    ("函館",     "0040",  41.769,  140.729),
    ("釧路",     "0050",  42.984,  144.381),
    ("帯広",     "0060",  42.924,  143.196),
    ("網走",     "0070",  44.013,  144.273),
    ("青森",     "0100",  40.820,  140.747),
    ("盛岡",     "0110",  39.703,  141.154),
    ("仙台",     "0120",  38.260,  140.882),
    ("秋田",     "0130",  39.718,  140.102),
    ("山形",     "0140",  38.240,  140.340),
    ("福島",     "0150",  37.750,  140.473),
    ("水戸",     "0210",  36.371,  140.471),
    ("宇都宮",   "0220",  36.549,  139.883),
    ("前橋",     "0230",  36.391,  139.063),
    ("さいたま", "0240",  35.861,  139.649),
    ("千葉",     "0250",  35.605,  140.123),
    ("東京",     "0260",  35.689,  139.692),
    ("横浜",     "0270",  35.444,  139.638),
    ("新潟",     "0310",  37.916,  139.023),
    ("富山",     "0320",  36.695,  137.213),
    ("金沢",     "0330",  36.561,  136.656),
    ("福井",     "0340",  36.064,  136.219),
    ("甲府",     "0350",  35.664,  138.569),
    ("長野",     "0360",  36.651,  138.181),
    ("岐阜",     "0370",  35.391,  136.722),
    ("静岡",     "0380",  34.976,  138.383),
    ("名古屋",   "0390",  35.181,  136.907),
    ("津",       "0400",  34.730,  136.509),
    ("大津",     "0410",  35.004,  135.869),
    ("京都",     "0420",  35.012,  135.768),
    ("大阪",     "0430",  34.686,  135.520),
    ("神戸",     "0440",  34.690,  135.195),
    ("奈良",     "0450",  34.685,  135.805),
    ("和歌山",   "0460",  34.226,  135.168),
    ("鳥取",     "0510",  35.501,  134.238),
    ("松江",     "0520",  35.472,  133.051),
    ("岡山",     "0530",  34.661,  133.934),
    ("広島",     "0540",  34.396,  132.459),
    ("山口",     "0550",  34.186,  131.470),
    ("徳島",     "0560",  34.066,  134.559),
    ("高松",     "0570",  34.340,  134.043),
    ("松山",     "0580",  33.841,  132.766),
    ("高知",     "0590",  33.560,  133.531),
    ("福岡",     "0610",  33.606,  130.418),
    ("佐賀",     "0620",  33.249,  130.299),
    ("長崎",     "0630",  32.745,  129.874),
    ("熊本",     "0640",  32.790,  130.742),
    ("大分",     "0650",  33.238,  131.613),
    ("宮崎",     "0660",  31.911,  131.424),
    ("鹿児島",   "0670",  31.560,  130.558),
    ("那覇",     "0760",  26.212,  127.679),
    ("石垣",     "0770",  24.337,  124.164),
    # 小笠原・南方諸島（硫黄島は日本最大の地殻変動点、火山隆起で30年数m超）
    ("硫黄島",   "0780",  24.784,  141.323),
    ("父島",     "0775",  27.094,  142.185),
    ("南鳥島",   "0785",  24.288,  153.979),
    # 活断層近傍・太平洋岸の追加点
    ("銚子",     "0255",  35.735,  140.869),
    ("勝浦",     "0256",  35.149,  140.316),
    ("館山",     "0257",  34.995,  139.868),
    ("沼津",     "0381",  35.096,  138.861),
    ("浜松",     "0382",  34.711,  137.726),
    ("気仙沼",   "0121",  38.907,  141.569),
    ("石巻",     "0122",  38.433,  141.303),
    ("相馬",     "0151",  37.797,  140.916),
    ("いわき",   "0152",  37.050,  140.888),
    ("日立",     "0211",  36.599,  140.651),
]

MEKIRA_BASE = "https://mekira.gsi.go.jp"
_mekira_station_cache: dict | None = None   # {station_id: {name, lat, lon}}
_mekira_vector_cache: dict | None = None    # {station_id: [east_cm, north_cm, up_cm]}


def _mekira_stations() -> dict:
    """mekira.gsi.go.jp から観測点一覧を取得してIDマップを返す（キャッシュあり）。"""
    global _mekira_station_cache
    if _mekira_station_cache is not None:
        return _mekira_station_cache
    try:
        r = requests.get(f"{MEKIRA_BASE}/api/getObservationStations",
                         params={"decimated": 0}, timeout=8,
                         headers={"Referer": MEKIRA_BASE, "User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        _mekira_station_cache = {
            s["id"]: {"name": s["name"], "lat": float(s["b"]), "lon": float(s["l"])}
            for s in r.json()
        }
    except Exception:
        _mekira_station_cache = {}
    return _mekira_station_cache


def _mekira_vectors(term: str = "10years") -> dict:
    """mekira から変動ベクトルを取得してIDマップを返す（キャッシュあり）。
    戻り値: {station_id: {"east_cm": float, "north_cm": float, "up_cm": float, "total_cm": float}}
    """
    global _mekira_vector_cache
    if _mekira_vector_cache is not None:
        return _mekira_vector_cache
    try:
        r = requests.get(f"{MEKIRA_BASE}/api/getCoordinateTransformationVectors",
                         params={"term": term, "decimated": 0}, timeout=10,
                         headers={"Referer": MEKIRA_BASE, "User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        result = {}
        for vec in data.get("vectors", []):
            sid, east, north, up = str(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])
            total = round(math.sqrt(east**2 + north**2 + up**2), 4)
            result[sid] = {"east_cm": east, "north_cm": north, "up_cm": up, "total_cm": total}
        _mekira_vector_cache = result
        return result
    except Exception:
        _mekira_vector_cache = {}
        return {}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の距離(km)"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def fetch_geonet(lat: float, lon: float, n: int = 3) -> dict:
    """
    指定座標に最も近いGEONET電子基準点を探し、
    mekira.gsi.go.jp の公開APIから10年間の変動ベクトルを取得する。
    """
    if lat is None or lon is None:
        return {"ok": False, "error": "座標不明"}

    # mekira から全観測点を取得して距離順にソート
    mekira_st = _mekira_stations()
    vectors    = _mekira_vectors()

    if mekira_st:
        # mekira の全観測点から計算
        stations_with_dist = []
        for sid, info in mekira_st.items():
            dist = _haversine(lat, lon, info["lat"], info["lon"])
            entry = {
                "name": info["name"], "id": sid,
                "lat": info["lat"], "lon": info["lon"],
                "dist_km": round(dist, 1),
            }
            if sid in vectors:
                v = vectors[sid]
                entry["vector"] = {
                    "east_cm": v["east_cm"],
                    "north_cm": v["north_cm"],
                    "up_cm": v["up_cm"],
                    "total_cm": v["total_cm"],
                }
            stations_with_dist.append(entry)
        stations_with_dist.sort(key=lambda x: x["dist_km"])
        nearest = stations_with_dist[:n]
        source = "国土地理院 mekira（地殻変動情報表示サイト）直近10年"
    else:
        # フォールバック: ハードコードのリストを使用
        stations_with_dist = []
        for name, pid, slat, slon in _GEONET_STATIONS:
            dist = _haversine(lat, lon, slat, slon)
            stations_with_dist.append({"name": name, "id": pid, "lat": slat, "lon": slon, "dist_km": round(dist, 1)})
        stations_with_dist.sort(key=lambda x: x["dist_km"])
        nearest = stations_with_dist[:n]
        source = "国土地理院 GEONET（オフライン）"

    return {
        "ok":        True,
        "epicenter": {"lat": lat, "lon": lon},
        "nearest":   nearest,
        "source":    source,
        "url":       f"https://mekira.gsi.go.jp/#lat={lat}&lon={lon}&zoom=9",
    }


if __name__ == "__main__":
    import sys
    eid = sys.argv[1] if len(sys.argv) > 1 else None
    result = run(eid)
    print(json.dumps(result, ensure_ascii=False, indent=2))
