"""
jishin_precursor.py — 前震モニタリング・事前予測モジュール
=============================================================
工程:
  ① 気象庁APIから最新地震リストを取得
  ② 日本の主要地域ごとに過去48hの活動量を集計
  ③ 蓄積ログと比較してベースラインとの乖離を計算
  ④ LAが「この地域でM5以上が今後24h以内に起きる確率」を予測
  ⑤ precursor_ledger.jsonl に記録

注: 開始直後はベースラインが薄い。2〜3週間蓄積すると比較精度が上がる。
"""
from __future__ import annotations

import os
import json
import math
import datetime
import statistics
import concurrent.futures
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

LEDGER_PATH = os.environ.get(
    "PRECURSOR_LEDGER",
    r"C:\Users\USER\workspace\yosouya\jishin_precursor_ledger.jsonl"
)

JMA_LIST_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"

# NOAA宇宙天気API
NOAA_KP_URL   = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
NOAA_IMF_URL  = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
NOAA_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

# ── 地域定義（緯度・経度の矩形） ────────────────────────────────
REGIONS = {
    "北海道":   {"lat": (42.0, 46.0), "lon": (139.0, 146.0)},
    "東北":     {"lat": (37.0, 42.0), "lon": (139.0, 142.0)},
    "関東":     {"lat": (34.5, 37.0), "lon": (138.5, 141.5)},
    "東海":     {"lat": (33.5, 36.0), "lon": (136.0, 139.5)},
    "北陸":     {"lat": (35.5, 38.0), "lon": (135.5, 138.5)},
    "近畿":     {"lat": (33.0, 35.5), "lon": (134.0, 137.0)},
    "中国四国": {"lat": (32.0, 35.0), "lon": (130.0, 135.0)},
    "九州":     {"lat": (30.0, 34.0), "lon": (129.0, 132.5)},
    "沖縄":     {"lat": (24.0, 28.0), "lon": (122.0, 130.0)},
}

DEFAULT_MEMBERS = [
    {"kote": "[震]巌",   "model": "deepseek-r1:32b"},
    {"kote": "[震]理明", "model": "gemma3:4b"},
    {"kote": "[震]否問", "model": "gemma3:4b"},
    {"kote": "[震]夕凪", "model": "gemma3:4b"},
    {"kote": "[震]神楽", "model": "gemma3:4b"},
    {"kote": "[震]銀風", "model": "gemma3:4b"},
]

_SYSTEM_PRECURSOR = """\
あなたは地震の専門家です。与えられた地域別の地震活動データを分析し、
今後24時間以内にその地域でM5以上の地震が発生する確率を推定してください。

以下の形式のみで回答してください（余計な説明不要）:

地域名: XX%

根拠にしてよい情報:
- 通常より地震活動が活発な地域は確率が上がる（前震の可能性）
- M3以上が短時間に複数発生している地域は注意
- 活動量が通常の3倍以上なら有意な増加とみなす
- ただし日本全体でM5以上が24h内に起きる確率は通常約30〜40%
- 活動が静穏な地域は確率を低め（5〜15%）に設定する
"""


# ── 電磁気・宇宙天気データ取得 ──────────────────────────────────

import re


def fetch_geomagnetic() -> dict:
    """
    NOAA 惑星Kp指数を取得（過去3時間平均・最大値）。
    Kp≥5で地磁気嵐、Kp≥7で大規模嵐。
    """
    try:
        r = requests.get(NOAA_KP_URL, timeout=10)
        data = r.json()
        # 末尾3時間分（1分値 → 180件）
        recent = [d for d in data if d.get("kp") is not None][-180:]
        if not recent:
            return {"ok": False, "reason": "データなし"}
        def _kp_float(v) -> float:
            # "5P" "3o" "4-" など記号混じり → 数値部分だけ取り出す
            return float(re.sub(r"[^0-9.]", "", str(v)) or "0")
        kp_vals = [_kp_float(d["kp"]) for d in recent]
        kp_avg  = round(sum(kp_vals) / len(kp_vals), 2)
        kp_max  = round(max(kp_vals), 2)
        level   = "STORM" if kp_max >= 5 else "ACTIVE" if kp_max >= 3 else "QUIET"
        return {
            "ok": True,
            "kp_avg_3h": kp_avg,
            "kp_max_3h": kp_max,
            "level": level,
            "latest_time": recent[-1].get("time_tag", ""),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def fetch_imf() -> dict:
    """
    NOAA リアルタイム太陽風磁場（IMF）を取得。
    Bz南向き（負値）が強いほど地磁気活動が活発。
    """
    try:
        r = requests.get(NOAA_IMF_URL, timeout=10)
        data = r.json()
        # 末尾1時間分（1分値 → 60件）
        recent = [d for d in data
                  if d.get("bz_gsm") is not None and d.get("bt") is not None][-60:]
        if not recent:
            return {"ok": False, "reason": "データなし"}
        bz_vals = [float(d["bz_gsm"]) for d in recent]
        bt_vals = [float(d["bt"])     for d in recent]
        bz_avg  = round(sum(bz_vals) / len(bz_vals), 2)
        bt_avg  = round(sum(bt_vals) / len(bt_vals), 2)
        bz_min  = round(min(bz_vals), 2)
        level   = "STORM" if bz_min <= -10 else "ACTIVE" if bz_min <= -5 else "QUIET"
        return {
            "ok": True,
            "bz_gsm_avg_1h": bz_avg,
            "bz_gsm_min_1h": bz_min,
            "bt_avg_1h": bt_avg,
            "level": level,
            "latest_time": recent[-1].get("time_tag", ""),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def fetch_xray() -> dict:
    """
    NOAA GOES衛星X線フラックスを取得（電離層擾乱の指標）。
    0.1-0.8nm帯域: C1=1e-6, M1=1e-5, X1=1e-4
    """
    try:
        r = requests.get(NOAA_XRAY_URL, timeout=10)
        data = r.json()
        # 0.1-0.8nm帯のみ（電離層影響が大きい帯域）
        band = [d for d in data if d.get("energy") == "0.1-0.8nm"][-60:]
        if not band:
            return {"ok": False, "reason": "データなし"}
        flux_vals = [float(d["flux"]) for d in band if d.get("flux") is not None]
        if not flux_vals:
            return {"ok": False, "reason": "フラックスデータなし"}
        flux_max = max(flux_vals)
        # フレアクラス判定
        if flux_max >= 1e-4:
            flare_class = "X"
        elif flux_max >= 1e-5:
            flare_class = "M"
        elif flux_max >= 1e-6:
            flare_class = "C"
        elif flux_max >= 1e-7:
            flare_class = "B"
        else:
            flare_class = "A"
        return {
            "ok": True,
            "flux_max_1h": flux_max,
            "flare_class": flare_class,
            "level": "ACTIVE" if flare_class in ("M", "X") else "QUIET",
            "latest_time": band[-1].get("time_tag", ""),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ── データ取得・集計 ────────────────────────────────────────────

def _parse_cod(cod: str) -> tuple[float | None, float | None]:
    m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", cod or "")
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def fetch_recent_quakes(limit: int = 500) -> list[dict]:
    try:
        r = requests.get(JMA_LIST_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        items = r.json()
    except Exception:
        return []

    quakes = []
    for item in items[:limit]:
        mag_str = item.get("mag", "")
        if not mag_str:
            continue
        try:
            mag = float(mag_str)
        except ValueError:
            continue
        lat, lon = _parse_cod(item.get("cod", ""))
        quakes.append({
            "eid":  item.get("eid", ""),
            "at":   item.get("at", ""),
            "mag":  mag,
            "lat":  lat,
            "lon":  lon,
            "area": item.get("anm", ""),
            "maxi": item.get("maxi", ""),
        })
    return quakes


def _region_of(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    for name, bbox in REGIONS.items():
        if bbox["lat"][0] <= lat <= bbox["lat"][1] and bbox["lon"][0] <= lon <= bbox["lon"][1]:
            return name
    return None


def aggregate_activity(quakes: list[dict], hours_back: int = 48, min_mag: float = 2.0) -> dict[str, dict]:
    """
    地域ごとに過去N時間の地震活動を集計。
    戻り値: {地域名: {count_all, count_m3, count_m4, max_mag, quakes}}
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=hours_back)

    activity: dict[str, dict] = {r: {"count_all": 0, "count_m3": 0, "count_m4": 0,
                                      "max_mag": 0.0, "quakes": []} for r in REGIONS}

    for q in quakes:
        if q["mag"] < min_mag:
            continue
        try:
            at_str = q["at"]
            # 気象庁のat形式: "20261108010400" (UTC+9)
            if len(at_str) == 14 and at_str.isdigit():
                naive = datetime.datetime.strptime(at_str, "%Y%m%d%H%M%S")
                at = naive.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
            else:
                at = datetime.datetime.fromisoformat(at_str)
        except Exception:
            continue
        if at < cutoff:
            continue

        region = _region_of(q["lat"], q["lon"])
        if region is None:
            continue

        a = activity[region]
        a["count_all"] += 1
        if q["mag"] >= 3.0:
            a["count_m3"] += 1
        if q["mag"] >= 4.0:
            a["count_m4"] += 1
        if q["mag"] > a["max_mag"]:
            a["max_mag"] = q["mag"]
        a["quakes"].append({"at": q["at"], "mag": q["mag"], "area": q["area"]})

    return activity


def load_baseline(days: int = 30) -> dict[str, dict]:
    """
    過去ログからベースライン（地域別の日平均活動量）を計算。
    ログが少ない場合は日本の経験的平均値にフォールバック。
    """
    # 経験的フォールバック（全国平均を地域別に按分した概算）
    fallback = {
        "北海道":   {"daily_avg": 3.0},
        "東北":     {"daily_avg": 5.0},
        "関東":     {"daily_avg": 4.0},
        "東海":     {"daily_avg": 3.5},
        "北陸":     {"daily_avg": 2.0},
        "近畿":     {"daily_avg": 2.0},
        "中国四国": {"daily_avg": 1.5},
        "九州":     {"daily_avg": 3.0},
        "沖縄":     {"daily_avg": 2.5},
    }

    path = Path(LEDGER_PATH)
    if not path.exists():
        return fallback

    cutoff_date = datetime.date.today() - datetime.timedelta(days=days)
    region_counts: dict[str, list[int]] = {r: [] for r in REGIONS}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        try:
            rec_date = datetime.date.fromisoformat(rec.get("date", ""))
        except Exception:
            continue
        if rec_date < cutoff_date:
            continue
        for region, data in rec.get("activity_48h", {}).items():
            if region in region_counts:
                # 48h → 1日換算
                region_counts[region].append(data.get("count_all", 0) / 2)

    result = {}
    for region in REGIONS:
        counts = region_counts[region]
        if len(counts) >= 7:
            result[region] = {"daily_avg": round(statistics.mean(counts), 2)}
        else:
            result[region] = fallback.get(region, {"daily_avg": 2.0})

    return result


def compute_anomaly(activity: dict[str, dict], baseline: dict[str, dict]) -> dict[str, dict]:
    """活動量のベースラインとの比（anomaly_ratio）を計算。"""
    result = {}
    for region, act in activity.items():
        base_daily = baseline.get(region, {}).get("daily_avg", 2.0)
        base_48h = base_daily * 2
        ratio = act["count_all"] / base_48h if base_48h > 0 else 1.0
        result[region] = {
            **act,
            "base_48h": round(base_48h, 2),
            "anomaly_ratio": round(ratio, 2),
            "anomaly_level": (
                "HIGH"   if ratio >= 3.0 else
                "MEDIUM" if ratio >= 1.5 else
                "NORMAL" if ratio >= 0.5 else
                "QUIET"
            ),
        }
    return result


# ── LA予測 ────────────────────────────────────────────────────

def _build_prompt(anomaly: dict[str, dict],
                  geo: dict | None = None,
                  imf: dict | None = None,
                  xray: dict | None = None) -> str:
    lines = ["【地域別地震活動（過去48時間）】\n"]
    for region, data in anomaly.items():
        level = data["anomaly_level"]
        lines.append(
            f"■ {region}: M2以上={data['count_all']}件 (通常の{data['anomaly_ratio']}倍, {level})"
            + (f" / M3以上={data['count_m3']}件" if data["count_m3"] > 0 else "")
            + (f" / M4以上={data['count_m4']}件" if data["count_m4"] > 0 else "")
            + (f" / 最大M{data['max_mag']}" if data["max_mag"] > 0 else "")
        )

    # 電磁気・宇宙天気データ
    em_lines = []
    if geo and geo.get("ok"):
        em_lines.append(
            f"  地磁気Kp指数: 3h平均={geo['kp_avg_3h']} / 最大={geo['kp_max_3h']} "
            f"[{geo['level']}]"
        )
    if imf and imf.get("ok"):
        em_lines.append(
            f"  IMF Bz(太陽風磁場): 1h平均={imf['bz_gsm_avg_1h']}nT / "
            f"最小={imf['bz_gsm_min_1h']}nT / Bt={imf['bt_avg_1h']}nT "
            f"[{imf['level']}]"
        )
    if xray and xray.get("ok"):
        em_lines.append(
            f"  太陽X線フラックス: 最大={xray['flux_max_1h']:.2e} "
            f"(フレアクラス:{xray['flare_class']}) [{xray['level']}]"
        )
    if em_lines:
        lines.append("\n【電磁気・宇宙天気（参考情報）】")
        lines.extend(em_lines)
        lines.append("  ※ 地磁気嵐・太陽フレアは電離層・地殻への影響が一部研究で示唆されている")

    lines.append("\n上記の活動状況をもとに、今後24時間以内に各地域でM5以上の地震が起きる確率を推定してください。")
    return "\n".join(lines)


def _ask_member(member: dict, prompt: str) -> dict:
    kote  = member["kote"]
    model = member["model"]
    payload = {
        "model": model,
        "prompt": f"{_SYSTEM_PRECURSOR}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 300},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        text = r.json().get("response", "")
        probs = _parse_probs(text)
        return {"kote": kote, "model": model, "ok": bool(probs), "probs": probs, "raw": text}
    except Exception as e:
        return {"kote": kote, "model": model, "ok": False, "probs": {}, "error": str(e)}


def _parse_probs(text: str) -> dict[str, float]:
    """
    "東海: 25%\n関東: 10%" → {"東海": 25.0, "関東": 10.0}
    """
    probs = {}
    for region in REGIONS:
        # "東海: 25%" or "東海：25%" or "東海 25%"
        m = re.search(rf"{region}[：:\s]+(\d+(?:\.\d+)?)\s*%", text)
        if m:
            probs[region] = float(m.group(1))
    return probs


def _aggregate_probs(member_results: list[dict]) -> dict[str, float | None]:
    region_vals: dict[str, list[float]] = {r: [] for r in REGIONS}
    for m in member_results:
        for region, prob in m.get("probs", {}).items():
            if region in region_vals:
                region_vals[region].append(prob)
    return {
        region: round(statistics.mean(vals), 1) if vals else None
        for region, vals in region_vals.items()
    }


# ── メインエントリ ─────────────────────────────────────────────

def run_precursor_monitor(members: list[dict] | None = None) -> dict:
    """
    前震モニタリングを実行して台帳に記録する。
    """
    if members is None:
        members = DEFAULT_MEMBERS

    now = datetime.datetime.now(datetime.timezone.utc)
    today = datetime.date.today().isoformat()

    print("[precursor] 気象庁APIから地震データ取得中...", flush=True)
    quakes = fetch_recent_quakes(500)
    if not quakes:
        return {"ok": False, "error": "地震データ取得失敗"}

    print(f"[precursor] {len(quakes)}件取得。活動集計中...", flush=True)
    activity = aggregate_activity(quakes, hours_back=48)
    baseline = load_baseline(days=30)
    anomaly  = compute_anomaly(activity, baseline)

    # 電磁気・宇宙天気データ取得
    print("[precursor] 電磁気・宇宙天気データ取得中...", flush=True)
    geo_data  = fetch_geomagnetic()
    imf_data  = fetch_imf()
    xray_data = fetch_xray()
    if geo_data.get("ok"):
        print(f"[precursor] 地磁気Kp: avg={geo_data['kp_avg_3h']} max={geo_data['kp_max_3h']} [{geo_data['level']}]", flush=True)
    if imf_data.get("ok"):
        print(f"[precursor] IMF Bz: avg={imf_data['bz_gsm_avg_1h']}nT min={imf_data['bz_gsm_min_1h']}nT [{imf_data['level']}]", flush=True)
    if xray_data.get("ok"):
        print(f"[precursor] X線フラックス: {xray_data['flare_class']}クラス [{xray_data['level']}]", flush=True)

    # HIGH/MEDIUM の地域をログ表示
    alert_regions = [r for r, d in anomaly.items() if d["anomaly_level"] in ("HIGH", "MEDIUM")]
    if alert_regions:
        print(f"[precursor] 活動活発な地域: {', '.join(alert_regions)}", flush=True)
    else:
        print("[precursor] 全地域で活動は平常レベル", flush=True)

    prompt = _build_prompt(anomaly, geo=geo_data, imf=imf_data, xray=xray_data)
    print("[precursor] LA予測開始...", flush=True)

    _heavy = [m for m in members if m["model"] in {"deepseek-r1:32b", "deepseek-r1:70b"}]
    _light = [m for m in members if m["model"] not in {"deepseek-r1:32b", "deepseek-r1:70b"}]

    member_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs_light = {ex.submit(_ask_member, m, prompt): m for m in _light}
        futs_heavy = {ex.submit(_ask_member, m, prompt): m for m in _heavy}
        for fut in concurrent.futures.as_completed({**futs_light, **futs_heavy}):
            member_results.append(fut.result())

    aggregated = _aggregate_probs(member_results)

    # 活動量サマリ（quakesリストは省いてコンパクトに）
    activity_summary = {
        region: {k: v for k, v in data.items() if k != "quakes"}
        for region, data in anomaly.items()
    }

    entry = {
        "id":           f"p_{now.strftime('%Y%m%d%H%M%S')}",
        "date":         today,
        "at":           now.isoformat(timespec="seconds"),
        "n_quakes_fetched": len(quakes),
        "activity_48h": activity_summary,
        "alert_regions": alert_regions,
        "geomagnetic":  geo_data,
        "imf":          imf_data,
        "xray":         xray_data,
        "member_results": member_results,
        "aggregated_probs": aggregated,
        "resolved": None,
    }

    path = Path(LEDGER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[precursor] 台帳に記録: {path}", flush=True)

    return {"ok": True, "entry": entry, "alert_regions": alert_regions, "aggregated": aggregated}


def format_summary(result: dict) -> str:
    """run_precursor_monitor の結果を人間が読める形に整形。"""
    if not result.get("ok"):
        return f"[precursor] エラー: {result.get('error')}"

    entry = result["entry"]
    lines = [f"=== 前震モニタリング {entry['date']} ==="]

    alert = result["alert_regions"]
    if alert:
        lines.append(f"⚠️  活動活発: {', '.join(alert)}")
    else:
        lines.append("✅ 全地域で活動は平常レベル")

    # 電磁気情報
    geo  = entry.get("geomagnetic", {})
    imf  = entry.get("imf", {})
    xray = entry.get("xray", {})
    em_parts = []
    if geo.get("ok"):
        em_parts.append(f"地磁気Kp={geo['kp_max_3h']}({geo['level']})")
    if imf.get("ok"):
        em_parts.append(f"IMF Bz={imf['bz_gsm_min_1h']}nT({imf['level']})")
    if xray.get("ok"):
        em_parts.append(f"太陽X線={xray['flare_class']}クラス({xray['level']})")
    if em_parts:
        lines.append("🌐 " + " / ".join(em_parts))

    lines.append("")
    lines.append("【今後24h M5以上の確率（LA集計）】")
    for region, prob in result["aggregated"].items():
        if prob is None:
            continue
        bar = "🔴" if prob >= 30 else "🟡" if prob >= 15 else "⬜"
        lines.append(f"  {bar} {region}: {prob}%")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_precursor_monitor()
    print(format_summary(result))
