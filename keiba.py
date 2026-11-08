"""
keiba.py — JRA競馬予想モジュール（第1段：材料あり独立見積もり）
=================================================================
工程:
  ① jiro8.sakura.ne.jp からレースデータ＋スピード指数を取得
  ② 材料を格付け・突き合わせ
  ③ 村人が各馬の勝率%を独立に見積もる
  ④ keiba_ledger.jsonl に記録

台帳は yosouya の ledger.jsonl とは別ファイル。
"""
from __future__ import annotations

import os
import re
import json
import time
import datetime
import statistics
import concurrent.futures

import requests
from bs4 import BeautifulSoup

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

LEDGER_PATH = os.environ.get(
    "KEIBA_LEDGER",
    r"C:\Users\USER\workspace\yosouya\keiba_ledger.jsonl"
)

# jiro8 の場所コード
PLACE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

DEFAULT_MEMBERS = [
    {"kote": "[競]巌",   "model": "deepseek-r1:32b"},
    {"kote": "[競]理明", "model": "gemma3:4b"},
    {"kote": "[競]否問", "model": "gemma3:4b"},
    {"kote": "[競]夕凪", "model": "gemma3:4b"},
    {"kote": "[競]神楽", "model": "gemma3:4b"},
    {"kote": "[競]銀風", "model": "gemma3:4b"},
]

_HEAVY_MODELS = {"deepseek-r1:32b", "deepseek-r1:70b"}

_SYSTEM_KEIBA = """\
あなたは競馬の予想師です。レースデータを分析し、以下の形式のみで回答してください。
余計な説明・解説は不要です。数字と所定フォーマットだけ出力してください。

【出力形式（必須）】
馬番1: X%
馬番2: X%
（全馬分、合計100%）
1着: 馬番N
2着: 馬番N
3着: 馬番N

スピード指数が高いほど過去の実績が高い。\
"""


# ── jiro8 スクレイピング ─────────────────────────────────────────

def _remove_spaces(text: str) -> str:
    """縦書きの各文字間のスペースを除去し、ｌ→ーに変換"""
    return text.replace(' ', '').replace('\u3000', '').replace('ｌ', 'ー').strip()


def fetch_race_data(place: str, kaisai: str, race: str) -> dict:
    """
    jiro8.sakura.ne.jp からレースデータを取得してパースする。

    place  : 場所コード "05"=東京, "06"=中山, etc.（または "東京" 等の日本語名）
    kaisai : 開催日 "20261108" (YYYYMMDD)
    race   : レース番号 "11" (01〜12)

    戻り値:
      {
        "ok": bool,
        "race_name": str,
        "date": str,
        "venue": str,
        "distance": str,
        "surface": str,
        "condition": str,
        "horses": [
          {
            "umaban": int,
            "wakuban": int,
            "name": str,
            "sex_age": str,
            "weight": float,
            "jockey": str,
            "trainer": str,
            "speed_index": float | None,
            "pace_index": float | None,
            "agari_index": float | None,
            "odds": float | None,
            "finish": str | None,     # レース後のみ
          }
        ],
        "source_url": str,
      }
    """
    # 日本語の場所名→コードに変換
    if place in PLACE_CODES:
        place = PLACE_CODES[place]

    url = f"https://jiro8.sakura.ne.jp/index.php?place={place}&kaisai={kaisai}&race={race}"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "cp932"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.find_all("tr")

    if not rows:
        return {"ok": False, "error": "テーブルが見つかりません"}

    # ── レース情報（見出し行を探す） ─────────────────────────────
    race_name = ""
    date_str = ""
    venue = ""
    distance = ""
    surface = ""
    condition = ""

    # レース名は <b><font>天皇賞(秋)（GI）</font></b> パターン
    for font_tag in soup.find_all("font"):
        t = font_tag.get_text(strip=True)
        if re.search(r"(賞|杯|ステークス|カップ|[GＧ][IＩ])", t) and len(t) < 60:
            race_name = t
            break

    for row in rows:
        text = row.get_text(" ", strip=True)
        if "年" in text and "月" in text and ("芝" in text or "ダ" in text) and re.search(r"\d{4}年\d+月\d+日", text):
            # 開催場（東京、中山 etc）: "4回東京11日" のパターン
            m_venue = re.search(r"\d+回([^\d\s]+?)\d+日", text)
            if m_venue:
                venue = m_venue.group(1)
            # 距離・馬場
            m = re.search(r"(芝|ダ)[・左右]*\s*(\d+m)", text)
            if m:
                surface = m.group(1)
                distance = m.group(2)
            m = re.search(r"(\d{4})年(\d+)月(\d+)日", text)
            if m:
                date_str = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            # 馬場状態
            for cond in ["良", "稍重", "重", "不良"]:
                if cond in text:
                    condition = cond
                    break
            break

    # ── 馬番行・馬名行・各データ行を探す ─────────────────────────
    umaban_row = None
    wakuban_row = None
    name_row = None
    sex_age_row = None
    weight_row = None
    jockey_row = None
    trainer_row = None
    finish_row = None
    odds_row = None
    speed_row = None
    pace_row = None
    agari_row = None
    senkou_row = None

    for i, row in enumerate(rows):
        tds = row.find_all("td")
        if not tds:
            continue
        last = tds[-1].get_text(strip=True)
        # 各行をラベルで識別
        if last == "馬番":
            umaban_row = tds
        elif last == "枠番":
            wakuban_row = tds
        elif last == "性齢":
            sex_age_row = tds
        elif last == "斤量":
            weight_row = tds
        elif last == "騎手":
            jockey_row = tds
        elif last == "調教師":
            trainer_row = tds
        elif last == "単勝オッズ":
            odds_row = tds
        elif last == "先行指数":
            senkou_row = tds
        elif last == "ペース指数":
            pace_row = tds
        elif last == "上がり指数":
            agari_row = tds
        elif last == "スピード指数":
            speed_row = tds
        elif last == "" and len(tds) > 5:
            # 着順行（最後が空ラベルで "着" が含まれる）
            first_text = tds[0].get_text(strip=True)
            if "着" in first_text and finish_row is None:
                finish_row = tds
        # 馬名行：最後のtdが空で td数が多く、2番目のtdの中に縦書きキャラが含まれる
        elif last == "馬名":
            name_row = tds
        elif last == "":
            # スピード指数行の判定：数値が17個並んでいて浮動小数点
            vals = [td.get_text(strip=True) for td in tds[:-1]]
            try:
                nums = [float(v) for v in vals if v]
                if len(nums) >= 15 and all(50 < n < 200 for n in nums) and speed_row is None:
                    speed_row = tds
            except ValueError:
                pass

    if umaban_row is None:
        return {"ok": False, "error": "馬番行が見つかりません"}

    # ── 馬番・枠番を配列化（最後のラベルセルを除く） ─────────────
    def row_to_values(tds, skip_last=True):
        end = len(tds) - 1 if skip_last else len(tds)
        return [td.get_text(strip=True) for td in tds[:end]]

    umabanlist = row_to_values(umaban_row)
    n_horses = len(umabanlist)

    wakubanlist = row_to_values(wakuban_row) if wakuban_row else [""] * n_horses
    sex_age_list = row_to_values(sex_age_row) if sex_age_row else [""] * n_horses
    weight_list = row_to_values(weight_row) if weight_row else [""] * n_horses
    jockey_list = row_to_values(jockey_row) if jockey_row else [""] * n_horses
    trainer_list = row_to_values(trainer_row) if trainer_row else [""] * n_horses
    finish_list = row_to_values(finish_row) if finish_row else [""] * n_horses
    pace_list = row_to_values(pace_row) if pace_row else []
    agari_list = row_to_values(agari_row) if agari_row else []
    speed_list = row_to_values(speed_row) if speed_row else []

    # オッズ行はカッコ内の人気順位を除いて数値部分だけ取る
    odds_list: list[str] = []
    if odds_row:
        for td in list(odds_row)[:-1]:
            txt = td.get_text(strip=True)
            m = re.match(r"([\d.]+)", txt)
            odds_list.append(m.group(1) if m else "")

    # ── 馬名のパース ──────────────────────────────────────────────
    # 各馬の td グループは5セル（母名 / 馬名 / 父名 / 毛色 / (combined)）
    # combined が index 0, 母名 1, 馬名 2, 父名 3, 毛色 4 の順
    name_list: list[str] = []
    if name_row:
        # name_row はラベル"馬名"セルを最後に持つ
        # 馬名セルは各グループの index 2（0-indexed）
        data_tds = list(name_row)[:-1]  # ラベルを除く
        # グループサイズを推定
        group_size = len(data_tds) // n_horses if n_horses else 5
        for g in range(n_horses):
            idx = g * group_size + (2 if group_size >= 3 else 0)
            if idx < len(data_tds):
                raw = data_tds[idx].get_text(" ", strip=True)
                name_list.append(_remove_spaces(raw))
            else:
                name_list.append("")
    else:
        name_list = [""] * n_horses

    # ── 馬ごとにまとめる ─────────────────────────────────────────
    def safe_float(lst, i):
        try:
            return float(lst[i]) if i < len(lst) and lst[i] else None
        except (ValueError, TypeError):
            return None

    horses = []
    for i in range(n_horses):
        horses.append({
            "umaban":      int(umabanlist[i]) if umabanlist[i].isdigit() else i + 1,
            "wakuban":     int(wakubanlist[i]) if i < len(wakubanlist) and wakubanlist[i].isdigit() else 0,
            "name":        name_list[i] if i < len(name_list) else "",
            "sex_age":     sex_age_list[i] if i < len(sex_age_list) else "",
            "weight":      safe_float(weight_list, i),
            "jockey":      jockey_list[i] if i < len(jockey_list) else "",
            "trainer":     trainer_list[i] if i < len(trainer_list) else "",
            "speed_index": safe_float(speed_list, i),
            "pace_index":  safe_float(pace_list, i),
            "agari_index": safe_float(agari_list, i),
            "odds":        safe_float(odds_list, i),
            "finish":      finish_list[i] if i < len(finish_list) else "",
        })

    # jiro8 は右→左（馬番17→1）で並んでいるので逆順にする
    horses.reverse()

    return {
        "ok":        True,
        "race_name": race_name,
        "date":      date_str,
        "venue":     venue,
        "distance":  distance,
        "surface":   "芝" if surface == "芝" else "ダート",
        "condition": condition,
        "horses":    horses,
        "source_url": url,
        "place":     place,
        "kaisai":    kaisai,
        "race":      race,
    }


# ── 追加ソース取得 ──────────────────────────────────────────────

def _fetch_netkeiba_shutuba(race_id: str) -> dict:
    """
    netkeiba.com 出馬表ページから追加情報（馬体重・調教タイム）を取得。
    race_id: "2026050511" 形式 (YYYYPPDD + RR, 10桁)

    失敗時は {"ok": False, "error": ...} を返しスキップ。
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://race.netkeiba.com/"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    horses_extra: list[dict] = []
    try:
        for tr in soup.select("table.Shutuba_Table tbody tr, .RaceTableWrap table tr"):
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue
            name_td = tr.select_one(".HorseName a, .Horse_Name a")
            weight_td = tr.select_one(".Weight, .HorseWeight")
            if not name_td:
                continue
            name = name_td.get_text(strip=True)
            weight_text = weight_td.get_text(strip=True) if weight_td else ""
            horses_extra.append({"name": name, "body_weight_text": weight_text})
    except Exception as e:
        return {"ok": False, "error": f"パース失敗: {e}"}

    if not horses_extra:
        return {"ok": False, "error": "出馬表テーブルが見つかりません"}

    return {"ok": True, "source": "netkeiba.com", "url": url, "horses_extra": horses_extra}


def _fetch_jra_today(place_name: str, race_num: str) -> dict:
    """
    JRA公式 当日レースページから馬場状態・確定オッズを取得（スクレイピング）。
    place_name: "東京" 等
    失敗時は {"ok": False, "error": ...} を返しスキップ。
    """
    url = "https://www.jra.go.jp/keiba/today/"
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    track_info: dict = {}
    try:
        # 馬場状態テーブルを探す
        for td in soup.find_all(["td", "th", "span"]):
            txt = td.get_text(strip=True)
            if place_name in txt and ("良" in txt or "稍重" in txt or "重" in txt or "不良" in txt):
                track_info["condition_raw"] = txt[:50]
                break
    except Exception:
        pass

    return {
        "ok": True,
        "source": "JRA公式",
        "url": url,
        "track_info": track_info,
    }


def _fetch_yahoo_keiba_marks(place_name: str, race_num: str, kaisai: str) -> dict:
    """
    Yahoo!競馬から専門家の印（◎○▲）を取得。
    失敗時は {"ok": False, "error": ...} を返しスキップ。
    """
    # Yahoo!競馬 レース一覧ページのURL（要スクレイピング）
    url = f"https://keiba.yahoo.co.jp/race/list/{kaisai[:8]}/"
    try:
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Accept-Language": "ja,en;q=0.9"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    marks: list[dict] = []
    try:
        # 印テーブルを探す（◎○▲△×）
        for td in soup.find_all(["td", "span"], class_=re.compile(r"mark|印|yoso|honmei")):
            mark_text = td.get_text(strip=True)
            if mark_text in ("◎", "○", "▲", "△", "×"):
                horse_td = td.find_next_sibling("td") or td.parent.find_next("td")
                if horse_td:
                    marks.append({"mark": mark_text, "horse": horse_td.get_text(strip=True)[:10]})
    except Exception:
        pass

    return {
        "ok": True,
        "source": "Yahoo!競馬",
        "url": url,
        "expert_marks": marks[:20],
    }


def _fetch_toriyako_picks(race_name: str) -> dict:
    """
    とりやこ競馬予想 (toriayako.com) から本命馬・穴馬の予想を取得。
    race_name: "天皇賞" 等の部分一致
    失敗時: {"ok": False, "error": str}
    成功時: {"ok": True, "source": "とりやこ", "picks": [{"mark": str, "horse": str}]}
    """
    url = "https://toriayako.com/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    picks: list[dict] = []
    try:
        # レース名を含む記事・セクションを探す
        for el in soup.find_all(["article", "section", "div", "h2", "h3"]):
            txt = el.get_text(strip=True)
            if race_name[:3] in txt and len(txt) < 500:
                # 本命・対抗・単穴などのマークを探す
                for mark_name, mark_sym in [("本命", "◎"), ("対抗", "○"), ("単穴", "▲"), ("穴", "△")]:
                    m = re.search(rf"{mark_name}[：:\s]*([^\s◎○▲△×\n]{{2,10}})", txt)
                    if m:
                        picks.append({"mark": mark_sym, "horse": m.group(1).strip()[:10]})
                if picks:
                    break
    except Exception as e:
        return {"ok": False, "error": f"パース失敗: {e}"}

    if not picks:
        return {"ok": False, "error": f"'{race_name}' の予想が見つかりません"}
    return {"ok": True, "source": "とりやこ", "url": url, "picks": picks}


def _fetch_umasan_buro_picks(race_name: str) -> dict:
    """
    馬三郎 (umasaburo.com) からレース予想を取得。
    失敗時: {"ok": False, "error": str}
    成功時: {"ok": True, "source": "馬三郎", "picks": [...], "comment": str}
    """
    url = "https://www.umasaburo.com/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    picks: list[dict] = []
    comment = ""
    try:
        for el in soup.find_all(["article", "section", "div"]):
            txt = el.get_text(strip=True)
            if race_name[:3] in txt and 30 < len(txt) < 600:
                comment = txt[:200]
                for mark_name, mark_sym in [("◎", "◎"), ("○", "○"), ("▲", "▲"), ("△", "△")]:
                    for m in re.finditer(rf"{re.escape(mark_sym)}\s*([^\s◎○▲△×\n]{{2,10}})", txt):
                        picks.append({"mark": mark_sym, "horse": m.group(1).strip()[:10]})
                if picks:
                    break
    except Exception as e:
        return {"ok": False, "error": f"パース失敗: {e}"}

    if not picks:
        return {"ok": False, "error": f"'{race_name}' の予想が見つかりません"}
    return {"ok": True, "source": "馬三郎", "url": url, "picks": picks, "comment": comment[:200]}


def _fetch_personal_site_picks(race_name: str) -> list[dict]:
    """
    個人競馬サイト群から予想印をまとめて取得。
    失敗したサイトはスキップ。戻り値は各サイトの結果リスト。
    """
    results = []
    for fetcher in [_fetch_toriyako_picks, _fetch_umasan_buro_picks]:
        try:
            res = fetcher(race_name)
            results.append(res)
        except Exception as e:
            results.append({"ok": False, "source": fetcher.__name__, "error": str(e)})
    return results


def _fetch_victory_road_jockey_winrates(place_code: str) -> dict:
    """
    Victory Road (ittai.net) から競馬場別騎手勝率テーブルを取得。
    place_code: "05" 等の2桁コード → "5" に変換してURLに使用。
    戻り値: {"ok": bool, "jockeys": {"騎手名": {"win": float, "rentai": float, "fukusho": float}}}
    """
    # Victory Road の場所コード → 「競馬場コード」（東京=5, 中山=6 等, 先頭0なし）
    place_num = str(int(place_code))
    url = f"https://ittai.net/RK/data_keibajyo.php?keibajyo={place_num}"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = r.apparent_encoding or "utf-8"
    except Exception as e:
        return {"ok": False, "error": str(e)}

    soup = BeautifulSoup(r.text, "html.parser")
    jockeys: dict[str, dict] = {}
    try:
        for tr in soup.select("table tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            name = tds[0].get_text(strip=True)
            if not name or name in ("騎手名", "氏名"):
                continue
            def to_pct(td):
                try:
                    return float(td.get_text(strip=True).replace("%", ""))
                except ValueError:
                    return None
            jockeys[name] = {
                "win":     to_pct(tds[1]),
                "rentai":  to_pct(tds[2]) if len(tds) > 2 else None,
                "fukusho": to_pct(tds[3]) if len(tds) > 3 else None,
            }
    except Exception as e:
        return {"ok": False, "error": f"パース失敗: {e}"}

    if not jockeys:
        return {"ok": False, "error": "データなし"}
    return {"ok": True, "source": "Victory Road", "url": url, "jockeys": jockeys}


def _fetch_cani_winrates() -> dict:
    """
    Cani競馬データ研究室 (cani.fool.jp) から騎手・調教師勝率を取得。
    戻り値: {
      "ok": bool,
      "jockeys": {"騎手名": {"win": float}},
      "trainers": {"調教師名": {"win": float}},
    }
    """
    jockeys: dict[str, dict] = {}
    trainers: dict[str, dict] = {}

    for category, out_dict in [("jockey", jockeys), ("trainer", trainers)]:
        url = f"http://cani.fool.jp/{category}.html"
        try:
            r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            r.encoding = r.apparent_encoding or "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            for tr in soup.select("table tr"):
                tds = tr.find_all("td")
                if len(tds) < 2:
                    continue
                name = tds[0].get_text(strip=True)
                if not name or name in ("騎手", "調教師", "名前"):
                    continue
                try:
                    win_pct = float(tds[1].get_text(strip=True).replace("%", ""))
                    out_dict[name] = {"win": win_pct}
                except ValueError:
                    pass
        except Exception:
            pass

    if not jockeys and not trainers:
        return {"ok": False, "error": "Caniからデータ取得失敗"}
    return {"ok": True, "source": "Cani競馬データ研究室", "jockeys": jockeys, "trainers": trainers}


def _kaisai_to_netkeiba_race_id(place: str, kaisai: str, race: str) -> str:
    """
    jiro8形式のパラメータをnetkeiba race_id形式に変換。
    place="05"(東京), kaisai="20261108", race="11" → "202605051108" (12桁)
    netkeiba形式: YYYY + PP + DD + RR
    YYYYは西暦、PP=場所コード、DDは開催日の日付部分
    """
    yyyy = kaisai[:4]
    pp   = place.zfill(2)
    mmdd = kaisai[4:8]   # MMDD
    rr   = race.zfill(2)
    return f"{yyyy}{pp}{mmdd}{rr}"


def fetch_race_data_multi(place: str, kaisai: str, race: str) -> dict:
    """
    複数ソースから競馬データを取得してマージする。

    1. jiro8（メイン: スピード指数）  → 必須、失敗時はエラー返却
    2. netkeiba.com（馬体重・調教）   → 任意、失敗時スキップ
    3. JRA公式（馬場状態）            → 任意、失敗時スキップ
    4. Yahoo!競馬（専門家の印）       → 任意、失敗時スキップ

    戻り値: fetch_race_data() の dict に "extra_sources" キーを追加したもの
    """
    # jiro8 は日本語名→コードに変換済み
    if place in PLACE_CODES:
        place = PLACE_CODES[place]

    # 1) jiro8（必須）
    base = fetch_race_data(place, kaisai, race)
    if not base.get("ok"):
        return base

    extra_sources: list[dict] = []

    # 2) netkeiba.com
    try:
        netkeiba_id = _kaisai_to_netkeiba_race_id(place, kaisai, race)
        nk = _fetch_netkeiba_shutuba(netkeiba_id)
        if nk.get("ok"):
            # 馬体重テキストをマージ
            nk_map = {h["name"]: h["body_weight_text"] for h in nk.get("horses_extra", [])}
            for h in base["horses"]:
                if h["name"] in nk_map:
                    h["body_weight_text"] = nk_map[h["name"]]
            extra_sources.append({"source": "netkeiba.com", "ok": True})
        else:
            print(f"[keiba] netkeiba スキップ: {nk.get('error', '')}")
            extra_sources.append({"source": "netkeiba.com", "ok": False,
                                   "error": nk.get("error", "")})
    except Exception as e:
        print(f"[keiba] netkeiba 例外: {e}")
        extra_sources.append({"source": "netkeiba.com", "ok": False, "error": str(e)})

    # 3) JRA公式
    try:
        reverse_codes = {v: k for k, v in PLACE_CODES.items()}
        place_name = reverse_codes.get(place, place)
        jra = _fetch_jra_today(place_name, race)
        if jra.get("ok") and jra.get("track_info"):
            base["condition_jra"] = jra["track_info"].get("condition_raw", "")
        extra_sources.append({"source": "JRA公式", "ok": jra.get("ok", False),
                               "error": jra.get("error", "")})
    except Exception as e:
        print(f"[keiba] JRA公式 例外: {e}")
        extra_sources.append({"source": "JRA公式", "ok": False, "error": str(e)})

    # 4) Yahoo!競馬
    try:
        reverse_codes = {v: k for k, v in PLACE_CODES.items()}
        place_name = reverse_codes.get(place, place)
        yh = _fetch_yahoo_keiba_marks(place_name, race, kaisai)
        if yh.get("ok") and yh.get("expert_marks"):
            base["expert_marks"] = yh["expert_marks"]
        extra_sources.append({"source": "Yahoo!競馬", "ok": yh.get("ok", False),
                               "error": yh.get("error", "")})
    except Exception as e:
        print(f"[keiba] Yahoo!競馬 例外: {e}")
        extra_sources.append({"source": "Yahoo!競馬", "ok": False, "error": str(e)})

    # 5) 個人競馬サイト予想印
    try:
        race_name = base.get("race_name", "")
        if race_name:
            personal_results = _fetch_personal_site_picks(race_name)
            # 全サイトの印を統合
            personal_marks: dict[str, list[str]] = {}  # 馬名 → マークリスト
            for site_res in personal_results:
                if site_res.get("ok"):
                    src = site_res.get("source", "?")
                    for pick in site_res.get("picks", []):
                        horse = pick.get("horse", "")
                        mark  = pick.get("mark", "")
                        if horse:
                            personal_marks.setdefault(horse, []).append(f"{src}{mark}")
            if personal_marks:
                base["personal_marks"] = personal_marks
            ok_count = sum(1 for r in personal_results if r.get("ok"))
            print(f"[keiba] 個人サイト: {ok_count}/{len(personal_results)}サイト取得")
            extra_sources.extend([
                {"source": r.get("source", "個人サイト"), "ok": r.get("ok", False),
                 "error": r.get("error", "")} for r in personal_results
            ])
    except Exception as e:
        print(f"[keiba] 個人サイト 例外: {e}")

    # 6) Victory Road 騎手勝率（競馬場別）
    try:
        vr = _fetch_victory_road_jockey_winrates(place)
        if vr.get("ok"):
            base["jockey_winrates"] = vr["jockeys"]
            print(f"[keiba] Victory Road: 騎手{len(vr['jockeys'])}名の勝率取得")
        else:
            print(f"[keiba] Victory Road スキップ: {vr.get('error', '')}")
        extra_sources.append({"source": "Victory Road", "ok": vr.get("ok", False),
                               "error": vr.get("error", "")})
    except Exception as e:
        print(f"[keiba] Victory Road 例外: {e}")
        extra_sources.append({"source": "Victory Road", "ok": False, "error": str(e)})

    # 7) Cani 騎手・調教師勝率（全場通算）
    try:
        cani = _fetch_cani_winrates()
        if cani.get("ok"):
            base["cani_jockey_winrates"]  = cani.get("jockeys", {})
            base["cani_trainer_winrates"] = cani.get("trainers", {})
            print(f"[keiba] Cani: 騎手{len(cani.get('jockeys', {}))}名 / 調教師{len(cani.get('trainers', {}))}名")
        else:
            print(f"[keiba] Cani スキップ: {cani.get('error', '')}")
        extra_sources.append({"source": "Cani", "ok": cani.get("ok", False),
                               "error": cani.get("error", "")})
    except Exception as e:
        print(f"[keiba] Cani 例外: {e}")
        extra_sources.append({"source": "Cani", "ok": False, "error": str(e)})

    base["extra_sources"] = extra_sources
    return base


def _build_race_summary(race: dict) -> str:
    """レースデータを村人への材料テキストに変換（マルチソース対応）"""
    lines = [
        f"【レース】 {race.get('race_name', '不明')}",
        f"コース: {race.get('surface', '')} {race.get('distance', '')} / 馬場: {race.get('condition', '不明')}",
    ]
    if race.get("condition_jra"):
        lines.append(f"馬場（JRA公式）: {race['condition_jra']}")
    lines += [
        "",
        "【出馬表】",
        "馬番 | 馬名         | 騎手         | 斤量 | スピード指数 | 単勝オッズ",
        "─" * 70,
    ]
    expert_map: dict[str, str] = {}
    for m in race.get("expert_marks", []):
        expert_map[m.get("horse", "")] = m.get("mark", "")

    personal_marks = race.get("personal_marks", {})  # 個人サイト印
    jw_map  = race.get("jockey_winrates", {})    # Victory Road 競馬場別騎手勝率
    cjw_map = race.get("cani_jockey_winrates", {}) # Cani 全場通算騎手勝率
    ctw_map = race.get("cani_trainer_winrates", {}) # Cani 調教師勝率

    for h in race["horses"]:
        si = f"{h['speed_index']:.1f}" if h["speed_index"] is not None else "  ─ "
        odds = f"{h['odds']:.1f}" if h["odds"] is not None else " ─"
        bw = f" ({h['body_weight_text']})" if h.get("body_weight_text") else ""
        mark = expert_map.get(h["name"], " ")
        # 騎手勝率（Victory Road 優先、なければ Cani）
        jockey = h.get("jockey", "")
        jw = jw_map.get(jockey) or cjw_map.get(jockey)
        jw_str = f" J勝{jw['win']:.0f}%" if jw and jw.get("win") is not None else ""
        # 調教師勝率（Cani）
        trainer = h.get("trainer", "")
        tw = ctw_map.get(trainer)
        tw_str = f" T勝{tw['win']:.0f}%" if tw and tw.get("win") is not None else ""
        pm = personal_marks.get(h["name"], [])
        pm_str = f" [{','.join(pm)}]" if pm else ""
        lines.append(
            f"{h['umaban']:>2}番 {mark} {h['name']:<10} {h['jockey']:<10} "
            f"{h.get('weight',''):<4} {si:>8}  {odds:>7}{bw}{jw_str}{tw_str}{pm_str}"
        )

    if expert_map:
        lines += ["", "【専門家の印（Yahoo!競馬）】",
                  " ".join(f"{m}:{n}" for m, n in sorted(expert_map.items(),
                                                          key=lambda x: ("◎○▲△×".find(x[1])
                                                                         if x[1] in "◎○▲△×" else 99)))]
    return "\n".join(lines)


# ── 格付け・信頼性評価 ──────────────────────────────────────────

def _rate_sources(race_data: dict) -> dict:
    """
    情報源の信頼性を評価する。

    - jiro8 はスピード指数のみ（個人サイト）→ ★★☆
    - JRA公式タイム（レース結果）が含まれる場合 → ★★★
    - 複数ソース一致確認は今後の拡張予定

    戻り値: {
      "credibility": str,
      "notes": [str],
    }
    """
    notes = []
    cred = "★★☆ 個人サイト（jiro8）"

    # スピード指数の有無
    horses_with_si = [h for h in race_data["horses"] if h["speed_index"] is not None]
    if len(horses_with_si) == 0:
        notes.append("⚠ スピード指数が取得できませんでした（レース前または未掲載）")
        cred = "★☆☆ データ不完全"
    else:
        notes.append(f"スピード指数あり: {len(horses_with_si)}/{len(race_data['horses'])}頭分")

    # 着順が入っている = レース後のデータ（採点用）
    horses_with_finish = [h for h in race_data["horses"] if h.get("finish")]
    if horses_with_finish:
        notes.append("着順データあり（レース後）")
        cred = "★★★ JRA確定結果含む"

    return {"credibility": cred, "notes": notes}


# ── 村人への問い合わせ ──────────────────────────────────────────

def _ask_one_keiba(member: dict, race_summary: str, timeout: float = 90.0,
                   n_horses: int = 18) -> dict:
    """1モデルに競馬予想を依頼。{kote, picks: [{umaban, prob, reason}], ok} を返す。"""
    kote = member.get("kote", "?")
    model = member.get("model", "")

    # プレフィックス補完：モデルに「馬番1: 」の続きを埋めさせる
    prefix = "馬番1: "
    completion_prompt = (
        f"{_SYSTEM_KEIBA}\n\n"
        f"{race_summary}\n\n"
        f"{prefix}"
    )

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": completion_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500},
            },
            timeout=timeout,
        )
        resp = (r.json().get("response", "") or "").strip()
        # プレフィックスを先頭に付けてパース可能にする
        text = prefix + resp
    except Exception as e:
        return {"kote": kote, "model": model, "picks": [], "ok": False, "error": str(e)}

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    picks, top3 = _parse_picks(text)
    print(f"[keiba] {kote} raw(200)={text[:200].replace(chr(10),' ')}")
    return {
        "kote":  kote,
        "model": model,
        "picks": picks,
        "top3":  top3,
        "raw":   text[:800],
        "ok":    len(picks) > 0,
    }


def _parse_picks(text: str) -> tuple[list[dict], list[int]]:
    """
    "馬番N: NN%" の勝率と "1着: 馬番N" の上位3頭をパース。
    戻り値: (picks, top3)
      picks : [{"umaban": int, "prob": int}, ...]
      top3  : [1着馬番, 2着馬番, 3着馬番]  (欠けた分は0)
    """
    # 馬番N: XX%  /  馬番N 馬名: XX%  /  N番 馬名: XX%  など複数形式対応
    prob_pattern = re.compile(r"(?:馬番)?\s*(\d{1,2})\s*番?[^\d\n]{0,20}[:：]\s*(\d{1,2})\s*[%％]")
    picks = []
    for m in prob_pattern.finditer(text):
        picks.append({"umaban": int(m.group(1)), "prob": int(m.group(2))})

    # "1着: 馬番N" / "1着: N" 形式
    top3 = [0, 0, 0]
    for rank in range(1, 4):
        m = re.search(rf"{rank}着\s*[:：]\s*(?:馬番)?\s*(\d+)", text)
        if m:
            top3[rank - 1] = int(m.group(1))

    return picks, top3


def _aggregate_picks(all_results: list[dict], n_horses: int) -> tuple[list[dict], dict]:
    """
    各村人の予想を集計して馬ごとの平均勝率と上位3頭推奨を計算。
    戻り値: (aggregated, combo)
      aggregated: 馬ごとの平均勝率リスト（降順）
      combo: {
        "trio":      [馬番, 馬番, 馬番]  # 3連複（順不同、最多得票）
        "trifecta":  [1着馬番, 2着馬番, 3着馬番]  # 3連単（最多得票）
        "place_votes": {馬番: 票数}  # 3着以内に選ばれた回数
      }
    """
    # ── 勝率集計 ──────────────────────────────────────────────────
    probs_by_horse: dict[int, list[float]] = {i: [] for i in range(1, n_horses + 1)}
    for result in all_results:
        if not result.get("ok"):
            continue
        picks = result.get("picks", [])
        total = sum(p["prob"] for p in picks)
        if total == 0:
            continue
        for pick in picks:
            if 1 <= pick["umaban"] <= n_horses:
                probs_by_horse[pick["umaban"]].append(pick["prob"] / total * 100)

    aggregated = []
    for umaban in range(1, n_horses + 1):
        probs = probs_by_horse[umaban]
        avg_prob = round(statistics.mean(probs), 1) if probs else None
        std_prob = round(statistics.stdev(probs), 1) if len(probs) >= 2 else 0.0
        # 一致度: std=0→100%, std=50→0% のリニアスケール
        consensus = round(max(0.0, 100.0 - std_prob * 2), 1) if probs else None
        aggregated.append({
            "umaban":       umaban,
            "avg_prob":     avg_prob,
            "std_prob":     std_prob,
            "consensus_pct": consensus,
            "n_votes":      len(probs),
        })
    aggregated.sort(key=lambda x: x["avg_prob"] if x["avg_prob"] is not None else -1, reverse=True)

    # ── top3 集計 ─────────────────────────────────────────────────
    place_votes: dict[int, int] = {}      # 3着以内に選ばれた回数
    trifecta_votes: dict[tuple, int] = {} # 3連単（順序あり）
    trio_votes: dict[tuple, int] = {}     # 3連複（順序なし）

    for result in all_results:
        if not result.get("ok"):
            continue
        t3 = result.get("top3", [0, 0, 0])
        if any(v == 0 for v in t3):
            continue  # 3頭揃っていない回答は除外
        for v in t3:
            place_votes[v] = place_votes.get(v, 0) + 1
        trifecta_key = tuple(t3)
        trifecta_votes[trifecta_key] = trifecta_votes.get(trifecta_key, 0) + 1
        trio_key = tuple(sorted(t3))
        trio_votes[trio_key] = trio_votes.get(trio_key, 0) + 1

    best_trifecta = max(trifecta_votes, key=trifecta_votes.get) if trifecta_votes else None
    best_trio     = max(trio_votes,     key=trio_votes.get)     if trio_votes     else None

    combo = {
        "trio":        list(best_trio)      if best_trio      else [],
        "trifecta":    list(best_trifecta)  if best_trifecta  else [],
        "place_votes": place_votes,
    }
    return aggregated, combo


# ── 台帳 ───────────────────────────────────────────────────────

def _append_ledger(entry: dict) -> str:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return LEDGER_PATH


# ── メイン実行関数 ──────────────────────────────────────────────

def run(place: str, kaisai: str, race: str, members: list | None = None) -> dict:
    """
    競馬予想を1レース分実行する。

    place  : "東京" or "05"
    kaisai : "20261108" (YYYYMMDD)
    race   : "11" (R番号, 1〜12)
    """
    members = members or DEFAULT_MEMBERS
    race_num = race.zfill(2)

    # ── 1. データ取得 ──────────────────────────────────────────
    print(f"[keiba] データ取得: {place} {kaisai} {race_num}R")
    race_data = fetch_race_data(place, kaisai, race_num)
    if not race_data.get("ok"):
        return {"ok": False, "error": race_data.get("error", "データ取得失敗")}

    n_horses = len(race_data["horses"])
    print(f"[keiba] {n_horses}頭 取得: {race_data.get('race_name', '')[:30]}")

    # ── 2. 信頼性評価 ──────────────────────────────────────────
    source_rating = _rate_sources(race_data)
    print(f"[keiba] 信頼性: {source_rating['credibility']}")
    for note in source_rating["notes"]:
        print(f"[keiba]   {note}")

    # ── 3. 材料テキスト生成 ────────────────────────────────────
    race_summary = _build_race_summary(race_data)

    # ── 4. 独立見積もり（軽量先行・重量後出し） ───────────────
    light_members = [m for m in members if m.get("model") not in _HEAVY_MODELS]
    heavy_members = [m for m in members if m.get("model") in _HEAVY_MODELS]

    all_results: list[dict] = []

    if light_members:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(light_members)) as ex:
            futs = {ex.submit(_ask_one_keiba, m, race_summary, 90.0, n_horses): m for m in light_members}
            for fut in concurrent.futures.as_completed(futs, timeout=120):
                res = fut.result()
                all_results.append(res)
                ok_str = "OK" if res.get("ok") else "NG"
                print(f"[keiba] {res['kote']} {ok_str} picks={len(res.get('picks', []))}")

    for m in heavy_members:
        print(f"[keiba] 重量モデル: {m['kote']} ({m['model']})")
        res = _ask_one_keiba(m, race_summary, 600.0, n_horses)
        all_results.append(res)
        ok_str = "OK" if res.get("ok") else "NG"
        print(f"[keiba] {res['kote']} {ok_str} picks={len(res.get('picks', []))}")

    # ── 5. 集計 ───────────────────────────────────────────────
    aggregated, combo = _aggregate_picks(all_results, n_horses)

    # 馬名を aggregated に付ける
    name_map = {h["umaban"]: h["name"] for h in race_data["horses"]}
    odds_map = {h["umaban"]: h["odds"] for h in race_data["horses"]}
    si_map = {h["umaban"]: h["speed_index"] for h in race_data["horses"]}
    for a in aggregated:
        a["name"]        = name_map.get(a["umaban"], "")
        a["odds"]        = odds_map.get(a["umaban"])
        a["speed_index"] = si_map.get(a["umaban"])

    # ── 6. 台帳に記録 ─────────────────────────────────────────
    entry = {
        "id":           "k_" + time.strftime("%Y%m%d%H%M%S"),
        "at":           datetime.datetime.now().isoformat(timespec="seconds"),
        "place":        place,
        "kaisai":       kaisai,
        "race":         race_num,
        "race_name":    race_data.get("race_name", ""),
        "date":         race_data.get("date", ""),
        "surface":      race_data.get("surface", ""),
        "distance":     race_data.get("distance", ""),
        "condition":    race_data.get("condition", ""),
        "n_horses":     n_horses,
        "source_url":   race_data.get("source_url", ""),
        "credibility":  source_rating["credibility"],
        "horses":       race_data["horses"],
        "member_results": [
            {
                "kote":  r["kote"],
                "model": r["model"],
                "ok":    r.get("ok", False),
                "picks": r.get("picks", []),
                "top3":  r.get("top3", []),
            }
            for r in all_results
        ],
        "aggregated":   aggregated,
        "combo":        combo,
        "resolved":     None,   # レース後に着順を記録する
    }
    ledger_path = _append_ledger(entry)
    print(f"[keiba] 台帳に記録: {ledger_path}")

    # ── 7. トップ3を表示 ─────────────────────────────────────
    print("\n【村の予想 勝率ランキング トップ3】")
    for i, a in enumerate(aggregated[:3]):
        odds_str = f"(単勝 {a['odds']}倍)" if a.get("odds") else ""
        si_str = f"SI={a['speed_index']:.0f}" if a.get("speed_index") else ""
        print(f"  {i+1}位: {a['umaban']}番 {a['name']} 平均{a['avg_prob']}% {odds_str} {si_str}")

    if combo.get("trio"):
        trio_names = [name_map.get(n, str(n)) for n in combo["trio"]]
        print(f"\n【3連複推奨】{'-'.join(str(n) for n in sorted(combo['trio']))} "
              f"({' / '.join(trio_names)})")
    if combo.get("trifecta"):
        tri_names = [name_map.get(n, str(n)) for n in combo["trifecta"]]
        print(f"【3連単推奨】{'-'.join(str(n) for n in combo['trifecta'])} "
              f"({' → '.join(tri_names)})")

    return {
        "ok":          True,
        "id":          entry["id"],
        "race_name":   race_data.get("race_name", ""),
        "n_horses":    n_horses,
        "aggregated":  aggregated,
        "combo":       combo,
        "members":     all_results,
        "credibility": source_rating["credibility"],
        "ledger":      ledger_path,
    }


# ── 開催スケジュール取得 ───────────────────────────────────────

def fetch_today_schedule(date_str: str | None = None) -> list[dict]:
    """
    jiro8 トップページを解析して、当日開催している競馬場一覧を返す。

    戻り値: [{"place": "05", "place_name": "東京", "date": "20261108"}, ...]

    1) jiro8 トップページのリンクから当年の YYKKPPDDHR コードを抽出
    2) PP（場所コード）に対応する開催場名を返す
    3) 日付は date_str が指定されていれば優先、なければ今日
    """
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y%m%d")

    # jiro8 トップページから当日レースを検索
    year_suffix = date_str[2:4]  # "2026" → "26"
    active_places: set[str] = set()

    try:
        r = requests.get(
            "https://jiro8.sakura.ne.jp/",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.encoding = r.apparent_encoding or "cp932"
        soup = BeautifulSoup(r.text, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            m = re.search(r"code=(\d{10})", href)
            if not m:
                continue
            code = m.group(1)
            yy = code[0:2]
            pp = code[4:6]
            dd = code[6:8]
            # 今年・9999以外（9999は年別ショートカット）
            if yy == year_suffix and dd != "99":
                active_places.add(pp)
    except Exception as e:
        print(f"[keiba] jiro8トップページ取得失敗: {e} → 全開催場を探索します")

    # トップページから検出できなかった場合はプローブで確認
    if not active_places:
        print("[keiba] 全開催場をプローブで確認中...")
        reverse_codes = {v: k for k, v in PLACE_CODES.items()}
        for pname, pcode in PLACE_CODES.items():
            data = fetch_race_data(pcode, date_str, "11")
            if data.get("ok"):
                active_places.add(pcode)
                print(f"[keiba]   {pname}({pcode}) 開催確認")
            time.sleep(0.3)

    reverse_codes = {v: k for k, v in PLACE_CODES.items()}
    schedule = [
        {"place": p, "place_name": reverse_codes.get(p, p), "date": date_str}
        for p in sorted(active_places)
        if p in reverse_codes
    ]
    print(f"[keiba] 本日の開催: {[s['place_name'] for s in schedule]}")
    return schedule


# ── 全レース一括予想 ───────────────────────────────────────────

def run_all_today(
    date_str: str | None = None,
    members: list | None = None,
    races: list[str] | None = None,
) -> dict:
    """
    当日の全開催場 × 全レースを順番に予想して台帳に記録する。

    date_str : 対象日（YYYYMMDD）。省略時は今日。
    members  : 予想する村人リスト。省略時は DEFAULT_MEMBERS。
    races    : 対象レース番号リスト（例: ["11","12"]）。省略時は01〜12全部。

    戻り値: {
      "ok": bool,
      "date": str,
      "schedule": [...],
      "results": [{"place": ..., "race": ..., "ok": bool, ...}],
      "n_success": int,
      "n_skip": int,
    }
    """
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y%m%d")
    if races is None:
        races = [f"{r:02d}" for r in range(1, 13)]

    schedule = fetch_today_schedule(date_str)
    if not schedule:
        return {
            "ok": False,
            "date": date_str,
            "error": "開催情報が取得できませんでした",
            "schedule": [],
            "results": [],
        }

    all_results = []
    n_success = 0
    n_skip = 0

    for venue in schedule:
        place = venue["place"]
        pname = venue["place_name"]
        for race_num in races:
            print(f"[keiba] run_all: {pname}({place}) {race_num}R ...", flush=True)
            try:
                result = run(place, date_str, race_num, members=members)
                if result.get("ok"):
                    n_success += 1
                    all_results.append({
                        "place": place, "place_name": pname,
                        "race": race_num, "ok": True,
                        "race_name": result.get("race_name", ""),
                        "id": result.get("id", ""),
                    })
                else:
                    n_skip += 1
                    all_results.append({
                        "place": place, "place_name": pname,
                        "race": race_num, "ok": False,
                        "error": result.get("error", "データなし"),
                    })
            except Exception as e:
                n_skip += 1
                all_results.append({
                    "place": place, "place_name": pname,
                    "race": race_num, "ok": False, "error": str(e),
                })
            time.sleep(1.0)   # API レート制限

    print(f"[keiba] run_all 完了: 成功={n_success} スキップ={n_skip}")

    # レース終了後（19:00以降）に自動で答え合わせするスレッドを起動
    if n_success > 0:
        import threading
        def _delayed_resolve():
            # 最終レース終了まで待つ（19:00JST まで待機してから照合）
            import datetime as _dt
            jst_offset = _dt.timezone(_dt.timedelta(hours=9))
            now_jst = _dt.datetime.now(tz=jst_offset)
            target = now_jst.replace(hour=19, minute=0, second=0, microsecond=0)
            if now_jst >= target:
                # すでに19時過ぎなら即座に照合
                wait_sec = 0
            else:
                wait_sec = (target - now_jst).total_seconds()
            if wait_sec > 0:
                print(f"[keiba] 答え合わせを {int(wait_sec/60)} 分後（19:00JST）に自動実行します")
                time.sleep(wait_sec)
            print("[keiba] 自動答え合わせ開始")
            resolve_past(limit=200)
        threading.Thread(target=_delayed_resolve, daemon=True).start()

    return {
        "ok":       True,
        "date":     date_str,
        "schedule": schedule,
        "results":  all_results,
        "n_success": n_success,
        "n_skip":    n_skip,
    }


# ── 過去レース結果の自動照合 ──────────────────────────────────

def resolve_past(limit: int = 50) -> dict:
    """
    keiba_ledger.jsonl の resolved=null なエントリを読み込み、
    jiro8 から着順を再取得して resolved フィールドを埋める。

    resolved フィールドの形式:
      {
        "1": "馬名",
        "2": "馬名",
        "3": "馬名",
        "trio": [馬番, 馬番, 馬番],  # 3着以内の馬番
        "hit":  bool,               # 予想trio と実績trioが一致したら True
        "resolved_at": "ISO8601",
      }

    戻り値: {"ok": True, "n_resolved": int, "n_missing": int, "entries": [...]}
    """
    path = __import__("pathlib").Path(LEDGER_PATH)
    if not path.exists():
        return {"ok": False, "error": "台帳が見つかりません"}

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            entries.append({"_raw": line})

    targets = [e for e in entries if isinstance(e, dict) and e.get("resolved") is None
               and e.get("place") and e.get("kaisai") and e.get("race")]
    targets = targets[:limit]

    n_resolved = 0
    n_missing = 0
    resolved_summary = []

    for entry in targets:
        place  = entry["place"]
        kaisai = entry["kaisai"]
        race   = entry["race"]
        eid    = entry.get("id", "?")
        print(f"[keiba] resolve: {eid} {place} {kaisai} {race}R", flush=True)

        race_data = fetch_race_data(place, kaisai, race)
        time.sleep(0.5)

        if not race_data.get("ok"):
            n_missing += 1
            resolved_summary.append({"id": eid, "ok": False,
                                     "error": race_data.get("error", "取得失敗")})
            continue

        # 着順を持つ馬を抽出（1〜3着）
        name_map = {h["umaban"]: h["name"] for h in race_data["horses"]}
        finish_map: dict[str, str] = {}  # finish文字列 → 馬名
        for h in race_data["horses"]:
            f = str(h.get("finish", "")).strip()
            if f and re.match(r"^\d+$", f):
                finish_map[f] = h["name"]

        if not finish_map:
            # まだ着順が出ていない（レース前 or データ未更新）
            n_missing += 1
            resolved_summary.append({"id": eid, "ok": False, "error": "着順データなし"})
            continue

        top3_names: dict[str, str] = {}  # "1" → 馬名
        top3_umabans: list[int] = []
        for rank in range(1, 4):
            name = finish_map.get(str(rank), "")
            top3_names[str(rank)] = name
            # 馬番を逆引き
            for h in race_data["horses"]:
                if h["name"] == name:
                    top3_umabans.append(h["umaban"])
                    break

        # 予想 combo.trio と実績 top3 の一致チェック
        predicted_trio: list[int] = entry.get("combo", {}).get("trio", [])
        actual_trio_set  = set(top3_umabans)
        predicted_trio_set = set(predicted_trio)
        hit = bool(predicted_trio_set and actual_trio_set and
                   predicted_trio_set == actual_trio_set)

        resolved = {
            **top3_names,
            "trio": top3_umabans,
            "hit":  hit,
            "resolved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        entry["resolved"] = resolved
        n_resolved += 1
        resolved_summary.append({
            "id": eid, "ok": True,
            "top3": top3_names, "hit": hit,
            "predicted_trio": predicted_trio,
            "actual_trio": top3_umabans,
        })
        print(f"[keiba]   → 1着:{top3_names.get('1','')} hit={hit}")

    # 台帳を書き戻す（全行を更新して上書き）
    updated_lines = []
    for e in entries:
        if isinstance(e, dict) and "_raw" not in e:
            updated_lines.append(json.dumps(e, ensure_ascii=False))
        elif isinstance(e, dict) and "_raw" in e:
            updated_lines.append(e["_raw"])
    path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    print(f"[keiba] resolve_past 完了: 解決={n_resolved} 未確定={n_missing}")
    return {
        "ok":          True,
        "n_resolved":  n_resolved,
        "n_missing":   n_missing,
        "entries":     resolved_summary,
    }


def stats(days: int = 90) -> dict:
    """
    keiba_ledger.jsonl の resolved済みエントリから的中率・ROIを集計する。

    戻り値:
      {
        "ok": bool,
        "n_total": int,          # 総レース数（resolved済み）
        "n_hit": int,            # 3連複的中数
        "hit_rate_pct": float,   # 的中率(%)
        "by_place": {場所名: {"n": int, "hits": int, "rate": float}},
        "by_member": {kote: {"votes": int, "top1_hits": int, "rate": float}},
      }
    """
    path = __import__("pathlib").Path(LEDGER_PATH)
    if not path.exists():
        return {"ok": False, "error": "台帳が見つかりません"}

    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    # resolved済み（dictで"hit"キーがある）かつ対象期間内
    resolved = [
        e for e in entries
        if isinstance(e.get("resolved"), dict)
        and e.get("date", "") >= cutoff
    ]

    if not resolved:
        return {"ok": True, "n_total": 0, "n_hit": 0, "hit_rate_pct": 0.0,
                "by_place": {}, "by_member": {}}

    n_total = len(resolved)
    n_hit   = sum(1 for e in resolved if e["resolved"].get("hit"))

    # 開催場別
    place_map: dict[str, dict] = {}
    reverse_codes = {v: k for k, v in PLACE_CODES.items()}
    for e in resolved:
        p = reverse_codes.get(e.get("place", ""), e.get("place", "?"))
        if p not in place_map:
            place_map[p] = {"n": 0, "hits": 0}
        place_map[p]["n"] += 1
        if e["resolved"].get("hit"):
            place_map[p]["hits"] += 1

    by_place = {
        p: {**v, "rate": round(v["hits"] / v["n"] * 100, 1) if v["n"] else 0.0}
        for p, v in sorted(place_map.items())
    }

    # 村人別（1着予想の的中：picks[0]が1着か）
    member_map: dict[str, dict] = {}
    for e in resolved:
        res  = e["resolved"]
        win1 = res.get("1", "")
        for mr in e.get("member_results", []):
            kote = mr.get("kote", "?")
            if kote not in member_map:
                member_map[kote] = {"votes": 0, "top1_hits": 0}
            picks = mr.get("picks", [])
            member_map[kote]["votes"] += 1
            if picks and win1 and picks[0].get("name") == win1:
                member_map[kote]["top1_hits"] += 1

    by_member = {
        k: {**v, "rate": round(v["top1_hits"] / v["votes"] * 100, 1) if v["votes"] else 0.0}
        for k, v in sorted(member_map.items())
    }

    # 馬1頭ごとの一致度（consensus）集計（予測があるエントリのみ）
    consensus_vals = []
    for e in resolved:
        agg = e.get("aggregated", [])
        if not agg:
            continue
        # 上位3頭の consensus_pct 平均
        top3_cons = [h["consensus_pct"] for h in agg[:3] if h.get("consensus_pct") is not None]
        if top3_cons:
            consensus_vals.append(statistics.mean(top3_cons))
    avg_consensus = round(statistics.mean(consensus_vals), 1) if consensus_vals else None

    return {
        "ok":            True,
        "n_total":       n_total,
        "n_hit":         n_hit,
        "hit_rate_pct":  round(n_hit / n_total * 100, 1),
        "avg_consensus": avg_consensus,  # 予想の一致度平均（高いほど村の意見が揃っている）
        "by_place":      by_place,
        "by_member":     by_member,
    }


if __name__ == "__main__":
    import sys
    place  = sys.argv[1] if len(sys.argv) > 1 else "東京"
    kaisai = sys.argv[2] if len(sys.argv) > 2 else "20261108"
    race   = sys.argv[3] if len(sys.argv) > 3 else "11"
    result = run(place, kaisai, race)
    print(json.dumps(result, ensure_ascii=False, indent=2))
