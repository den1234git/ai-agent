"""
e-Gov法令APIから条文を取得する共通処理。
import_shobo_law.py（初期一括投入用）と learning.py（通常学習サイクル用）の
両方から呼べるように切り出したもの。
"""

import requests
import xml.etree.ElementTree as ET

LAWS = [
    {"id": "323AC1000000186", "name": "消防法"},
    {"id": "336CO0000000037", "name": "消防法施行令"},
    {"id": "336M50000008006", "name": "消防法施行規則"},
    {"id": "325AC0000000201", "name": "建築基準法"},
    {"id": "325CO0000000338", "name": "建築基準法施行令"},
    {"id": "417M60000008038", "name": "消防法及び石油コンビナート等災害防止法の規定に基づく民間事業者等が行う書面の保存等における情報通信の技術の利用に関する規則"},
    {"id": "422M60000008112", "name": "消火器の技術上の規格を定める省令の一部を改正する省令の施行に伴う消防法施行令第三十条第二項及び危険物の規制に関する政令第二十二条第二項の技術上の基準に関する特例を定める省令"},
    {"id": "425M60000008027", "name": "消防用ホースの技術上の規格を定める省令等の施行に伴う消防法施行令第三十条第二項及び危険物の規制に関する政令第二十二条第二項の技術上の基準に関する特例を定める省令"},
    # ★ 学習サイクル用に周辺法令を追加（必要に応じて増やしてください）
    {"id": "334AC0000000045", "name": "労働安全衛生法"},
]

EGOV_API_URL = "https://laws.e-gov.go.jp/api/1/lawdata/{law_id}"


def find_law_by_query(query: str) -> dict:
    """クエリ文字列（法令名の一部）にマッチする法令をLAWSから探す。見つからなければ先頭にフォールバック。"""
    q = (query or "").strip()
    for law in LAWS:
        if q and q in law["name"]:
            return law
    return LAWS[0]


def fetch_law_articles(law_id: str, law_name: str, max_articles: int = 3) -> dict:
    """
    指定した法令から条文をいくつか取得する。
    戻り値: {"ok": bool, "items": [{"title":..., "content":...}], "error": str|None}
    """
    url = EGOV_API_URL.format(law_id=law_id)
    headers = {"User-Agent": "AI_001-learning/1.0"}
    try:
        res = requests.get(url, headers=headers, timeout=30)
        res.raise_for_status()
        root = ET.fromstring(res.text)

        items = []
        for article in root.iter("Article"):
            num = article.get("Num", "")
            text = "".join(article.itertext()).strip()
            if len(text) < 20:
                continue
            items.append({"title": f"{law_name}第{num}条", "content": text[:500]})
            if len(items) >= max_articles:
                break

        return {"ok": True, "items": items, "error": None}
    except Exception as e:
        return {"ok": False, "items": [], "error": str(e)}