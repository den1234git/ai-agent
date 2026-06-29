import requests
import xml.etree.ElementTree as ET
from memory_store import save_shobo_knowledge

LAWS = [
    {"id": "323AC1000000186", "name": "消防法"},
    {"id": "336CO0000000037", "name": "消防法施行令"},
    {"id": "336M50000008006", "name": "消防法施行規則"},  # ← 修正
    {"id": "325AC0000000201", "name": "建築基準法"},
    {"id": "325CO0000000338", "name": "建築基準法施行令"},
]

def fetch_law(law_id, law_name):
    url = f"https://laws.e-gov.go.jp/api/1/lawdata/{law_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    print(f"\n{law_name}を取得中...")
    res = requests.get(url, headers=headers, timeout=30)
    print("ステータス:", res.status_code)

    root = ET.fromstring(res.text)

    count = 0
    for article in root.iter("Article"):
        num = article.get("Num", "")
        text = "".join(article.itertext()).strip()

        if len(text) < 20:
            continue

        title = f"{law_name}第{num}条"
        save_shobo_knowledge(
            content=text[:500],
            title=title,
            source="e-gov",
            category=law_name,
        )
        count += 1
        print(f"保存: {title}")

    print(f"完了: {count}件")
    return count

if __name__ == "__main__":
    total = 0
    for law in LAWS:
        total += fetch_law(law["id"], law["name"])
    print(f"\n全体完了: {total}件保存しました")