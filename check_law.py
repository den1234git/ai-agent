import requests
import xml.etree.ElementTree as ET

keywords = ["消防法施行令", "消防法施行規則", "建築基準法施行令"]

for category in range(1, 15):
    url = f"https://laws.e-gov.go.jp/api/1/lawlists/{category}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=30)
    if res.status_code != 200:
        continue
    try:
        root = ET.fromstring(res.text)
    except:
        continue
    for law in root.iter("LawNameListInfo"):
        name = law.find("LawName")
        law_id = law.find("LawId")
        if name is not None:
            for kw in keywords:
                if kw == name.text:
                    print(f"カテゴリ{category} LawName: {name.text}, LawId: {law_id.text}")