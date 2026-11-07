"""
WikiSkill Maintainer
失敗/村相談が起きた対話を分析して la_wiki.json にノートを追記する。
"""
import json
import os
import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
WIKI_FILE = os.path.join(_BASE, "la_wiki.json")
WIKI_MAX = 200


def _load() -> list:
    if not os.path.exists(WIKI_FILE):
        return []
    try:
        with open(WIKI_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(data: list):
    with open(WIKI_FILE, "w", encoding="utf-8") as f:
        json.dump(data[-WIKI_MAX:], f, ensure_ascii=False, indent=2)


def analyze_and_write(
    question: str,
    la_answer: str,
    outcome: str,
    village_summary: str = "",
) -> dict:
    """
    対話を分析してwikiノートを1件書く。バックグラウンドスレッドから呼んでよい。

    outcome:
        "village_consulted"  - 村相談が起動した（LAが自信なし or 確信度低）
        "user_marked_bad"    - ユーザーが「悪い回答」と評価した
        "user_marked_good"   - ユーザーが「良い回答」と評価した
    """
    from ai import ask_ai

    village_block = f"\n村の結論: {village_summary[:200]}" if village_summary else ""
    prompt = f"""以下の対話を分析してください。

質問: {question[:300]}
LAの回答: {la_answer[:300] if la_answer else "（なし）"}
結果: {outcome}{village_block}

分析してください：
1. LAが失敗した（または自信がなかった）原因を一言で
2. このケースから得られる教訓を1文で
3. 同じ失敗を防ぐためのルールを「〜の場合は〜すべき/すべきでない」の形で

以下のJSONのみ返してください：
{{"failure_pattern": "失敗の原因", "lesson": "教訓1文", "rule": "防止ルール"}}
"""
    failure_pattern, lesson, rule = "（分析失敗）", "", ""
    try:
        result = ask_ai(prompt, model="swallow")
        failure_pattern = (result.get("failure_pattern") or "").strip() or "（分析失敗）"
        lesson = (result.get("lesson") or "").strip()
        rule = (result.get("rule") or "").strip()
    except Exception as e:
        print(f"[WIKI_MAINTAINER] 分析エラー: {e}")

    note = {
        "id": f"wiki_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.date.today().isoformat(),
        "question": question[:200],
        "la_answer": la_answer[:200] if la_answer else "",
        "outcome": outcome,
        "village_summary": village_summary[:200] if village_summary else "",
        "failure_pattern": failure_pattern,
        "lesson": lesson,
        "rule": rule,
        "used_in_skill": False,
    }

    wiki = _load()
    wiki.append(note)
    _save(wiki)
    print(f"[WIKI_MAINTAINER] ノート追記: {note['id']} / {failure_pattern[:50]}")
    return note


def get_unused_notes() -> list:
    return [n for n in _load() if not n.get("used_in_skill")]


def get_all_notes() -> list:
    return _load()


def mark_used(note_ids: list):
    """指定ノートを used_in_skill=True にする"""
    wiki = _load()
    changed = False
    for note in wiki:
        if note.get("id") in note_ids:
            note["used_in_skill"] = True
            changed = True
    if changed:
        with open(WIKI_FILE, "w", encoding="utf-8") as f:
            json.dump(wiki, f, ensure_ascii=False, indent=2)
