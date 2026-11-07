"""
WikiSkill Proposer
wiki ノートを読んでスキル追加を1件提案し、承認後に la_skills.json に反映する。
"""
import json
import os
import datetime

_BASE = os.path.dirname(os.path.abspath(__file__))
SKILLS_FILE = os.path.join(_BASE, "la_skills.json")
SKILLS_MAX = 20


def _load_skills() -> dict:
    if not os.path.exists(SKILLS_FILE):
        return {"version": 1, "skills": []}
    try:
        with open(SKILLS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"version": 1, "skills": []}
    except Exception:
        return {"version": 1, "skills": []}


def _save_skills(data: dict):
    with open(SKILLS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_skills() -> list:
    return _load_skills().get("skills", [])


def propose_skill_update() -> dict:
    """
    未使用のwikiノートからスキル追加を1件提案する（まだ la_skills.json には書かない）。
    返値: {"status": "proposed", "rule": ..., "reasoning": ..., "based_on_ids": [...]}
          {"status": "no_notes", ...} ノートがない場合
          {"status": "error", ...}   失敗した場合
    """
    from wiki_maintainer import get_unused_notes
    from ai import ask_ai

    notes = get_unused_notes()
    if not notes:
        return {"status": "no_notes", "message": "未使用のwikiノートがありません"}

    existing_skills = get_skills()
    existing_rules = (
        "\n".join(f"- {s['rule']}" for s in existing_skills)
        if existing_skills
        else "（なし）"
    )

    # 直近10件のノートを使う
    recent = notes[-10:]
    notes_text = "\n".join(
        f"[{n['id']}] 失敗パターン: {n['failure_pattern']} / ルール候補: {n['rule']}"
        for n in recent
    )
    candidate_ids = [n["id"] for n in recent]

    prompt = f"""以下は失敗から学んだwikiノートです。

wikiノート（直近{len(recent)}件）:
{notes_text}

現在のスキル（既存ルール）:
{existing_rules}

これらのノートをもとに、LAが今後使うべきルールを1つだけ提案してください。
条件：
- 既存ルールと重複しないこと
- 「〜の場合は〜すべき」または「〜の場合は〜すべきでない」の形で30文字以内
- できるだけ複数のノートに共通するパターンを選ぶこと

以下のJSONのみ返してください：
{{"rule": "提案ルール", "reasoning": "提案理由（1文）", "based_on_ids": ["wiki_idA", "wiki_idB"]}}
"""
    try:
        result = ask_ai(prompt, model="swallow")
        rule = (result.get("rule") or "").strip()
        reasoning = (result.get("reasoning") or "").strip()
        based_on = result.get("based_on_ids") or candidate_ids[:3]
        # based_on が candidate_ids の範囲内かチェック
        based_on = [b for b in based_on if b in candidate_ids][:5] or candidate_ids[:3]
    except Exception as e:
        return {"status": "error", "message": str(e)}

    if not rule:
        return {"status": "error", "message": "ルールの生成に失敗しました"}

    return {
        "status": "proposed",
        "rule": rule,
        "reasoning": reasoning,
        "based_on_ids": based_on,
        "proposed_at": datetime.datetime.now().isoformat(),
    }


def apply_proposal(rule: str, based_on_ids: list) -> dict:
    """
    承認されたプロポーザルを la_skills.json に追加し、
    使用済みwikiノートにフラグを立てる。
    """
    from wiki_maintainer import mark_used

    skills_data = _load_skills()
    skills = skills_data.get("skills", [])

    new_skill = {
        "id": f"skill_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "rule": rule,
        "based_on": based_on_ids,
        "adopted_at": datetime.date.today().isoformat(),
    }
    skills.append(new_skill)
    skills_data["skills"] = skills[-SKILLS_MAX:]
    skills_data["version"] = skills_data.get("version", 1) + 1
    _save_skills(skills_data)

    mark_used(based_on_ids)

    print(f"[WIKI_PROPOSER] スキル適用: {new_skill['id']} / {rule[:60]}")
    return {"status": "applied", "skill": new_skill}


def delete_skill(skill_id: str) -> dict:
    """スキルを削除する（間違ったルールを外したいとき用）"""
    skills_data = _load_skills()
    before = len(skills_data.get("skills", []))
    skills_data["skills"] = [s for s in skills_data.get("skills", []) if s["id"] != skill_id]
    after = len(skills_data["skills"])
    if before == after:
        return {"status": "not_found"}
    skills_data["version"] = skills_data.get("version", 1) + 1
    _save_skills(skills_data)
    return {"status": "deleted", "skill_id": skill_id}
