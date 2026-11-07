"""
「壁打ち」機能。

アイデアの深掘り・思考の整理を目的とする対話モード。
回数は無制限で、ユーザーが「まとめる」を押すまで続く。
AIは質問だけでなく、提案・アイデアも出す。
まとめた内容はそのまま制作パイプラインに流せる。
"""
from ai import ask_ai, OllamaUnavailableError
from build_prompt import build_kabeuchi_question_prompt, build_kabeuchi_summary_prompt

_state = {
    "active": False,
    "topic": "",
    "round": 0,
    "qa_pairs": [],
    "pending_question": "",
}


def _reset():
    global _state
    _state = {
        "active": False,
        "topic": "",
        "round": 0,
        "qa_pairs": [],
        "pending_question": "",
    }


def start_kabeuchi(topic: str) -> dict:
    topic = (topic or "").strip()
    if not topic:
        return {"status": "error", "message": "テーマが空です。考えたいことを入力してください。"}

    global _state
    _state = {
        "active": True,
        "topic": topic,
        "round": 0,
        "qa_pairs": [],
        "pending_question": "",
    }
    return _ask_next_question()


def submit_answer(answer: str) -> dict:
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "壁打ちが開始されていません。"}

    answer = (answer or "").strip()
    _state["qa_pairs"].append({
        "question": _state["pending_question"],
        "answer": answer if answer else "（回答なし）",
    })

    return _ask_next_question()


def finish_kabeuchi() -> dict:
    """ユーザーが「まとめる」を押した時に呼ばれる。"""
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "壁打ちが開始されていません。"}

    topic = _state["topic"]
    qa_pairs = _state["qa_pairs"]

    summary = ""
    try:
        result = ask_ai(build_kabeuchi_summary_prompt(topic, qa_pairs), model="swallow")
        summary = (result.get("summary") or "").strip()
    except Exception as e:
        print(f"[KABEUCHI] まとめ生成に失敗しました（質疑応答をそのまま使います）: {e}")

    if not summary:
        lines = [f"テーマ「{topic}」についての壁打ちまとめ：", ""]
        for qa in qa_pairs:
            lines.append(f"・{qa['question']} → {qa['answer']}")
        summary = "\n".join(lines)

    _reset()
    return {"status": "done", "summary": summary}


def cancel_kabeuchi() -> dict:
    _reset()
    return {"status": "cancelled"}


def _ask_next_question() -> dict:
    global _state
    _state["round"] += 1
    try:
        result = ask_ai(
            build_kabeuchi_question_prompt(
                _state["topic"], _state["qa_pairs"], _state["round"]
            ),
            model="router",
        )
        question = (result.get("question") or "").strip()
    except OllamaUnavailableError as e:
        _reset()
        return {"status": "error", "message": f"Ollamaが起動していません: {e}"}
    except Exception as e:
        print(f"[KABEUCHI] 質問生成に失敗しました: {e}")
        question = ""

    if not question:
        question = "もう少し詳しく教えてもらえる？具体的にはどんなイメージ？"

    _state["pending_question"] = question
    return {
        "status": "question",
        "round": _state["round"],
        "question": question,
    }
