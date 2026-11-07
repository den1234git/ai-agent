"""
「仕様インタビュー」機能。

shobo-appの新しい画面・機能を作る前に、複数ラウンドかけて仕様を深掘りする
専用ボタン用のバックエンドロジック。★2026-08-02 追加。

★設計上の注意（ユーザーからの明確な要望）:
  「作ってもずっと質問だけで進まないかも」という懸念があったため、
  必ずどこかで終わる設計にしている。
    - ラウンド数固定（5/10/20）の場合: その回数で必ず終了する。
    - 「満足するまで」の場合: AIが十分と判断すれば終了するが、
      それとは別に MAX_INTERVIEW_ROUNDS_HARD_CAP による強制終了も必ず働く
      （AIの判定自体が失敗し続けても無限ループにならないようにする安全装置）。

LAは単一ユーザーのローカルアプリのため、インタビューの状態はモジュール内の
グローバル変数で保持する（同時に複数インタビューが走ることは想定しない）。
"""
from ai import ask_ai, OllamaUnavailableError
from build_prompt import (
    build_interview_question_prompt,
    build_interview_satisfaction_prompt,
    build_interview_summary_prompt,
)

# ★安全装置：「満足するまで」モードでも、AIの判定が機能しなかった場合に
# 備えて、ここを超えたら必ず終了する。
MAX_INTERVIEW_ROUNDS_HARD_CAP = 10

_state = {
    "active": False,
    "topic": "",
    "max_rounds": None,   # None = 満足するまで（ただしハードキャップは常に適用）
    "round": 0,
    "qa_pairs": [],        # [{"question": ..., "answer": ...}]
    "pending_question": "",
}


def _reset():
    global _state
    _state = {
        "active": False,
        "topic": "",
        "max_rounds": None,
        "round": 0,
        "qa_pairs": [],
        "pending_question": "",
    }


def start_interview(topic: str, max_rounds: int | None) -> dict:
    """
    インタビューを開始し、最初の質問を返す。
    max_rounds: 5 / 10 / 20 のいずれか、または None（満足するまで）。
    """
    topic = (topic or "").strip()
    if not topic:
        return {"status": "error", "message": "テーマが空です。作りたいものを一言でも入力してください。"}

    global _state
    _state = {
        "active": True,
        "topic": topic,
        "max_rounds": max_rounds,
        "round": 0,
        "qa_pairs": [],
        "pending_question": "",
    }
    return _ask_next_question()


def submit_answer(answer: str) -> dict:
    """
    直前の質問への回答を記録し、次の質問 or 終了（まとめ）を返す。
    """
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "インタビューが開始されていません。"}

    answer = (answer or "").strip()
    _state["qa_pairs"].append({
        "question": _state["pending_question"],
        "answer": answer if answer else "（回答なし）",
    })

    # ハードキャップ：これを超えたら理由を問わず必ず終了する
    if _state["round"] >= MAX_INTERVIEW_ROUNDS_HARD_CAP:
        return _finish_interview()

    # 指定した質問数（上限）に達したら、そこで必ず終了する
    if _state["max_rounds"] is not None and _state["round"] >= _state["max_rounds"]:
        return _finish_interview()

    # ★2026-08-02 追加：質問数を固定していても（5/10/20問）、AIが
    # 「もう十分」と判断すればそこで早めに終了できるようにする。
    # 理由：質問数を多めに指定すると、聞くことがなくなった後も無理に
    # 数を埋めようとして質問内容が散漫（枝葉末節）になるという懸念が
    # あったため。max_roundsは「必ずその数だけ聞く」ではなく
    # 「最大でもその数まで」という上限として扱う。
    try:
        result = ask_ai(
            build_interview_satisfaction_prompt(_state["topic"], _state["qa_pairs"]),
            model="router",
        )
        satisfied = bool(result.get("satisfied", False))
    except Exception as e:
        print(f"[INTERVIEW] 満足度判定に失敗しました（続行します）: {e}")
        satisfied = False

    if satisfied:
        return _finish_interview()
    return _ask_next_question()


def cancel_interview() -> dict:
    _reset()
    return {"status": "cancelled"}


def _ask_next_question() -> dict:
    global _state
    _state["round"] += 1
    try:
        result = ask_ai(
            build_interview_question_prompt(
                _state["topic"], _state["qa_pairs"], _state["round"], _state["max_rounds"]
            ),
            model="router",
        )
        question = (result.get("question") or "").strip()
    except OllamaUnavailableError as e:
        _reset()
        return {"status": "error", "message": f"Ollamaが起動していません: {e}"}
    except Exception as e:
        print(f"[INTERVIEW] 質問生成に失敗しました: {e}")
        question = ""

    if not question:
        # 質問生成に失敗した場合、無理に続けず終了してまとめに進む
        print("[INTERVIEW] 質問が空だったため、ここで打ち切ってまとめに進みます。")
        return _finish_interview()

    _state["pending_question"] = question
    return {
        "status": "question",
        "round": _state["round"],
        "max_rounds": _state["max_rounds"],
        "question": question,
    }


def _finish_interview() -> dict:
    global _state
    topic = _state["topic"]
    qa_pairs = _state["qa_pairs"]

    summary = ""
    try:
        result = ask_ai(build_interview_summary_prompt(topic, qa_pairs), model="swallow")
        summary = (result.get("summary") or "").strip()
    except Exception as e:
        print(f"[INTERVIEW] まとめ生成に失敗しました（質疑応答をそのまま使います）: {e}")

    if not summary:
        # まとめ生成に失敗した場合のフォールバック：質疑応答をそのまま並べる
        lines = [f"{topic}という画面/コンポーネントを作ってください。", ""]
        for qa in qa_pairs:
            lines.append(f"・{qa['question']} → {qa['answer']}")
        summary = "\n".join(lines)

    _reset()
    return {"status": "done", "summary": summary}
