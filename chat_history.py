"""
直近の会話ターンをプロセスメモリ上に保持する、軽量な会話記憶。

★2026-07-30 新設: 「なんで失敗したの？」のような直前の文脈を踏まえた
質問に、/chat が毎回「初対面」として答えてしまい的外れになる問題への対応。

設計方針（意図的にシンプルにしている）：
- DBは使わず、サーバープロセスの寿命の間だけ有効なリストで十分とする
  （再起動すればリセットされてよい。ChatGPT的な無制限保存ではない）。
- モデルのコンテキスト長が4096トークンしかなく、shobo_knowledge検索結果や
  社訓（motto）と同じ枠を奪い合うため、保持件数・文字数を絞る。
- 生成失敗時の実際のエラー内容（ユーザーには見せていない詳細）も
  别途保持し、「なんで失敗したの？」に答えられるようにする。
"""

MAX_TURNS = 3  # 直近何往復まで保持するか（1往復 = user発言1件 + LA発言1件）
MAX_CHARS_PER_ENTRY = 300  # 1件あたりの最大文字数（プロンプト肥大化防止）
MAX_ERROR_ENTRIES = 3

_turns: list[dict] = []
_recent_errors: list[dict] = []


def _truncate(text: str, limit: int = MAX_CHARS_PER_ENTRY) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + "…（以下省略）"
    return text


def add_turn(role: str, content: str) -> None:
    """role: "user" または "assistant"。"""
    if not content or not content.strip():
        return
    _turns.append({"role": role, "content": _truncate(content)})
    # 直近 MAX_TURNS 往復分（= MAX_TURNS * 2 件）だけ残す
    max_entries = MAX_TURNS * 2
    if len(_turns) > max_entries:
        del _turns[: len(_turns) - max_entries]


def add_error(user_message: str, detail: str) -> None:
    """
    ユーザーには見せていない生成失敗の詳細を記録する。
    直後に「なんで失敗したの？」と聞かれたときに参照できるようにするため。
    """
    if not detail:
        return
    _recent_errors.append({
        "user_message": _truncate(user_message, 100),
        "detail": _truncate(detail),
    })
    if len(_recent_errors) > MAX_ERROR_ENTRIES:
        del _recent_errors[: len(_recent_errors) - MAX_ERROR_ENTRIES]


def get_recent_context() -> str:
    """
    プロンプトに差し込むための短いテキストブロックを返す。
    何も保持していなければ空文字を返す（呼び出し側で無条件に差し込んでよい）。
    """
    if not _turns and not _recent_errors:
        return ""

    lines = []
    if _turns:
        lines.append("[直近のやりとり]")
        for t in _turns:
            speaker = "ユーザー" if t["role"] == "user" else "LA"
            lines.append(f"{speaker}: {t['content']}")

    if _recent_errors:
        lines.append("[直近の生成失敗の詳細（ユーザーには見せていない内部情報）]")
        for e in _recent_errors:
            lines.append(f"「{e['user_message']}」への対応中に失敗: {e['detail']}")

    return "\n".join(lines)


def clear() -> None:
    _turns.clear()
    _recent_errors.clear()
