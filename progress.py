"""
処理の進捗状態をプロセス内メモリで保持するだけの簡易モジュール。
WebSocketは使わず、フロント側がポーリング（GET /chat/status）で
この状態を読みに来る想定。

1リクエスト=1処理が前提（同時に複数の生成を並行させない）。

「中断」対応について：
Ollamaへの実際のHTTPリクエスト（requests.post）はキャンセルできないため、
中断ボタンが押された後も裏側のスレッドは処理を続け、いずれ finish() を呼ぶ。
これが画面に反映されてしまわないよう、リクエストごとに generation 番号を持たせ、
abort() が呼ばれた世代の結果は無視する。
"""

import threading

_lock = threading.Lock()

_generation = 0

_state = {
    "status": "idle",      # idle | running | done | error | aborted
    "percent": 0,
    "step": "",             # 現在のステップ名（人間向け表示用）
    "result": None,         # 完了時：route()の戻り値（dict）
    "error": None,          # エラー時：エラーメッセージ
}


def start_new() -> int:
    """新しい処理を開始する。この処理専用の世代番号を返す。"""
    global _generation
    with _lock:
        _generation += 1
        _state.update({
            "status": "running",
            "percent": 0,
            "step": "受付中",
            "result": None,
            "error": None,
        })
        return _generation


def update(generation: int, percent: int, step: str):
    with _lock:
        if generation != _generation:
            return  # 中断済み・古い世代からの更新は無視
        _state["percent"] = percent
        _state["step"] = step
        _state["status"] = "running"


def finish(generation: int, result: dict):
    with _lock:
        if generation != _generation:
            return  # すでに中断・次の処理が始まっている場合は無視
        _state["status"] = "done"
        _state["percent"] = 100
        _state["step"] = "完了"
        _state["result"] = result


def fail(generation: int, error_message: str):
    with _lock:
        if generation != _generation:
            return
        _state["status"] = "error"
        _state["error"] = error_message


def abort():
    """ユーザー操作による中断。現在の世代を無効化し、以降の更新を無視させる。"""
    global _generation
    with _lock:
        _generation += 1  # 世代を進めることで、進行中スレッドからの更新を無効化する
        _state.update({
            "status": "aborted",
            "step": "中断しました",
        })


def snapshot():
    with _lock:
        return dict(_state)
