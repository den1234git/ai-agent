"""
html_render_check.py

★2026-08-06追加: HTMLモックアップ生成の「見た目」と「時間経過・操作での
実際の変化」を検証するためのモジュール。

これまでの _check_html_semantic_match（router.py）はコードを読むだけのLLM判定で、
実際にブラウザで描画したときに要素がズレている・本当は動いていない、といった
「コードを読んだだけでは分からない」バグを検知できなかった。
実例1: アナログ時計のCSSで margin-top/margin-left のオフセットが抜けていて、
        針の回転軸が時計盤の中心からズレていた。コードだけ読むLLM判定では
        2回とも見抜けなかった。
実例2: 消し忘れの@keyframesアニメーションがJSのstyle.transformを上書きし、
        実際には現在時刻と無関係にループ回転し続けているだけだった。
        1枚のスクリーンショットでは「ある瞬間の針の角度」しか見えないため、
        本物の時刻反映か単なるループアニメか区別できず、これも見抜けなかった。

★2026-08-06 追加(2): 実例2を受けて、1枚ではなく一定間隔をあけた複数枚の
スクリーンショットをvisionモデルに一度に見せ、「時間経過や操作で実際に
変化しているか」も合わせて判定できるようにした。時計・カウンター・タイマー
のような時間で変わるものはもちろん、「2枚が全く同じ」であれば、それ自体が
JS/イベントが機能していない強い証拠になる。

依存: playwright（未インストールの場合は下記コマンドが必要）
  pip install playwright
  playwright install chromium

★重要: 上記のインストールをしていない環境では capture_html_screenshots() が
例外を送出する。check_html_visual_match() はその例外を捕まえて
checked=False（見た目チェック自体を実施できなかった＝判定不能）を返す。
呼び出し側（router.py）は checked=False のときは今まで通りコードチェックの
みで進める（他のチェック関数と同じフォールバック方針）。
"""
import base64
import os
import tempfile

DEFAULT_SHOT_COUNT = 2
DEFAULT_INTERVAL_MS = 1500


def capture_html_screenshots(
    html_content: str,
    shots: int = DEFAULT_SHOT_COUNT,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    width: int = 800,
    height: int = 600,
) -> list:
    """
    HTML文字列をヘッドレスブラウザで描画し、interval_msの間隔でshots枚の
    PNGスクリーンショットを撮る。時間で変化する要素(時計・カウンター等)が
    実際に動いているかどうかを、複数枚を並べて確認できるようにするため。

    戻り値: data URL形式のbase64文字列（data:image/png;base64,...）のリスト
    （撮影順、古い→新しい）。

    Playwrightが未インストールの場合はImportErrorを送出する
    （呼び出し側でtry/exceptしてスキップする想定）。
    """
    from playwright.sync_api import sync_playwright  # 未インストール時はここでImportError

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)

        file_url = "file:///" + tmp_path.replace(os.sep, "/")

        shot_data_urls = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(file_url)
                # setInterval等によるJSの初期描画が落ち着くのを少し待つ
                page.wait_for_timeout(800)

                for i in range(max(1, shots)):
                    screenshot_bytes = page.screenshot()
                    b64 = base64.b64encode(screenshot_bytes).decode("ascii")
                    shot_data_urls.append(f"data:image/png;base64,{b64}")
                    if i < shots - 1:
                        page.wait_for_timeout(interval_ms)
            finally:
                browser.close()

        return shot_data_urls
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def build_html_visual_check_prompt(task: str, num_shots: int, interval_ms: int) -> str:
    interval_sec = interval_ms / 1000
    return f"""
これはUIモックアップとして生成されたHTMLページのスクリーンショットです。
{num_shots}枚を、約{interval_sec}秒間隔で撮影した順番（古い→新しい）で並べています。

依頼内容：
{task}

この画像を見て、以下の観点で判定してください。
1. 見た目：要素同士が重なっていないか、意図した位置からズレていないか、表示が崩れていないか。
2. 動き：依頼内容が時間経過や操作で変化するもの（時計・カウンター・タイマー・
   進捗バーなど）を求めている場合、複数枚の画像を比べて実際にその変化が
   起きているか。もし変化しているはずなのに全ての画像が同じに見える場合は、
   JavaScriptが正しく動作していない証拠として不一致と判定すること。
   （時計であれば、単に針が回っているだけでなく、経過時間相当だけ動いているかも見ること）

判定結果は必ず先頭を「OK」または「NG」で始め、そのあとに一言で理由を書いてください。
例：「OK: 中央に円形の時計が表示され、2枚の間で秒針が経過時間分だけ進んでいる」
例：「NG: 2枚の画像で針の位置が全く変化していない」
例：「NG: 時計の針が時計盤の中心からズレて左端に寄っている」
""".strip()


def check_html_visual_match(
    task: str,
    html_content: str,
    shots: int = DEFAULT_SHOT_COUNT,
    interval_ms: int = DEFAULT_INTERVAL_MS,
) -> dict:
    """
    実際にレンダリングした複数枚のスクリーンショットをvisionモデルに見せて、
    「見た目」と「時間経過での実際の変化」の両方を判定する。

    戻り値: {"matches": bool, "reason": str, "checked": bool}
    checked=False は、playwright未導入などの理由で見た目チェック自体を
    実施できなかった（＝判定不能。コードチェックのみにフォールバックすべき）
    ことを示す。matches はそのとき常にTrue（生成を止めない方針）。
    """
    from ai import ask_ai

    try:
        screenshots = capture_html_screenshots(html_content, shots=shots, interval_ms=interval_ms)
    except Exception as e:
        print(f"[VISUAL_CHECK] スクリーンショット取得に失敗、見た目チェックをスキップします: {e}")
        return {"matches": True, "reason": "見た目チェックをスキップ（スクリーンショット取得不可）", "checked": False}

    try:
        result = ask_ai(
            build_html_visual_check_prompt(task, len(screenshots), interval_ms),
            model="vision",
            image_base64=screenshots,
            json_mode=False,
        )
        text = ""
        if isinstance(result, dict):
            text = result.get("code", "") or ""
            if not text:
                for a in result.get("actions", []):
                    if a.get("content"):
                        text = a["content"]
                        break
        text = (text or "").strip()

        stripped = text.upper()
        matches = not stripped.startswith("NG")
        return {"matches": matches, "reason": text or "判定理由なし", "checked": True}
    except Exception as e:
        print(f"[VISUAL_CHECK] vision判定に失敗、見た目チェックをスキップします: {e}")
        return {"matches": True, "reason": "見た目チェックをスキップ（vision判定失敗）", "checked": False}
