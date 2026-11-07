"""
VOICEVOX Engine（ローカルAPI, http://localhost:50021）を叩いて
テキストから音声(wav)を生成する薄いラッパー。

前提：VOICEVOX（GUI版 or Engine単体）が起動していて、
50021番ポートでAPIが立ち上がっていること。
"""

import os
import re
import time
import requests

VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 2  # 四国めたん（ノーマル）

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sanitize_for_tts(text: str) -> str:
    """
    answers に入ってくるテキストには、コードブロックやJSON断片が
    混ざることがあるため、読み上げに適さない記号類を簡易的に除去する。
    完全なクリーニングではなく、読み上げが極端に崩れるのを防ぐ最低限の処理。
    """
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # コードブロック除去
    text = re.sub(r"[{}\\[\\]<>]", "", text)  # 記号類除去
    text = text.strip()
    return text


def synthesize(text: str, speaker: int = SPEAKER_ID) -> str | None:
    """
    テキストを音声合成し、生成したwavファイルのパスを返す。
    VOICEVOXが起動していない等で失敗した場合は None を返す
    （呼び出し側はNoneなら音声なしで続行できるようにする）。
    """
    if not text:
        return None
    clean_text = _sanitize_for_tts(text)
    if not clean_text:
        return None

    try:
        # ① audio_query: テキスト→音声合成用クエリ生成
        query_res = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": clean_text, "speaker": speaker},
            timeout=30,
        )
        query_res.raise_for_status()
        query = query_res.json()

        # ② synthesis: クエリ→wavバイナリ生成
        synth_res = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker},
            json=query,
            timeout=60,
        )
        synth_res.raise_for_status()

        filename = f"tts_{int(time.time() * 1000)}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(synth_res.content)

        return filename

    except Exception as e:
        print(f"[TTS] 音声合成に失敗しました（VOICEVOXが起動していない可能性）: {e}")
        return None