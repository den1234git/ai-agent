"""
faster-whisper + webrtcvad を使ったローカル音声認識(STT)モジュール。

VOICEVOX(tts.py)と対になる構成：
- tts.py  : テキスト → 音声（VOICEVOX Engine, localhost:50021）
- stt.py  : 音声 → テキスト（faster-whisper, ローカルで直接推論）

VAD(音声区間検出)で「人の声っぽい区間」だけを検出し、
その区間がまとまったタイミングでWhisperに渡して文字起こしする。
無音・環境音（物音、テレビ等）はそのまま無視される。
"""

import io
import wave
import collections
import webrtcvad
from faster_whisper import WhisperModel

# --- Whisperモデルのロード（プロセス起動時に1回だけ） ---
# base: 精度と速度のバランス型。GTX 1050(2GB VRAM)でも実用速度。
# device="cpu" にしているのは、VOICEVOX(GPU/CPU) と同時に動かす際に
# VRAM圧迫を避けるための保険。速度的にbaseならCPUでも十分実用範囲。
_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- VAD設定 ---
# aggressiveness: 0(甘い)〜3(厳しい)。数字が大きいほど「声」と判定する基準が厳しくなり、
# 物音等を誤検出しにくくなる代わりに、小さい声を拾い損ねることがある。
# 当初は2で運用していたが、環境音・ファンノイズ等を誤検出してWhisperのハルシネーション
# （空耳的な定型文の生成）を招いていたため、最も厳しい3に引き上げた。
_vad = webrtcvad.Vad(3)

SAMPLE_RATE = 16000          # Whisper/webrtcvad共通の前提サンプルレート
FRAME_DURATION_MS = 30       # 1フレームあたりの長さ（webrtcvadは10/20/30msのみ対応）
FRAME_BYTES = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0) * 2)  # 16bit=2byte

# 発話とみなすための連続無音判定（この長さ無音が続いたら「発話終わり」とみなす）
SILENCE_TIMEOUT_MS = 800
SILENCE_FRAMES_THRESHOLD = SILENCE_TIMEOUT_MS // FRAME_DURATION_MS

# 発話区間として認める最小の長さ（これより短い区間はノイズ・物音として無視する。
# 以前は FRAME_BYTES * 5 = 150ms 相当だったが、これだと「カツン」等の単発物音でも
# VAD(2)では発話扱いされてしまい、Whisperに極短音声が渡ってハルシネーションの
# 主因になっていた。人の発話として自然な最短の長さを踏まえ 500ms 相当まで引き上げる）
MIN_SPEECH_FRAMES = int(500 / FRAME_DURATION_MS)  # 500ms分のフレーム数

# faster-whisperが「無音に近い」と自己申告した場合の閾値。
# no_speech_probがこれを超えるセグメントはハルシネーションの可能性が高いため破棄する。
NO_SPEECH_PROB_THRESHOLD = 0.6

# 平均対数確率（モデルの自信度）の下限。これより低い（自信がない）セグメントは、
# 「聞き取れていないのに適当に単語を埋めた」ハルシネーションの可能性が高いため破棄する。
# faster-whisper公式が推奨するハルシネーション検出の標準的な閾値。
AVG_LOGPROB_THRESHOLD = -1.0

# 圧縮率（生成テキストの繰り返し度合い）の上限。これを超えると、
# 「同じような音節の繰り返し」等、不自然なパターンのハルシネーションである可能性が高い。
COMPRESSION_RATIO_THRESHOLD = 2.4

# Whisperが無音・ノイズ入力時によく生成する定型ハルシネーションフレーズ。
# 部分一致でも弾く（「ご視聴ありがとうございました」等のYouTube的な締め文言が典型）。
HALLUCINATION_PHRASES = [
    "ご視聴いただき",
    "ご視聴ありがとう",
    "チャンネル登録",
    "最後までご視聴",
    "字幕視聴者の皆様",
    "この動画が良かったら",
    "またお会いしましょう",
    "お疲れ様でした",
]


def _looks_like_hallucination(text: str) -> bool:
    """定型ハルシネーションフレーズに一致するかを簡易判定する。"""
    return any(phrase in text for phrase in HALLUCINATION_PHRASES)


class VADSession:
    """
    WebSocket 1接続＝1セッションとして使う。
    フレーム単位(30ms)のPCM音声を受け取りながら、
    「発話区間がまとまった」タイミングで文字起こし結果を返す。
    """

    def __init__(self):
        self._voiced_frames = []
        self._silence_count = 0
        self._triggered = False
        self._buffer = b""  # フレーム境界に満たない端数を貯めておく

    def add_chunk(self, pcm_bytes: bytes) -> str | None:
        """
        音声チャンク(任意サイズのPCM16LE, 16kHz, mono)を追加する。
        発話の区切りを検出できたら文字起こし結果(str)を返す。
        まだ区切りが来ていない場合は None を返す。
        """
        self._buffer += pcm_bytes
        result_text = None

        while len(self._buffer) >= FRAME_BYTES:
            frame = self._buffer[:FRAME_BYTES]
            self._buffer = self._buffer[FRAME_BYTES:]

            is_speech = _vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                self._voiced_frames.append(frame)
                self._silence_count = 0
                self._triggered = True
            else:
                if self._triggered:
                    self._silence_count += 1
                    self._voiced_frames.append(frame)  # 語尾の余韻を残すため無音も少し含める

                    if self._silence_count >= SILENCE_FRAMES_THRESHOLD:
                        # 発話区間が確定 → 文字起こし実行
                        result_text = self._transcribe(self._voiced_frames)
                        self._voiced_frames = []
                        self._silence_count = 0
                        self._triggered = False

        return result_text

    def _transcribe(self, frames: list) -> str | None:
        pcm_data = b"".join(frames)
        if len(pcm_data) < FRAME_BYTES * MIN_SPEECH_FRAMES:  # 極端に短い区間はノイズとみなして無視
            return None

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        wav_buffer.seek(0)

        try:
            segments, info = _model.transcribe(
                wav_buffer,
                language="ja",
                beam_size=5,
                # 直前の文字起こし結果を次のセグメントの文脈として使わない。
                # これをTrue（faster-whisperのデフォルト）のままにしておくと、
                # 一度ハルシネーションが出た際にその内容が「文脈」として次の
                # セグメントにも引き継がれ、同じ幻覚フレーズが連鎖・増幅しやすい。
                condition_on_previous_text=False,
                # no_speech_prob等の付加情報を取得するためVAD filterはfaster-whisper側では使わず
                # （webrtcvadで既に区間検出済みのため）、セグメント単位の自己申告確率のみ利用する。
            )

            text_parts = []
            for seg in segments:
                # Whisper自身が「これはほぼ無音区間だった」と申告しているセグメントは、
                # webrtcvadの検出漏れ（環境音等）に対するハルシネーションの可能性が高いため破棄する
                if getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_PROB_THRESHOLD:
                    print(f"[STT] no_speech_prob={seg.no_speech_prob:.2f} のため破棄: {seg.text!r}")
                    continue

                # 自信度が極端に低いセグメントも、意味不明な単語列を「それっぽく」生成した
                # ハルシネーションの可能性が高いため破棄する（ブラックリストにない未知の
                # ハルシネーションパターンを検出するための保険）
                avg_logprob = getattr(seg, "avg_logprob", 0.0)
                if avg_logprob < AVG_LOGPROB_THRESHOLD:
                    print(f"[STT] avg_logprob={avg_logprob:.2f} のため破棄: {seg.text!r}")
                    continue

                # 圧縮率が異常に高い（同じような音節・単語の繰り返しが多い）セグメントも
                # ハルシネーションの典型パターンのため破棄する
                compression_ratio = getattr(seg, "compression_ratio", 0.0)
                if compression_ratio > COMPRESSION_RATIO_THRESHOLD:
                    print(f"[STT] compression_ratio={compression_ratio:.2f} のため破棄: {seg.text!r}")
                    continue

                text_parts.append(seg.text)

            text = "".join(text_parts).strip()

            if not text:
                return None

            if _looks_like_hallucination(text):
                print(f"[STT] 定型ハルシネーションフレーズと判定し破棄: {text!r}")
                return None

            return text
        except Exception as e:
            print(f"[STT] 文字起こしに失敗しました: {e}")
            return None