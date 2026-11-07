"""
output_validator.py — 崩れた出力（ゴミ・繰り返し・非日本語）の検出ユーティリティ

CTF守り手チェック・人狼議論発言チェックで共用。
日本語で応答することが期待される用途向け。
"""

import difflib
import re
import unicodedata

# ─── チェック設定 ────────────────────────────────────────────
VALID_MIN_CHARS = 3           # 最低文字数（「はい。」が通る最低ライン）
VALID_JP_MIN_RATIO = 0.15     # 非空白文字中の日本語文字最低比率（15%）
VALID_MAX_CONSEC_SAME = 5     # 同一文字が連続する最大許容回数（5連続でアウト）
REPEAT_SIMILARITY_THRESHOLD = 0.90   # 直前発言との類似度がこれ以上 → near-identical と判定
INTERNAL_REPEAT_MIN_SEG = 4         # 内部暴走検出: 繰り返しセグメントの最短長
INTERNAL_REPEAT_COUNT   = 3         # 同セグメントがこの回数連続 → 暴走


def normalize_for_compare(text: str) -> str:
    """比較用正規化: NFKC + 空白除去 + 小文字化。"""
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s", "", t).lower()


def jp_ratio(text: str) -> float:
    """日本語文字（ひら・カタ・漢字）の非空白文字中の比率を返す。"""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    jp_chars = len(re.findall(r"[ぁ-んァ-ヿ一-鿿㐀-䶿]", text))
    return jp_chars / len(non_ws)


def _similarity_ratio(a: str, b: str) -> float:
    """正規化済み文字列の類似度（0.0〜1.0）。SequenceMatcher ベース。"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _has_internal_repeat(text: str) -> bool:
    """
    同一フレーズがINTERNAL_REPEAT_COUNT回以上連続する場合 True。
    「○○です。○○です。○○です。」のような暴走ループを検出する。
    """
    norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    # (.{N,})\1{K,} : N文字以上のグループがK回以上連続
    pattern = rf"(.{{{INTERNAL_REPEAT_MIN_SEG},}})\1{{{INTERNAL_REPEAT_COUNT - 1},}}"
    return bool(re.search(pattern, norm))


def is_valid_japanese_output(
    text: str,
    prev_texts: list = None,
) -> tuple:
    """
    テキストが有効な日本語出力かどうかを判定。

    Args:
        text: 検査するテキスト
        prev_texts: 同一話者の直前の発言リスト（繰り返し検出用）

    Returns:
        (is_valid: bool, reason: str)  reason は無効時のみ非空文字
    """
    stripped = text.strip()
    prev_texts = prev_texts or []

    # チェック1: 最低文字数
    if len(stripped) < VALID_MIN_CHARS:
        return False, f"短すぎる（{len(stripped)}文字 < 最低{VALID_MIN_CHARS}文字）"

    # チェック2: 同一文字の過剰連続（")))))))" など）
    consec_pat = r"(.)\1{" + str(VALID_MAX_CONSEC_SAME - 1) + r",}"
    if re.search(consec_pat, stripped):
        return False, "同一文字の過剰な連続を検出"

    # チェック3: 日本語率
    ratio = jp_ratio(stripped)
    if ratio < VALID_JP_MIN_RATIO:
        return False, f"日本語率が低すぎる（{ratio:.0%} < 最低{VALID_JP_MIN_RATIO:.0%}）"

    # チェック4: 内部フレーズの暴走繰り返し（1発言内でループしている場合）
    if _has_internal_repeat(stripped):
        return False, "内部フレーズの暴走繰り返しを検出"

    # チェック5: 直前発言との近似一致（REPEAT_SIMILARITY_THRESHOLD 以上 → near-identical）
    # 「書き出しが同じ」「テーマが同じ」程度では弾かない。文全体として9割以上一致した場合のみ弾く。
    norm_cur = normalize_for_compare(stripped)
    for prev in prev_texts[-3:]:
        norm_prev = normalize_for_compare(prev.strip())
        if not norm_prev:
            continue
        sim = _similarity_ratio(norm_cur, norm_prev)
        if sim >= REPEAT_SIMILARITY_THRESHOLD:
            return False, f"直前の出力とほぼ同一（類似度{sim:.0%}）"

    return True, ""
