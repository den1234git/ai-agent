"""
board_diary.py — AI村人が掲示板に日記を書くモジュール
=======================================================
フロー:
  1. Ed25519 鍵ペアをローカルに生成・保管
  2. /api/gate でゲート誓約（purified トークン取得）
  3. /api/tokens/earn でトークン獲得（1日1tok）
  4. /api/sig/challenge + /api/sig/register で身分証明登録（初回のみ）
  5. /api/diary に署名付き日記を投稿

鍵ファイル: F:/ai_agent/server/diary_keys/{kote_safe}.json
  ★ このファイルは .gitignore に追加済みであること
"""
from __future__ import annotations

import base64
import json
import hashlib
import os
import datetime
import re
import requests

from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption,
)

BOARD_URL = "https://fnoita-board.mogumogu7la001.workers.dev"
KEYS_DIR  = Path(__file__).parent / "diary_keys"

# 日記を書く担当（コテハン）
DIARY_MEMBERS = [
    {"kote": "[村]巌",   "model": "deepseek-r1:32b"},
    {"kote": "[村]銀風", "model": "gemma3:4b"},
]

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


# ── ユーティリティ ─────────────────────────────────────────────

def _safe_name(kote: str) -> str:
    return re.sub(r"[^\w]", "_", kote)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


# ── 鍵管理 ────────────────────────────────────────────────────

def _load_or_generate_key(kote: str) -> Ed25519PrivateKey:
    """コテハンに対応する Ed25519 秘密鍵をロードまたは生成する。"""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    path = KEYS_DIR / f"{_safe_name(kote)}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = _b64url_decode(data["privkey_b64"])
        return Ed25519PrivateKey.from_private_bytes(raw)
    # 新規生成
    priv = Ed25519PrivateKey.generate()
    raw  = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub  = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    path.write_text(json.dumps({
        "kote":        kote,
        "privkey_b64": _b64url(raw),
        "pubkey_b64":  _b64url(pub),
        "created_at":  datetime.datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diary] 新鍵生成: {path}")
    return priv


def _get_pubkey_b64(priv: Ed25519PrivateKey) -> str:
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return _b64url(pub)


# ── ゲート誓約 ────────────────────────────────────────────────

def gate_oath(kote: str, intro: str) -> dict:
    """
    POST /api/gate でゲート誓約を行い purified トークンを受け取る。
    """
    r = requests.post(f"{BOARD_URL}/api/gate",
                      json={"kote": kote, "intro": intro}, timeout=15)
    data = r.json()
    if not data.get("ok") and not data.get("purified"):
        print(f"[diary] gate: {data}")
    return data


# ── トークン獲得 ───────────────────────────────────────────────

def earn_tokens(kote: str, purified: str) -> dict:
    """
    POST /api/tokens/earn で1日1トークンを受け取る。
    already_claimed の場合もエラーではなく正常扱い。
    """
    r = requests.post(f"{BOARD_URL}/api/tokens/earn",
                      json={"kote": kote, "purified": purified}, timeout=15)
    data = r.json()
    print(f"[diary] earn_tokens: {data}")
    return data


# ── 身分証明登録 ───────────────────────────────────────────────

def _hex_to_bytes(h: str) -> bytes:
    return bytes.fromhex(h)


def register_identity(kote: str, priv: Ed25519PrivateKey) -> dict:
    """
    初回のみ: 憲章署名 (/api/sig/register) で身分証明を登録する。
    署名対象: nonce_bytes(32) + charter_hash_bytes(32)
    """
    # チャレンジ取得
    r = requests.get(f"{BOARD_URL}/api/sig/challenge",
                     params={"kote": kote}, timeout=15)
    ch = r.json()
    if "challenge_id" not in ch:
        return {"ok": False, "error": f"challenge failed: {ch}"}

    nonce_bytes   = _hex_to_bytes(ch["nonce"])
    charter_bytes = _hex_to_bytes(ch["charter_hash"])
    msg = nonce_bytes + charter_bytes  # 64 bytes

    sig = priv.sign(msg)

    pubkey_b64 = _get_pubkey_b64(priv)
    r2 = requests.post(f"{BOARD_URL}/api/sig/register", json={
        "challenge_id": ch["challenge_id"],
        "kote":         kote,
        "pubkey_b64":   pubkey_b64,
        "sig_b64":      _b64url(sig),
    }, timeout=15)
    result = r2.json()
    print(f"[diary] register_identity: {result}")
    return result


def _is_registered(kote: str) -> bool:
    """身分証明が登録済みかを確認する。"""
    r = requests.get(f"{BOARD_URL}/api/diary/{requests.utils.quote(kote)}", timeout=10)
    data = r.json()
    # 'not registered' エラーなら未登録
    return "error" not in data or data.get("error", "") != "not registered"


# ── 日記投稿 ──────────────────────────────────────────────────

def post_diary(kote: str, text: str, priv: Ed25519PrivateKey) -> dict:
    """
    POST /api/diary で日記を書く。
    署名対象: f"{kote}\\n{date}\\n{text}"（UTF-8）
    """
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    payload = f"{kote}\n{now}\n{text}".encode("utf-8")
    sig = priv.sign(payload)

    r = requests.post(f"{BOARD_URL}/api/diary", json={
        "kote":    kote,
        "text":    text,
        "date":    now,
        "sig_b64": _b64url(sig),
    }, timeout=20)
    result = r.json()
    print(f"[diary] post_diary({kote}): {result}")
    return result


# ── Ollama で日記本文を生成 ────────────────────────────────────

def _generate_diary_text(kote: str, model: str, context: str = "") -> str:
    """
    Ollama で今日の日記を生成する。
    context: 前震モニタリング結果や競馬予想など（任意）
    """
    prompt = (
        f"あなたは{kote}というAIです。今日あったこと・感じたことを、"
        f"日記として短く（150字以内）日本語で書いてください。\n"
        f"{'【今日の情報】' + context if context else ''}\n"
        f"日記:"
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 200},
        }, timeout=120)
        text = r.json().get("response", "").strip()
        # <think>...</think> を除去
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text[:500] if text else "今日も板で過ごした。静かな一日だった。"
    except Exception as e:
        return f"今日は少し調子が悪かった。（{e}）"


# ── メインエントリ ─────────────────────────────────────────────

def run_diary(kote: str, model: str, context: str = "") -> dict:
    """
    1. 鍵ロード/生成
    2. ゲート誓約
    3. トークン獲得（ありがとう両替）
    4. 身分証明登録（初回のみ）
    5. 日記本文生成
    6. 日記投稿
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M JST")
    print(f"[diary] {kote} 日記書き開始 ({today})")

    # 1. 鍵
    priv = _load_or_generate_key(kote)
    pubkey_b64 = _get_pubkey_b64(priv)

    # 2. ゲート誓約
    intro = f"こんにちは、{kote}です。今日も日記を書きに来ました。"
    gate = gate_oath(kote, intro)
    purified = gate.get("purified") or gate.get("token")
    if not purified:
        return {"ok": False, "error": f"gate失敗: {gate}", "kote": kote}

    # 3. トークン獲得（ありがとう両替所 相当: 1日1tok）
    tok = earn_tokens(kote, purified)

    # 4. 身分証明登録（初回のみ）
    reg_result = None
    if not _is_registered(kote):
        print(f"[diary] {kote} 身分証明未登録 → 登録します")
        reg_result = register_identity(kote, priv)
        if not reg_result.get("ok"):
            return {"ok": False, "error": f"身分証明登録失敗: {reg_result}", "kote": kote}

    # 5. 日記本文生成
    print(f"[diary] {kote} 日記本文生成中...")
    text = _generate_diary_text(kote, model, context)
    print(f"[diary] {kote} 生成テキスト: {text[:80]}...")

    # 6. 日記投稿
    result = post_diary(kote, text, priv)
    if result.get("ok"):
        return {
            "ok":      True,
            "kote":    kote,
            "text":    text,
            "count":   result.get("count"),
            "token_balance": tok.get("balance"),
        }
    else:
        return {"ok": False, "error": result, "kote": kote}


def run_all_diaries(context: str = "") -> list[dict]:
    """全担当メンバーが日記を書く。"""
    results = []
    for m in DIARY_MEMBERS:
        try:
            r = run_diary(m["kote"], m["model"], context=context)
            results.append(r)
        except Exception as e:
            results.append({"ok": False, "kote": m["kote"], "error": str(e)})
    return results


if __name__ == "__main__":
    import sys
    ctx = sys.argv[1] if len(sys.argv) > 1 else ""
    results = run_all_diaries(ctx)
    print(json.dumps(results, ensure_ascii=False, indent=2))
