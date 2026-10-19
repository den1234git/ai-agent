"""
honeypot_sync.py — ハニーポットログをローカルに同期する

毎朝 03:00 に village_daemon.py から実行される。

動作:
  1. Worker KV の hp_* キーを GET /api/admin/honeypot で取得（管理者認証必須）
  2. 未記録の新規エントリのみ honeypot_log.jsonl に追記（追記専用、上書きなし）
  3. 同期済みキー一覧を honeypot_sync_state.json に保存（重複防止）

【記録の分離構造】
  Cloudflare KV（hp_* キー）  ← 一時ステージング（90日で自然消滅）
      ↓ 毎朝03:00 sync
  ローカル honeypot_log.jsonl  ← 永続的な安全コピー（この機械上のみ）

KV が攻撃者に消されてもローカルコピーは残る。
ローカルに到達するには Cloudflare とは別の攻撃が必要になる。
"""

import datetime
import hashlib
import json
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8")

_BASE       = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(_BASE, "honeypot_log.jsonl")
STATE_FILE  = os.path.join(_BASE, "honeypot_sync_state.json")
API_BASE    = "https://fnoita-board.mogumogu7la001.workers.dev"


def _admin_token() -> str:
    secret = os.environ.get("BOARD_ADMIN_SECRET", "")
    if not secret:
        raise RuntimeError("BOARD_ADMIN_SECRET が設定されていません")
    secret_hash = hashlib.sha256(secret.encode()).hexdigest()
    ymd = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d")
    return hashlib.sha256((secret_hash + ymd).encode()).hexdigest()


def load_state() -> set:
    """同期済みキーの集合を返す。"""
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return set(json.load(f).get("synced_keys", []))
    except Exception:
        return set()


def save_state(synced_keys: set):
    """同期済みキーを保存する。"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"synced_keys": sorted(synced_keys)}, f, ensure_ascii=False)


def fetch_honeypot_logs() -> list:
    """Worker KV から hp_* ログを取得する。"""
    token = _admin_token()
    r = requests.get(
        f"{API_BASE}/api/admin/honeypot",
        headers={"X-Admin-Token": token},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"取得失敗: {r.status_code} {r.text[:100]}")
    return r.json().get("entries", [])


def append_to_log(entries: list):
    """新規エントリを追記専用で JSONL ファイルに書き込む。"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def run():
    print(f"[HP_SYNC] 開始 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    synced_keys = load_state()
    entries = fetch_honeypot_logs()
    print(f"[HP_SYNC] KV から {len(entries)} 件取得")

    new_entries = [e for e in entries if e.get("key") not in synced_keys]
    print(f"[HP_SYNC] 新規 {len(new_entries)} 件を追記")

    if new_entries:
        append_to_log(new_entries)
        for e in new_entries:
            synced_keys.add(e["key"])
        save_state(synced_keys)
        print(f"[HP_SYNC] honeypot_log.jsonl に追記完了")
    else:
        print(f"[HP_SYNC] 新規なし（スキップ）")

    print(f"[HP_SYNC] 完了 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    run()
