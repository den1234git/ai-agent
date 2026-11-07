"""
cleanup_broken_pages.py

G:\\ai_agent\\shobo-app\\src\\pages\\ に溜まった、過去の生成実験由来の壊れた
(tscで構文エラーになる)ファイルを一括で退避する。

削除はしない。_broken_pages_backup_{タイムスタンプ}\\ という新しいフォルダに
元のディレクトリ構造を保ったまま「移動」するだけなので、後で必要になれば
そこから戻せる。

router.pyの_typecheck_code()と同じ理屈で「-p tsconfig.json」でプロジェクト
全体をtscに通し、構文/型エラーが出たファイルの一覧を得て、それらを退避する。
一覧の抽出だけを目的にした簡易実装で、_typecheck_code()自体の実装には
手を入れない。

使い方:
    cd G:\\ai_agent\\server
    python cleanup_broken_pages.py           # まず一覧だけ表示(dry-run)
    python cleanup_broken_pages.py --apply   # 実際に退避を実行
"""
import os
import re
import sys
import shutil
import subprocess
from datetime import datetime

BASE = os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
TSCONFIG_PATH = os.path.join(BASE, "tsconfig.json")

ERROR_LINE_RE = re.compile(r"^(.*?\.tsx?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)$")


def run_tsc() -> str:
    if not os.path.exists(TSCONFIG_PATH):
        print(f"[ERROR] tsconfig.jsonが見つかりません: {TSCONFIG_PATH}")
        sys.exit(1)

    cmd = f'npx tsc --noEmit -p "{TSCONFIG_PATH}"'
    print(f"実行中: {cmd}\n(プロジェクト全体のチェックなので数十秒〜数分かかることがあります)")
    proc = subprocess.run(
        cmd,
        cwd=BASE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        shell=True,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def extract_broken_files(tsc_output: str) -> list:
    files = set()
    for line in tsc_output.splitlines():
        line = line.strip()
        m = ERROR_LINE_RE.match(line)
        if m:
            files.add(m.group(1).replace("\\", "/"))
    return sorted(files)


def main():
    apply_mode = "--apply" in sys.argv

    output = run_tsc()
    broken_files = extract_broken_files(output)

    if not broken_files:
        print("エラーのあるファイルは見つかりませんでした(プロジェクトはクリーンです)。")
        return

    print(f"\n=== エラーのあるファイル: {len(broken_files)}件 ===")
    for f in broken_files:
        print(f"  - {f}")

    if not apply_mode:
        print("\n(dry-runモードです。実際に退避するには --apply を付けて再実行してください)")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(BASE, f"_broken_pages_backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    moved = []
    skipped = []
    for rel_path in broken_files:
        src_path = os.path.join(BASE, rel_path.replace("/", os.sep))
        if not os.path.isfile(src_path):
            skipped.append((rel_path, "ファイルが見つかりません"))
            continue
        dst_path = os.path.join(backup_dir, rel_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.move(src_path, dst_path)
            moved.append(rel_path)
        except Exception as e:
            skipped.append((rel_path, str(e)))

    print(f"\n=== 退避完了 ===")
    print(f"退避先: {backup_dir}")
    print(f"移動: {len(moved)}件")
    for f in moved:
        print(f"  - {f}")
    if skipped:
        print(f"スキップ: {len(skipped)}件")
        for f, reason in skipped:
            print(f"  - {f} ({reason})")


if __name__ == "__main__":
    main()
