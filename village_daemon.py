"""
village_daemon.py — Windowsサービスとして常駐するスケジューラー

毎日 03:00 → honeypot_sync.py（ハニーポットログをローカルに同期）
毎日 09:00 → village_study.py
毎日 12:00 → board_members_post（掲示板出動 1回目）
毎日 18:00 → board_members_post（掲示板出動 2回目）
毎日 21:00 → board_news_cycle.py
"""

import subprocess
import sys
import time
import datetime
import os
import logging

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

LOG_FILE = os.path.join(BASE, "village_daemon.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

BOARD_MEMBERS_SCRIPT = r"F:\掲示板専用\village_board_members.py"

SCHEDULE = [
    {"hour":  3, "minute": 0, "script": "honeypot_sync.py",         "key": "hp_sync"},
    {"hour":  9, "minute": 0, "script": "village_study.py",         "key": "study"},
    {"hour": 12, "minute": 0, "script": BOARD_MEMBERS_SCRIPT,       "key": "board_am",  "abs": True},
    {"hour": 18, "minute": 0, "script": BOARD_MEMBERS_SCRIPT,       "key": "board_pm",  "abs": True},
    {"hour": 21, "minute": 0, "script": "board_news_cycle.py",      "key": "news"},
]

def run_script(script: str, abs_path: bool = False):
    path = script if abs_path else os.path.join(BASE, script)
    logging.info(f"START {script}")
    try:
        result = subprocess.run(
            [PYTHON, path],
            cwd=BASE,
            timeout=3600,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logging.info(f"END {script} (rc={result.returncode})")
        if result.stdout:
            logging.info(result.stdout[-2000:])
        if result.stderr:
            logging.warning(result.stderr[-500:])
    except subprocess.TimeoutExpired:
        logging.error(f"TIMEOUT {script}")
    except Exception as e:
        logging.error(f"ERROR {script}: {e}")


def main():
    logging.info("village_daemon 起動")
    ran_today = {}  # key → date

    while True:
        now = datetime.datetime.now()
        today = now.date()

        for job in SCHEDULE:
            key = job["key"]
            if (now.hour == job["hour"] and now.minute == job["minute"]
                    and ran_today.get(key) != today):
                ran_today[key] = today
                run_script(job["script"], abs_path=job.get("abs", False))

        time.sleep(30)  # 30秒ごとに時刻チェック


if __name__ == "__main__":
    main()
