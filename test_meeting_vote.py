"""
meeting_vote.pyの動作確認用スモークテスト。
本番は30回だが、まず軽く3回で配線が通るか確認する。
"""
from meeting_vote import convene_meeting

result = convene_meeting(
    "今日の点検報告書のタイトルは「点検結果報告書」と「点検報告書」どちらが適切か？",
    context="消防設備の法定点検で使う様式のタイトルを決めたい。",
    rounds=3,
)

print("=== 結果 ===")
print(f"decided: {result['decided']}")
print(f"verdict: {result['verdict']}")
print(f"vote_count/total_votes: {result['vote_count']}/{result['total_votes']}")
print(f"needs_human: {result['needs_human']}")
print(f"log_path: {result['log_path']}")
print()
print("=== 生の意見 ===")
for o in result["opinions"]:
    print(f"- [{o['role']}] voted={o['voted']} opinion={o['opinion']!r} reason={o['reason']!r}")
