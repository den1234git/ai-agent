from learning import run_learning_cycle

result = run_learning_cycle(
    recent_conversation="",
    generation=None,
    theme="日本の消防法、火災予防条例、消火設備の法的基準"
)

print("=== 結果 ===")
print(f"総保存件数: {result['total_saved']}")
print(f"arXiv: {result['arxiv']['saved']}件")
print(f"Wikipedia: {result['wikipedia']['saved']}件")
print(f"e-Gov: {result['egov']['saved']}件")
print(f"Qiita: {result['qiita']['saved']}件")
print(f"ログ: {result['log_path']}")