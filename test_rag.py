from ai import ask_ai_with_rag

result = ask_ai_with_rag(
    prompt="消防法における自動火災報知設備の設置基準について教えてください",
    model="swallow",
    knowledge_query="消防法 自動火災報知設備"
)

print("=== 結果 ===")
print(result)