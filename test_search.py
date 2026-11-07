from memory_store import search_shobo_knowledge

results = search_shobo_knowledge("消防法の目的について教えて")
for r in results:
    print(r["title"], "-", r["distance"])
    print(r["content"][:100])
    print()