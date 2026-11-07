import memory_store as ms, re
from pathlib import Path

col = ms._get_knowledge_col()
data = col.get(include=['metadatas','documents'])
ids, metas, docs = data['ids'], data['metadatas'], data['documents']

out_dir = Path(r"F:\\ai_agent\_c2s_sample100")
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.glob("*.txt"):
    f.unlink()

total = len(ids)
stride = max(total // 100, 1)
count = 0
for i in range(0, total, stride):
    if count >= 100:
        break
    meta = metas[i] or {}
    title = meta.get('title') or ''
    doc = docs[i] or ''
    text = f"{title}\n\n{doc}".strip()
    if not text:
        continue
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', ids[i])[:80] or f"doc{count}"
    (out_dir / f"{safe}.txt").write_text(text, encoding='utf-8')
    count += 1

print(f"wrote {count} to {out_dir}")