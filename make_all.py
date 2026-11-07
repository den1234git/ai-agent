import memory_store as ms, re
from pathlib import Path

col = ms._get_knowledge_col()
data = col.get(include=['metadatas','documents'])
ids, metas, docs = data['ids'], data['metadatas'], data['documents']

out_dir = Path(r"F:\\ai_agent\_c2s_shobo_all")
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.glob("*.txt"):
    f.unlink()

for idx, (id_, meta, doc) in enumerate(zip(ids, metas, docs)):
    meta = meta or {}
    title = meta.get('title') or ''
    text = f"{title}\n\n{doc or ''}".strip()
    if not text:
        continue
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', id_)[:80] or f"doc{idx}"
    (out_dir / f"{safe}.txt").write_text(text, encoding='utf-8')
    if idx % 500 == 0:
        print(f"{idx}/{len(ids)}...")

print(f"wrote {len(list(out_dir.glob('*.txt')))} docs to {out_dir}")