"""
shobo_knowledge / medical_knowledge の全レコードを
Open Knowledge Format (OKF) v0.2 (https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
準拠のMarkdownバンドルとしてエクスポートする。

出力先:
  G:\\ai_agent\\okf_export\\shobo\\    （shobo_knowledge由来）
  G:\\ai_agent\\okf_export\\medical\\  （medical_knowledge由来）

方針（2026-07-25 合意）: OKFのコア5フィールドのみを使う。
trust/provenance/lifecycle等の拡張ファミリー（generated, verified, sources等）は使わない。
  - type       : "Law Article"（e-govの法令データ） / "Teacher Correction"（/teach/correctでの訂正）
                 / "Knowledge Note"（それ以外、/learnで取り込んだ知識など）
  - title      : メタデータのtitle（なければdoc_id）
  - description: 本文冒頭を1文程度（140文字）に切り出したもの
  - resource   : sourceがURLならそれをそのまま使う。それ以外は元レコードを指す内部URI
                 （chroma://<collection>/<doc_id>）
  - tags       : [category, source]（sourceがURLの場合はtagsに入れない。resourceで表現済みのため）

ディレクトリ構成（OKFのbundle構造・SPEC.md §3に準拠）:
  okf_export/shobo/
    index.md                  # カテゴリ一覧
    <category>/
      index.md                # そのカテゴリの概念一覧
      <doc_id>.md ...
  okf_export/medical/
    （同様）

再実行について: ファイル名はChromaDBのdoc_idに基づく1対1対応なので、
再実行すれば既存ファイルは上書き更新、新規レコードは新規ファイルとして
追加される（idempotentなフルエクスポート）。
"""

import os
import re
import chromadb
from chromadb.config import Settings
from datetime import datetime, timezone

DB_DIR = r"F:\\ai_agent\server\chroma_db"
EXPORT_ROOT = r"F:\\ai_agent\okf_export"

DESCRIPTION_MAX_LEN = 140
DIRNAME_MAX_LEN = 80


def _sanitize_dirname(name: str) -> str:
    name = (name or "").strip() or "uncategorized"
    # Windowsのファイル名で使えない文字を置換する
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:DIRNAME_MAX_LEN]


def _make_description(content: str) -> str:
    one_line = " ".join((content or "").split())
    if len(one_line) > DESCRIPTION_MAX_LEN:
        return one_line[:DESCRIPTION_MAX_LEN].rstrip() + "…"
    return one_line


def _make_type(source: str, category: str) -> str:
    if category == "correction":
        return "Teacher Correction"
    if source in ("e-gov", "egov"):
        return "Law Article"
    return "Knowledge Note"


def _make_resource(collection_name: str, doc_id: str, source: str) -> str:
    if source and source.startswith("http"):
        return source
    return f"chroma://{collection_name}/{doc_id}"


def _yaml_dquote(value: str) -> str:
    """YAMLのダブルクォート文字列としてエスケープする（改行は事前に除去しておくこと）"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_concept_file(filepath, type_, title, description, resource, tags, content, footer_note):
    title = (title or "").replace("\n", " ").strip()
    lines = ["---", f"type: {_yaml_dquote(type_)}"]
    if title:
        lines.append(f"title: {_yaml_dquote(title)}")
    if description:
        lines.append(f"description: {_yaml_dquote(description)}")
    if resource:
        lines.append(f"resource: {_yaml_dquote(resource)}")
    if tags:
        tag_str = ", ".join(_yaml_dquote(t) for t in tags)
        lines.append(f"tags: [{tag_str}]")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title or os.path.splitext(os.path.basename(filepath))[0]}")
    lines.append("")
    lines.append((content or "").strip())
    if footer_note:
        lines.append("")
        lines.append("---")
        lines.append(f"*{footer_note}*")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_collection(client, collection_name: str, out_dir: str) -> dict:
    col = client.get_collection(collection_name)
    count = col.count()
    print(f"[OKF] {collection_name}: {count}件をエクスポートします")
    if count == 0:
        return {"exported": 0, "categories": {}}

    all_data = col.get(include=["metadatas", "documents"])
    os.makedirs(out_dir, exist_ok=True)

    category_files = {}  # サニタイズ後のディレクトリ名 -> [(doc_id, title)]

    for doc_id, meta, content in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        category = meta.get("category", "") or ""
        source = meta.get("source", "") or ""
        title = meta.get("title", "") or doc_id
        timestamp = meta.get("timestamp")

        ts_str = ""
        if timestamp:
            try:
                ts_str = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                ts_str = ""

        type_ = _make_type(source, category)
        description = _make_description(content)
        resource = _make_resource(collection_name, doc_id, source)
        tags = [t for t in [category, (source if source and not source.startswith("http") else "")] if t]

        dirname = _sanitize_dirname(category or "uncategorized")
        cat_dir = os.path.join(out_dir, dirname)
        os.makedirs(cat_dir, exist_ok=True)

        footer_note = f"chroma_id: {doc_id}"
        if ts_str:
            footer_note += f" / saved_at: {ts_str}"

        filepath = os.path.join(cat_dir, f"{doc_id}.md")
        _write_concept_file(filepath, type_, title, description, resource, tags, content, footer_note)

        category_files.setdefault(dirname, []).append((doc_id, title.replace("\n", " ").strip() or doc_id))

    # カテゴリごとのindex.md
    for dirname, items in category_files.items():
        cat_dir = os.path.join(out_dir, dirname)
        lines = [f"# {dirname}", ""]
        for doc_id, title in items:
            safe_title = title.replace("[", "(").replace("]", ")")
            lines.append(f"* [{safe_title}]({doc_id}.md)")
        with open(os.path.join(cat_dir, "index.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ルートのindex.md
    root_lines = [f"# {collection_name} OKF bundle", "", f"ChromaDBの`{collection_name}`コレクションから生成。", ""]
    for dirname in sorted(category_files.keys()):
        root_lines.append(f"* [{dirname}]({dirname}/index.md) - {len(category_files[dirname])}件")
    with open(os.path.join(out_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(root_lines) + "\n")

    return {"exported": count, "categories": {k: len(v) for k, v in category_files.items()}}


def main():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    shobo_result = export_collection(client, "shobo_knowledge", os.path.join(EXPORT_ROOT, "shobo"))
    medical_result = export_collection(client, "medical_knowledge", os.path.join(EXPORT_ROOT, "medical"))

    print("\n=== エクスポート結果 ===")
    print(f"shobo: {shobo_result['exported']}件")
    for cat, n in sorted(shobo_result["categories"].items()):
        print(f"   {cat}: {n}件")
    print(f"medical: {medical_result['exported']}件")
    for cat, n in sorted(medical_result["categories"].items()):
        print(f"   {cat}: {n}件")


if __name__ == "__main__":
    main()
