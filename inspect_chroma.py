"""
ChromaDBに保存されている内容を調査するスクリプト。

対象コレクション：
  - la_experience    … コード生成の経験（＝プログラム関連）
  - shobo_knowledge  … 消防法令 + AI学習で取り込んだ知識（＝火災関連が中心。
                        ただし on_the_horizon の項目6で「React TypeScriptテーマに
                        切り替えるか検討中」とあるので、混在している可能性がある）

やること：
  1. ChromaDBの永続化ディレクトリを自動探索（見つからなければ手動指定を促す）
  2. 各コレクションの件数（count）を表示
  3. メタデータのキー一覧（＝「テーブルのカラム名」に相当するもの）を集計
  4. メタデータ中の分類っぽいフィールド（source / theme / law_name / category など）
     があれば、その値ごとに件数を集計 → プログラム関連 / 火災関連の内訳が見える
  5. 各コレクションからサンプルを数件、内容つきで表示

実行方法：
    cd F:\\ai_agent\server
    pip install chromadb   （導入済みなら不要）
    python inspect_chroma.py

もし「パスが見つかりません」と出た場合は、下の CANDIDATE_PATHS に
実際のパスを追加するか、まさの環境で
    findstr /s /i "PersistentClient" *.py
を実行して初期化箇所を探し、そのパスを直接 CHROMA_PATH に指定してください。
"""

import os
import sys
import json
from collections import Counter

# ---------------------------------------------------------------------------
# 1. パスの自動探索
# ---------------------------------------------------------------------------

CANDIDATE_PATHS = [
    r"F:\\ai_agent\server\chroma_db",
    r"F:\\ai_agent\server\chromadb",
    r"F:\\ai_agent\server\.chroma",
    r"F:\\ai_agent\chroma_db",
    r"F:\\ai_agent\.cache\chroma",
]

# 分類の手がかりになりそうなメタデータキー候補
CATEGORY_FIELD_CANDIDATES = [
    "source", "type", "category", "theme", "law_name",
    "origin", "topic", "domain", "kind",
]

SAMPLE_SIZE = 5


def find_chroma_path():
    for p in CANDIDATE_PATHS:
        if os.path.isdir(p) and os.listdir(p):
            return p
    return None


def load_client(path):
    import chromadb
    return chromadb.PersistentClient(path=path)


# ---------------------------------------------------------------------------
# 2〜5. 集計・表示
# ---------------------------------------------------------------------------

def inspect_collection(client, name):
    print("=" * 70)
    print(f"■ コレクション: {name}")
    print("=" * 70)

    try:
        col = client.get_collection(name)
    except Exception as e:
        print(f"  → 取得失敗: {e}")
        return

    count = col.count()
    print(f"件数: {count}")

    if count == 0:
        print("  (データなし)")
        return

    data = col.get(include=["metadatas", "documents"])
    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    # --- メタデータのキー一覧（＝カラム名相当） ---
    key_counter = Counter()
    for m in metadatas:
        if m:
            key_counter.update(m.keys())

    print("\n[メタデータのキー一覧（出現回数）]")
    if key_counter:
        for k, v in key_counter.most_common():
            print(f"  - {k}: {v}件")
    else:
        print("  (メタデータなし)")

    # --- 分類フィールドごとの内訳 ---
    print("\n[分類フィールドごとの件数内訳]")
    found_any_category = False
    for field in CATEGORY_FIELD_CANDIDATES:
        if field not in key_counter:
            continue
        found_any_category = True
        value_counter = Counter()
        for m in metadatas:
            if m and field in m:
                value_counter[str(m[field])] += 1
        print(f"  ● {field} 別:")
        for val, cnt in value_counter.most_common():
            print(f"      {val}: {cnt}件")
    if not found_any_category:
        print("  (source/theme/law_name等の分類フィールドは見つかりませんでした。"
              "  documentの中身から手動で判断してください)")

    # --- サンプル表示 ---
    print(f"\n[サンプル（先頭{min(SAMPLE_SIZE, count)}件）]")
    for i in range(min(SAMPLE_SIZE, count)):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc = documents[i] if i < len(documents) else ""
        doc_preview = (doc[:150] + "…") if doc and len(doc) > 150 else doc
        print(f"  --- #{i+1} ---")
        print(f"  metadata: {json.dumps(meta, ensure_ascii=False)}")
        print(f"  content : {doc_preview}")
    print()


def main():
    chroma_path = find_chroma_path()
    if not chroma_path:
        print("ChromaDBの永続化ディレクトリが自動では見つかりませんでした。")
        print("このファイル冒頭の CANDIDATE_PATHS に実際のパスを追加するか、")
        print("下記コマンドで初期化コードを探して、パスを確認してください：")
        print(r'    findstr /s /i "PersistentClient" *.py')
        sys.exit(1)

    print(f"ChromaDBパス: {chroma_path}\n")
    client = load_client(chroma_path)

    all_collections = [c.name for c in client.list_collections()]
    print(f"存在するコレクション一覧: {all_collections}\n")

    for name in ["la_experience", "shobo_knowledge", "medical_knowledge", "village_knowledge"]:
        if name in all_collections:
            inspect_collection(client, name)
        else:
            print(f"※ コレクション '{name}' が見つかりませんでした（{all_collections} のみ存在）\n")


if __name__ == "__main__":
    main()