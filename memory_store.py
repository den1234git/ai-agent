"""
ChromaDBを使ったLAの長期記憶モジュール。

2つのコレクションを管理する：
  - la_experience  : LAが生成したコード・タスク・結果・感想を蓄積
  - shobo_knowledge: 消防法・設備知識・点検基準などの業務知識を蓄積

埋め込みはChromaDBのデフォルト（all-MiniLM-L6-v2）を使用。
CPU-onlyで動作し、Ollamaとは独立している。
初回起動時に約90MBのモデルが自動ダウンロードされる。

DBの保存先: D:/ai_agent/server/chroma_db/
"""

import os
import json
import time
import chromadb
from chromadb.config import Settings

# ★ キャッシュをDドライブに向ける（ここを追加）
os.environ.setdefault("CHROMA_CACHE_DIR", r"D:\ai_agent\.cache\chroma")

# DBの保存先
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 検索時に返す類似件数
N_EXPERIENCE_RESULTS = 3
N_KNOWLEDGE_RESULTS  = 2

_client = None
_exp_col = None
_know_col = None


def _get_client():
    global _client
    if _client is None:
        # onnxモデルのキャッシュ先を明示的にDドライブに指定
        os.environ["ONNX_MODELS_PATH"] = r"D:\ai_agent\.cache\chroma\onnx_models"
        os.makedirs(r"D:\ai_agent\.cache\chroma\onnx_models", exist_ok=True)
        _client = chromadb.PersistentClient(
            path=DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                chroma_cache_dir=r"D:\ai_agent\.cache\chroma",
            ),
        )
    return _client


def _get_experience_col():
    global _exp_col
    if _exp_col is None:
        _exp_col = _get_client().get_or_create_collection("la_experience")
    return _exp_col


def _get_knowledge_col():
    global _know_col
    if _know_col is None:
        _know_col = _get_client().get_or_create_collection("shobo_knowledge")
    return _know_col


# ─────────────────────────────────────────
# la_experience：生成経験の保存・検索
# ─────────────────────────────────────────

def save_experience(
    task: str,
    code: str,
    outcome: str,           # "success" | "failure"
    learned: str = "",      # LAが学んだこと（swallowに生成させる）
    reflection: str = "",   # LAの感想（swallowに生成させる）
    file_path: str = "",
) -> None:
    """
    1回の生成経験をla_experienceコレクションに保存する。
    検索キー（document）はtaskとlearnedの組み合わせにする。
    """
    col = _get_experience_col()
    doc_id = f"exp_{int(time.time() * 1000)}"

    # documentはベクトル化される文字列。taskを主軸にlearnedも含める
    document = f"{task}\n{learned}".strip()

    metadata = {
        "task": task[:500],          # 長すぎる場合は切り詰め
        "outcome": outcome,
        "learned": learned[:500],
        "reflection": reflection[:500],
        "file_path": file_path,
        "timestamp": int(time.time()),
    }

    # codeは大きくなるのでmetadataには入れずjsonで別保存
    # （ChromaDBのmetadata値は512文字制限があるため）
    col.add(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata],
    )
    print(f"[MEMORY] 経験を保存しました: {doc_id} outcome={outcome}")


def search_experience(query: str, n: int = N_EXPERIENCE_RESULTS) -> list[dict]:
    """
    過去の類似経験をベクトル検索して返す。
    戻り値: [{"task": ..., "outcome": ..., "learned": ..., "reflection": ...}, ...]
    """
    col = _get_experience_col()
    try:
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        items = []
        for i, meta in enumerate(results["metadatas"][0]):
            items.append({
                "task": meta.get("task", ""),
                "outcome": meta.get("outcome", ""),
                "learned": meta.get("learned", ""),
                "reflection": meta.get("reflection", ""),
                "file_path": meta.get("file_path", ""),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return items
    except Exception as e:
        print(f"[MEMORY] 経験の検索に失敗しました（無視して続行）: {e}")
        return []


# ─────────────────────────────────────────
# shobo_knowledge：消防知識の保存・検索
# ─────────────────────────────────────────

def save_shobo_knowledge(
    content: str,
    title: str = "",
    source: str = "",       # "manual" | "web" | "pdf"
    category: str = "",     # "法令" | "設備" | "点検基準" など
) -> None:
    """
    消防に関する知識テキストをshobo_knowledgeコレクションに保存する。
    同じtitleが既にある場合は上書きする。
    """
    col = _get_knowledge_col()
    doc_id = f"know_{abs(hash(title or content[:50]))}"

    metadata = {
        "title": title[:200],
        "source": source,
        "category": category,
        "timestamp": int(time.time()),
    }

    # 既存IDがあればupsert（update or insert）
    try:
        col.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
        )
        print(f"[MEMORY] 消防知識を保存しました: {title or doc_id}")
    except Exception as e:
        print(f"[MEMORY] 消防知識の保存に失敗しました: {e}")


def search_shobo_knowledge(query: str, n: int = N_KNOWLEDGE_RESULTS) -> list[dict]:
    """
    消防知識をベクトル検索して返す。
    戻り値: [{"title": ..., "content": ..., "category": ...}, ...]
    """
    col = _get_knowledge_col()
    try:
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        items = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            items.append({
                "title": meta.get("title", ""),
                "content": doc,
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return items
    except Exception as e:
        print(f"[MEMORY] 消防知識の検索に失敗しました（無視して続行）: {e}")
        return []


# ─────────────────────────────────────────
# 統計・管理
# ─────────────────────────────────────────

def get_stats() -> dict:
    """DBの件数を返す（UI表示・デバッグ用）"""
    try:
        return {
            "la_experience": _get_experience_col().count(),
            "shobo_knowledge": _get_knowledge_col().count(),
        }
    except Exception as e:
        return {"error": str(e)}


def format_experience_context(experiences: list[dict]) -> str:
    """
    search_experience()の結果をプロンプトに差し込める文字列に整形する。
    """
    if not experiences:
        return ""
    lines = ["【過去の類似経験】"]
    for i, exp in enumerate(experiences, 1):
        outcome_label = "✓ 成功" if exp["outcome"] == "success" else "✗ 失敗"
        lines.append(f"{i}. {outcome_label} | {exp['task'][:80]}")
        if exp["learned"]:
            lines.append(f"   学び: {exp['learned'][:100]}")
    return "\n".join(lines)


def format_knowledge_context(knowledge: list[dict]) -> str:
    """
    search_shobo_knowledge()の結果をプロンプトに差し込める文字列に整形する。
    """
    if not knowledge:
        return ""
    lines = ["【関連する消防知識】"]
    for i, k in enumerate(knowledge, 1):
        title = k["title"] or f"知識{i}"
        lines.append(f"{i}. {title}")
        lines.append(f"   {k['content'][:150]}")
    return "\n".join(lines)