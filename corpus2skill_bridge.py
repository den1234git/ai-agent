"""Corpus2Skill (G:\\ai_agent\\Corpus2Skill) と、このサーバーのChromaDB知識ベース・
Ollamaバックエンドを橋渡しするモジュール。「深く考える」モード用。

「通常」モード（ask_ai_with_rag）はshobo_knowledge/medical_knowledgeをベクトル検索で
上位N件そのままプロンプトに詰め込むだけなのに対し、こちらはCorpus2Skillが事前に
作ったSKILL.md/INDEX.mdの階層をLLMに1階層ずつ辿らせて絞り込んでから答える
（Anthropicの hosted Skills API が本来やる役割を、ask_ai() 経由のOllama呼び出しで
自前実装したもの）。速度より精度を優先する用途向け。

処理の流れ：
  1. compile_knowledge_skills(): shobo_knowledge / medical_knowledge のChromaDB
     コレクションを一時ディレクトリにtxtダンプし、Corpus2Skillのcompile_corpus()で
     server/c2s_output_<target>/.claude/skills/ 配下にSKILL.md/INDEX.md階層を作る。
  2. think(): その階層をトップから辿る。各階層でask_ai()に「どの候補が質問に
     関連するか」を選ばせ、葉ノードに着いたら関連文書をdocuments.jsonから読み出し、
     最終的な回答を生成する。
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

# Corpus2Skillはこのワークスペース内の兄弟ディレクトリとして独立にクローンされている
# （server/ai.pyへの依存はCorpus2Skill側から張っている。こちらはCorpus2Skillの
# パッケージをインポートする側）。
_CORPUS2SKILL_DIR = Path(__file__).resolve().parents[1] / "Corpus2Skill"
if str(_CORPUS2SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_CORPUS2SKILL_DIR))

from corpus2skill.config import CompileConfig  # noqa: E402
from corpus2skill.compile import compile_corpus  # noqa: E402

from ai import ask_ai  # noqa: E402

_BASE_DIR = Path(__file__).resolve().parent

# shobo_knowledge / medical_knowledge を分けてコンパイルする（合意済みの設計）。
OUTPUT_DIRS: dict[str, Path] = {
    "shobo": _BASE_DIR / "c2s_output_shobo",
    "medical": _BASE_DIR / "c2s_output_medical",
}
TARGET_LABELS = {"shobo": "消防知識", "medical": "医学知識"}

THINK_MODEL = "swallow"
MAX_NAV_DEPTH = 6
MAX_DOCS_FOR_ANSWER = 3

ProgressCallback = Optional[Callable[[int, str], None]]


def _tick(progress_cb: ProgressCallback, percent: int, step: str) -> None:
    if progress_cb:
        progress_cb(percent, step)


# ---------------------------------------------------------------------------
# compile: ChromaDB → 一時ディレクトリ(txt) → Corpus2Skill compile_corpus()
# ---------------------------------------------------------------------------

def _export_collection_to_dir(collection, out_dir: Path) -> int:
    """ChromaDBコレクションの全ドキュメントを1件1ファイルのtxtとして書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collection.get(include=["metadatas", "documents"])
    count = 0
    for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"] or []):
        title = (meta or {}).get("title") or ""
        text = f"{title}\n\n{doc}".strip() if title else (doc or "").strip()
        if not text:
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)[:100] or f"doc{count}"
        (out_dir / f"{safe_id}.txt").write_text(text, encoding="utf-8")
        count += 1
    return count


def compile_knowledge_skills(target: str, progress_cb: ProgressCallback = None) -> dict:
    """target ("shobo" | "medical") のChromaDB知識をエクスポートし、
    server/c2s_output_<target>/ にSKILL.md/INDEX.md階層としてコンパイルする。
    """
    if target not in OUTPUT_DIRS:
        raise ValueError(f"unknown target: {target}")

    from memory_store import _get_knowledge_col, _get_medical_col
    collection = _get_knowledge_col() if target == "shobo" else _get_medical_col()
    output_dir = OUTPUT_DIRS[target]

    _tick(progress_cb, 5, f"{TARGET_LABELS[target]}をChromaDBからエクスポートしています")

    with tempfile.TemporaryDirectory(prefix=f"c2s_staging_{target}_") as staging:
        staging_dir = Path(staging)
        num_docs = _export_collection_to_dir(collection, staging_dir)
        if num_docs == 0:
            return {"status": "empty", "num_docs": 0, "output_dir": str(output_dir)}

        _tick(progress_cb, 15, f"{num_docs}件をスキル階層にコンパイルしています（時間がかかります）")

        cfg = CompileConfig(input_dir=staging_dir, output_dir=output_dir)
        compile_corpus(cfg)

    _tick(progress_cb, 95, "コンパイル結果を確認しています")
    skills_dir = output_dir / ".claude" / "skills"
    num_skills = sum(1 for d in skills_dir.iterdir() if d.is_dir()) if skills_dir.exists() else 0

    return {
        "status": "ok",
        "num_docs": num_docs,
        "num_skills": num_skills,
        "output_dir": str(output_dir),
    }


def compiled_status() -> dict:
    """各targetがコンパイル済みかどうかをUI表示用に返す。"""
    result = {}
    for target, output_dir in OUTPUT_DIRS.items():
        skills_dir = output_dir / ".claude" / "skills"
        doc_store = output_dir / "documents.json"
        compiled = skills_dir.exists() and doc_store.exists()
        num_skills = sum(1 for d in skills_dir.iterdir() if d.is_dir()) if compiled else 0
        result[target] = {"compiled": compiled, "num_skills": num_skills}
    return result


# ---------------------------------------------------------------------------
# think: SKILL.md / INDEX.md 階層をLLMで辿って絞り込み、回答する
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_DOC_ROW_RE = re.compile(r"^- `([^`]+)`(.*)$", re.MULTILINE)
_SUBGROUP_ROW_RE = re.compile(r"^- \*\*([^*]+)/\*\*\s*\((\d+) docs?\):\s*(.+)$", re.MULTILINE)


def _parse_frontmatter(text: str) -> dict:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    block = m.group(1)
    name_m = re.search(r"^name:\s*(.+)$", block, re.MULTILINE)
    desc_m = re.search(r"^description:\s*>\s*\n((?:[ \t].*\n?)*)", block, re.MULTILINE)
    description = ""
    if desc_m:
        description = " ".join(line.strip() for line in desc_m.group(1).splitlines()).strip()
    return {
        "name": name_m.group(1).strip() if name_m else "",
        "description": description,
    }


def _extract_section(text: str, header_prefix: str) -> Optional[str]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("### ") or lines[i].startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end])


def _parse_subgroups(text: str) -> list[dict]:
    section = _extract_section(text, "### Sub-groups")
    if section is None:
        return []
    return [
        {"name": m.group(1), "num_docs": int(m.group(2)), "summary": m.group(3).strip()}
        for m in _SUBGROUP_ROW_RE.finditer(section)
    ]


def _parse_documents(text: str) -> list[dict]:
    section = _extract_section(text, "### Documents")
    if section is None:
        return []
    docs = []
    for m in _DOC_ROW_RE.finditer(section):
        doc_id = m.group(1)
        rest = m.group(2).strip().lstrip(":—-").strip()
        docs.append({"doc_id": doc_id, "summary": rest})
    return docs


def _nav_file_for(dir_path: Path) -> Optional[Path]:
    for name in ("SKILL.md", "INDEX.md"):
        p = dir_path / name
        if p.exists():
            return p
    return None


def _list_top_level_skills(skills_dir: Path) -> list[dict]:
    entries = []
    for d in sorted(skills_dir.iterdir()):
        nav = d / "SKILL.md"
        if not d.is_dir() or not nav.exists():
            continue
        fm = _parse_frontmatter(nav.read_text(encoding="utf-8"))
        entries.append({"path": d.name, "dir": d, "description": fm.get("description", "")})
    return entries


def _choose_one(query: str, candidates: list[dict], key_field: str, desc_field: str, kind_label: str) -> str:
    """候補群から1つだけ選ばせる（トップレベルskill選択・サブグループ選択で共用）。"""
    if len(candidates) == 1:
        return candidates[0][key_field]

    listing = "\n".join(
        f"- {c[key_field]}: {(c.get(desc_field) or '')[:300]}" for c in candidates
    )
    prompt = (
        f"ユーザーの質問に最も関連する{kind_label}を1つだけ選んでください。\n\n"
        f"質問: {query}\n\n"
        f"候補:\n{listing}\n\n"
        "以下のJSONのみ返してください：\n"
        '{ "choice": "<候補の名前をそのまま>" }'
    )
    result = ask_ai(prompt, model=THINK_MODEL, json_mode=True)
    choice = str(result.get("choice", "")).strip()

    valid = {c[key_field] for c in candidates}
    if choice in valid:
        return choice
    # モデルの出力が候補名と完全一致しない場合、部分一致で救済する
    for c in candidates:
        if c[key_field] and (c[key_field] in choice or choice in c[key_field]):
            return c[key_field]
    return candidates[0][key_field]


def _choose_documents(query: str, docs: list[dict], max_docs: int = MAX_DOCS_FOR_ANSWER) -> list[str]:
    if len(docs) <= max_docs:
        return [d["doc_id"] for d in docs]

    listing = "\n".join(f"- {d['doc_id']}: {d.get('summary', '')[:300]}" for d in docs)
    prompt = (
        f"ユーザーの質問に答えるために読むべき文書を、関連度の高い順に最大{max_docs}件選んでください。\n\n"
        f"質問: {query}\n\n"
        f"候補:\n{listing}\n\n"
        "以下のJSONのみ返してください：\n"
        '{ "doc_ids": ["...", "..."] }'
    )
    result = ask_ai(prompt, model=THINK_MODEL, json_mode=True)
    ids = result.get("doc_ids") or []
    valid = {d["doc_id"] for d in docs}
    picked = [i for i in ids if i in valid][:max_docs]
    return picked or [d["doc_id"] for d in docs[:max_docs]]


def _load_document_texts(output_dir: Path, doc_ids: list[str]) -> dict[str, str]:
    doc_store_path = output_dir / "documents.json"
    if not doc_store_path.exists():
        return {}
    data = json.loads(doc_store_path.read_text(encoding="utf-8"))
    return {did: data[did] for did in doc_ids if did in data}


def think(
    query: str,
    target: str,
    model: str = THINK_MODEL,
    max_depth: int = MAX_NAV_DEPTH,
    progress_cb: ProgressCallback = None,
) -> dict:
    """スキル階層をトップから辿って質問に答える。

    戻り値: {"answer": str, "trail": [str, ...], "sources": [doc_id, ...]}
    trail は「どの階層をどう辿ったか」のログ（回答の根拠を人間が確認できるように）。
    """
    if target not in OUTPUT_DIRS:
        raise ValueError(f"unknown target: {target}")

    output_dir = OUTPUT_DIRS[target]
    skills_dir = output_dir / ".claude" / "skills"

    if not skills_dir.exists():
        return {
            "answer": (
                f"「{TARGET_LABELS.get(target, target)}」のスキル階層がまだコンパイルされていません。"
                "先に「スキル階層を更新」を実行してください。"
            ),
            "trail": [],
            "sources": [],
        }

    trail: list[str] = []
    top_candidates = _list_top_level_skills(skills_dir)
    if not top_candidates:
        return {"answer": "コンパイル済みのスキルが見つかりませんでした。", "trail": [], "sources": []}

    _tick(progress_cb, 20, "関連するスキルを絞り込んでいます")
    chosen_name = _choose_one(query, top_candidates, "path", "description", "スキル")
    current_dir = next(c["dir"] for c in top_candidates if c["path"] == chosen_name)
    trail.append(f"skill: {chosen_name}")

    docs: list[dict] = []
    doc_lookup: dict[str, dict] = {}
    percent = 25

    for _ in range(max_depth):
        nav = _nav_file_for(current_dir)
        if nav is None:
            docs = []
            break

        text = nav.read_text(encoding="utf-8")
        subgroups = _parse_subgroups(text)
        docs = _parse_documents(text)
        for d in docs:
            doc_lookup[d["doc_id"]] = d

        if subgroups and not docs:
            percent = min(percent + 10, 60)
            _tick(progress_cb, percent, "サブグループを絞り込んでいます")
            chosen = _choose_one(query, subgroups, "name", "summary", "サブグループ")
            trail.append(f"sub-group: {chosen}")
            current_dir = current_dir / chosen
            continue

        break

    if not docs:
        return {
            "answer": "関連する文書が見つかりませんでした。表現を変えて質問し直してください。",
            "trail": trail,
            "sources": [],
        }

    _tick(progress_cb, 65, "参照する文書を選んでいます")
    chosen_doc_ids = _choose_documents(query, docs)
    trail.append("documents: " + ", ".join(chosen_doc_ids))

    doc_texts = _load_document_texts(output_dir, chosen_doc_ids)

    _tick(progress_cb, 80, "回答を生成しています")
    context_block = "\n\n".join(
        f"--- 資料: {doc_lookup.get(did, {}).get('summary', did)[:80]} ---\n{doc_texts[did]}"
        for did in chosen_doc_ids
        if did in doc_texts
    )

    if not context_block:
        return {
            "answer": "関連文書の本文を読み込めませんでした。",
            "trail": trail,
            "sources": chosen_doc_ids,
        }

    answer_prompt = (
        "以下の参考資料をもとに、ユーザーの質問に答えてください。"
        "資料に書かれていないことは推測で補わず、資料からわからない場合は正直にそう述べてください。\n\n"
        f"質問: {query}\n\n"
        f"参考資料:\n{context_block}\n\n"
        "回答:"
    )
    result = ask_ai(answer_prompt, model=model, json_mode=False)
    answer = result.get("code", "") or "回答の生成に失敗しました。"

    return {"answer": answer, "trail": trail, "sources": chosen_doc_ids}
