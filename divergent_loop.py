# -*- coding: utf-8 -*-
"""
divergent_loop.py
==================
AI_001向け「発散的生成 → LFM評価 → ChromaDB書き戻し」の最小実装。

このファイルは G:\\ai_agent\\server\\divergent_loop.py に配置する想定。
既存の ai.py（ask_ai経由でOllamaを呼ぶ）と memory_store.py（G:のChromaDBに
接続済みのクライアント）をそのまま使い回す。D:は一切使わない。

構成:
    1. generate_divergent_ideas()   : ワーカー役(qwen2.5:3b)に定石外れの実装案を複数出させる
    2. evaluate_with_orchestrator() : オーケストレーター役(qwen2.5:3b)が各案をスコアリング
    3. write_back_to_chromadb()     : 一定スコア以上の案を「発散パターン候補」として保存
    4. run_once()                   : 上記を1テーマ分まとめて実行するエントリポイント

前提・注意:
    - ワーカー/オーケストレーターは両方とも router 役の qwen2.5:3b を使う
      （lfm2.5:8bは削除済みのため）。
    - ask_ai() は既にJSONパース済みのdictを返す設計なので、本スクリプトでは
      「JSON配列/JSONオブジェクトのテキストを自前で抽出する」処理は行わず、
      ask_ai() が返したdictをそのまま解釈する。
    - ChromaDBのクライアントは memory_store.py の _get_client() を使い回す。
      新規クライアントは作らない。コレクション名は "divergent_candidates"
      （既存の "la_experience" / "shobo_knowledge" は汚さない）。
    - まず --dry-run（デフォルト）でログ出力だけ確認してから、
      --commit で実際の書き込みを有効にすることを推奨します。
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ai import ask_ai
from memory_store import _get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("divergent_loop")


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

@dataclass
class Idea:
    """ワーカーAIが生成した1つの発散案"""
    id: str
    theme: str
    content: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Evaluation:
    """オーケストレーターによる評価結果"""
    idea_id: str
    score_feasibility: int      # 実装可能性 1-5
    score_novelty: int          # 新規性 1-5
    score_usefulness: int       # 有用性（FDMA要件・実務適合） 1-5
    reason: str                 # 一言理由
    total_score: float = field(init=False)

    def __post_init__(self) -> None:
        # 重み付けはお好みで調整可。実務直結を重視して usefulness をやや厚めに。
        self.total_score = round(
            self.score_feasibility * 0.3
            + self.score_novelty * 0.3
            + self.score_usefulness * 0.4,
            2,
        )


# ---------------------------------------------------------------------------
# 1. 生成: 発散モードのプロンプトでワーカー役(qwen2.5:3b)に案を出させる
# ---------------------------------------------------------------------------

DIVERGENT_PROMPT_TEMPLATE = """あなたは消防点検管理ソフトウェアの実装アイデアを考えるエンジニアです。
以下のテーマについて、"あえて定石やよくある実装から外れた"、独自性のあるアプローチを{n}個提案してください。

# テーマ
{theme}

# 制約
- 既存アーキテクチャ（JSONスキーマベース、ReportLabでのPDF生成、QRコードでのデータ転送、
  ローカル推論による機微データの端末内処理）とは矛盾しないこと
- 実装が完全に不可能な空想は避け、現実的に検討できる範囲の独自性であること

JSONのみ返してください。説明文・コードブロック不要。

出力形式：
{{
  "ideas": [
    {{"idea": "案の説明（200字程度）"}}
  ]
}}
"""


def build_divergent_prompt(theme: str, n: int = 3) -> str:
    return DIVERGENT_PROMPT_TEMPLATE.format(theme=theme, n=n)


def call_worker_model(prompt: str) -> dict:
    """
    ワーカー役(qwen2.5:3b)を ask_ai 経由で呼ぶ。
    ask_ai は既にJSONパース済みのdictを返すので、そのまま返す。
    """
    return ask_ai(prompt, model="router")


def generate_divergent_ideas(theme: str, n: int = 3) -> list[Idea]:
    """発散モードでワーカー役にn件のアイデアを生成させる"""
    prompt = build_divergent_prompt(theme, n)
    result = call_worker_model(prompt)

    raw_ideas = result.get("ideas", [])
    if not raw_ideas:
        logger.warning(
            "アイデアが抽出できませんでした（想定外の形式で応答された可能性）: %r",
            result,
        )

    ideas = [
        Idea(id=str(uuid.uuid4()), theme=theme, content=item.get("idea", ""))
        for item in raw_ideas
        if item.get("idea")
    ]
    logger.info("生成完了: theme=%r, %d件", theme, len(ideas))
    return ideas


# ---------------------------------------------------------------------------
# 2. 評価: オーケストレーター役(qwen2.5:3b)が各案をスコアリング
# ---------------------------------------------------------------------------

EVAL_PROMPT_TEMPLATE = """あなたは消防点検管理ソフトウェア開発のオーケストレーターAIです。
以下の実装アイデアを3つの観点で評価してください。

# アイデア
{idea}

# 評価観点（各1-5点）
- feasibility: 現在のアーキテクチャで実装可能か
- novelty: 既存の一般的な実装と比べてどれだけ新規性があるか
- usefulness: 消防法・FDMA要件や実務での有用性

JSONのみ返してください。説明文・コードブロック不要。

出力形式：
{{
  "feasibility": 1,
  "novelty": 1,
  "usefulness": 1,
  "reason": "一言理由（50字程度）"
}}
"""


def call_orchestrator_model(prompt: str) -> dict:
    """
    オーケストレーター役(qwen2.5:3b)を ask_ai 経由で呼ぶ。
    ワーカーと同じrouterモデルを使う（lfm2.5:8bは削除済みのため）。
    """
    return ask_ai(prompt, model="router")


def evaluate_with_orchestrator(idea: Idea) -> Evaluation:
    """オーケストレーター役に1件のアイデアを評価させる"""
    prompt = EVAL_PROMPT_TEMPLATE.format(idea=idea.content)
    result = call_orchestrator_model(prompt)

    try:
        evaluation = Evaluation(
            idea_id=idea.id,
            score_feasibility=int(result.get("feasibility", 1)),
            score_novelty=int(result.get("novelty", 1)),
            score_usefulness=int(result.get("usefulness", 1)),
            reason=result.get("reason", ""),
        )
    except (TypeError, ValueError) as e:
        logger.warning(
            "評価結果のパースに失敗しました（最低評価として扱います）: %s / raw=%r",
            e, result,
        )
        evaluation = Evaluation(
            idea_id=idea.id,
            score_feasibility=1,
            score_novelty=1,
            score_usefulness=1,
            reason="評価失敗",
        )

    logger.info(
        "評価完了: idea_id=%s total=%.2f (feas=%d, nov=%d, use=%d) reason=%s",
        idea.id[:8], evaluation.total_score,
        evaluation.score_feasibility, evaluation.score_novelty,
        evaluation.score_usefulness, evaluation.reason,
    )
    return evaluation


# ---------------------------------------------------------------------------
# 3. 書き戻し: 一定スコア以上の案をChromaDBに「発散パターン候補」として保存
#    memory_store.py の _get_client() を使い回す（新規クライアントは作らない）
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 3.5
COLLECTION_NAME = "divergent_candidates"  # 既存の la_experience / shobo_knowledge とは別枠


def write_back_to_chromadb(
    idea: Idea,
    evaluation: Evaluation,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = True,
) -> bool:
    """
    スコアが閾値以上のアイデアだけをChromaDBに書き戻す。
    まさが後で目視確認して、良ければ「成功パターン」コレクションに昇格させる想定
    （＝ここでは一旦 divergent_candidates という別コレクションに留め、既存の
      la_experience / shobo_knowledge コレクションを汚さないようにしている）。

    dry_run=True の間は実際の書き込みはせず、ログ出力のみ行う。
    """
    if evaluation.total_score < threshold:
        logger.info(
            "却下: idea_id=%s score=%.2f < threshold=%.2f",
            idea.id[:8], evaluation.total_score, threshold,
        )
        return False

    record = {
        "id": idea.id,
        "theme": idea.theme,
        "content": idea.content,
        "created_at": idea.created_at,
        "score_total": evaluation.total_score,
        "score_feasibility": evaluation.score_feasibility,
        "score_novelty": evaluation.score_novelty,
        "score_usefulness": evaluation.score_usefulness,
        "reason": evaluation.reason,
        "status": "pending_review",  # まさの目視確認待ち
    }

    if dry_run:
        logger.info(
            "[DRY-RUN] ChromaDB書き戻し（未実行）: %s",
            json.dumps(record, ensure_ascii=False),
        )
        return True

    # memory_store.py が管理しているクライアント（= F:\\ai_agent\server\chroma_db）を使い回す
    client = _get_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    collection.add(
        ids=[record["id"]],
        documents=[record["content"]],
        metadatas=[{k: v for k, v in record.items() if k not in ("id", "content")}],
    )
    logger.info("ChromaDB書き戻し完了: idea_id=%s", idea.id[:8])
    return True


# ---------------------------------------------------------------------------
# 4. エントリポイント: 1テーマ分をまとめて実行
# ---------------------------------------------------------------------------

def run_once(
    theme: str,
    n_ideas: int = 3,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = True,
) -> list[tuple[Idea, Evaluation, bool]]:
    """
    1テーマについて 生成→評価→書き戻し を一気通貫で実行し、結果一覧を返す。
    戻り値: [(Idea, Evaluation, 書き戻したか), ...]
    """
    results = []
    ideas = generate_divergent_ideas(theme, n_ideas)
    for idea in ideas:
        evaluation = evaluate_with_orchestrator(idea)
        written = write_back_to_chromadb(
            idea, evaluation, threshold=threshold, dry_run=dry_run,
        )
        results.append((idea, evaluation, written))

    accepted = sum(1 for _, _, w in results if w)
    logger.info("完了: theme=%r, 生成=%d件, 採用=%d件", theme, len(ideas), accepted)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="発散的生成→評価→書き戻しループの実行")
    parser.add_argument("theme", type=str, help="発散させたいテーマ（例: 'CSV取込のエラーハンドリング'）")
    parser.add_argument("-n", "--num-ideas", type=int, default=3, help="生成するアイデア数")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD, help="採用スコア閾値")
    parser.add_argument("--dry-run", action="store_true", default=True, help="実際には書き込まずログのみ（デフォルト）")
    parser.add_argument("--commit", dest="dry_run", action="store_false", help="実際にChromaDBへ書き込む")
    args = parser.parse_args()

    if not args.dry_run:
        logger.warning("--commit 指定: 実際にChromaDBへ書き込みます")

    run_once(
        theme=args.theme,
        n_ideas=args.num_ideas,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()