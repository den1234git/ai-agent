"""
meeting_vote.py

LAが判断に迷った時／自己修正ループが上限まで失敗し続けた時に「会議室」を
招集し、教室の生徒たち(router.pyのROUTER_STUDENT_POOL)からランダムに
繰り返し抽出して意見を集め、多数決で結論を出す仕組み。

設計方針(2026-08-05、ユーザーと合意):
- 各回は完全に独立して聞く。他の回の回答は一切見せない。
  互いの発言に引っ張られる(アンカリング)問題を、構造的に「見せない」ことで
  防ぐ(peer_review.pyの_ask_one_evaluatorと同じ考え方)。わざと対立を作る
  ような工夫は不要という結論になった。
  ※ これとは別に「仕様書を会議室(MeetingRoomPanel)で議論して煮詰める」方は
     お互いの発言が見える設計が正解(議論させたい場面のため)。あちらは
     同調/引っ張られ対策として「反対意見担当」の検討が別途必要(未実装)。
- デフォルト30回。速度より結果の質を優先(ユーザー明言、2026-08-05)。
- 回答は自由文になりうるため単純な文字列一致では多数決できない。
  30件集めたあと、別の1体(要約役)に「似た意見をグループ分けして」と
  依頼し、グループごとの票数を数える。
- 過半数(有効票の50%超)に届くグループが無い場合は自動で決めず、
  needs_human=Trueを返す(無理に多数決で押し切らない)。
- 会議の記録は毎回 server/meeting_logs/ にJSONで残す
  (今後作るフロー可視化UIの「仕様書ノード」等でも使う想定)。

呼び出し例(router.pyのrun_loop_generation、失敗時のフォールバックとして):
    from meeting_vote import convene_meeting
    result = convene_meeting(
        "この方針で進めていいか？",
        context="直近3回の自己評価が全て不合格だった",
        generation=generation,  # progress.update()で進捗を出すため(任意)
    )
    if result["needs_human"]:
        ...人に確認を求めるメッセージを返す...
"""

import os
import json
import random
from datetime import datetime

from ai import ask_ai
import progress as _p

MEETING_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meeting_logs")

DEFAULT_ROUNDS = 30
SUMMARIZER_ROLE = "router"  # 要約役。5軸評価でrouter(qwen2.5:3b)が総合最高だったため固定
MAJORITY_RATIO = 0.5  # 最多クラスタの票数/有効票数 がこれを超えたら多数決成立


def _ask_one_opinion(role: str, question: str, context: str) -> dict:
    """
    1体に完全に独立で意見を聞く。他の回の回答は一切見せない(アンカリング防止)。
    失敗した場合は棄権として扱い、多数決の母数からも除外する
    (peer_review.pyのevaluate_relevance_multiと同じ方針)。
    """
    context_block = f"\n背景:\n{context}\n" if context.strip() else ""
    prompt = f"""
あなたはAI_001の生徒の1人として、判断に迷っている以下の議題について
あなた自身の意見を述べてください。他の生徒の意見は見えない状態です。
自分の考えだけで判断してください。
{context_block}
議題:
{question}

以下のJSONのみ返してください：
{{
  "opinion": "あなたの結論を短く",
  "reason": "その理由を1〜2文で"
}}
""".strip()
    try:
        result = ask_ai(prompt, model=role)
        return {
            "role": role,
            "voted": True,
            "opinion": str(result.get("opinion", "")).strip(),
            "reason": str(result.get("reason", "")).strip(),
        }
    except Exception as e:
        print(f"[MEETING] {role} の意見取得に失敗しました（棄権として扱います）: {e}")
        return {"role": role, "voted": False, "opinion": None, "reason": f"取得失敗（{e}）"}


def _aggregate_opinions(question: str, opinions: list) -> dict:
    """
    自由文の意見群を、別の1体(要約役)に似たものどうしグループ分けさせる。
    票数の集計自体はPython側で行う(モデルの自己申告する数字は信用しない)。

    戻り値: {"ok": bool, "clusters": [{"label": str, "member_indexes": [int,...]}], "error": str|None}
    """
    voted = [o for o in opinions if o["voted"]]
    if not voted:
        return {"ok": False, "clusters": [], "error": "全員棄権"}

    listed = "\n".join(f"{i+1}. {o['opinion']}（理由: {o['reason']}）" for i, o in enumerate(voted))
    prompt = f"""
以下は「{question}」という議題に対する、複数の生徒からの意見一覧です。
内容が実質的に同じ意見どうしをグループ分けしてください
（表現が違うだけで結論が同じなら同じグループにしてください）。

意見一覧:
{listed}

以下のJSONのみ返してください：
{{
  "clusters": [
    {{"label": "グループの結論を短く要約", "member_numbers": [1, 3, 7]}}
  ]
}}
""".strip()
    try:
        result = ask_ai(prompt, model=SUMMARIZER_ROLE)
        clusters_raw = result.get("clusters", [])
        clusters = []
        for c in clusters_raw:
            numbers = [n for n in c.get("member_numbers", []) if isinstance(n, int) and 1 <= n <= len(voted)]
            if numbers:
                clusters.append({"label": str(c.get("label", "")).strip(), "member_indexes": [n - 1 for n in numbers]})
        if not clusters:
            return {"ok": False, "clusters": [], "error": "集約結果が空でした"}
        return {"ok": True, "clusters": clusters, "error": None}
    except Exception as e:
        print(f"[MEETING] 意見の集約に失敗しました: {e}")
        return {"ok": False, "clusters": [], "error": str(e)}


def convene_meeting(question: str, context: str = "", rounds: int = DEFAULT_ROUNDS, generation: int | None = None) -> dict:
    """
    会議室を招集し、questionについて教室の生徒たちの多数決で結論を出す。

    generation を渡すと、progress.update()でこの会議の進行状況を
    (run_loop_generation等と同じ仕組みで)UIから見られるようにする。
    Ollamaは同時1件しか処理できないため、rounds=30だと直列で数分かかりうる
    (ユーザー了承済み、速度より結果の質を優先する方針)。

    戻り値:
    {
      "decided": bool,       # 多数決が成立したか
      "verdict": str|None,   # 成立した場合の結論(クラスタのlabel)
      "vote_count": int,     # verdictを支持した票数
      "total_votes": int,    # 有効票数(棄権除く)
      "needs_human": bool,   # 過半数不成立・集約失敗などで人に判断を委ねるべきか
      "opinions": [...],     # rounds件の生データ
      "clusters": [...],     # 集約結果
      "log_path": str,       # 保存したログファイルのパス
    }
    """
    # router.pyとの循環importを避けるため、関数内で遅延importする。
    from router import ROUTER_STUDENT_POOL

    opinions = []
    for i in range(rounds):
        if generation is not None:
            pct = int(90 * (i / rounds)) if rounds else 0
            _p.update(generation, pct, f"会議中（{i + 1}/{rounds}体目）")
        role = random.choice(ROUTER_STUDENT_POOL)
        opinions.append(_ask_one_opinion(role, question, context))

    if generation is not None:
        _p.update(generation, 92, "意見を集約しています")

    agg = _aggregate_opinions(question, opinions)

    voted_count = sum(1 for o in opinions if o["voted"])
    decided = False
    verdict = None
    vote_count = 0

    if agg["ok"] and voted_count > 0:
        best = max(agg["clusters"], key=lambda c: len(c["member_indexes"]))
        vote_count = len(best["member_indexes"])
        if vote_count / voted_count > MAJORITY_RATIO:
            decided = True
            verdict = best["label"]

    needs_human = not decided

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(MEETING_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(MEETING_LOGS_DIR, f"{timestamp}.json")
    log_data = {
        "timestamp": timestamp,
        "question": question,
        "context": context,
        "rounds": rounds,
        "opinions": opinions,
        "aggregation": agg,
        "decided": decided,
        "verdict": verdict,
        "vote_count": vote_count,
        "total_votes": voted_count,
        "needs_human": needs_human,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    if generation is not None:
        _p.update(generation, 95, "会議が終わりました" if decided else "会議室でも結論が出ませんでした")

    return {
        "decided": decided,
        "verdict": verdict,
        "vote_count": vote_count,
        "total_votes": voted_count,
        "needs_human": needs_human,
        "opinions": opinions,
        "clusters": agg.get("clusters", []),
        "log_path": log_path,
    }
