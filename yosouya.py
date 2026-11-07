"""
yosouya.py — 予想屋さん（第1段：材料あり独立見積もり）
============================================================
第1段の工程：
  ① 問いを受け取る
  ② 調べ物リスト生成（LLMが「何を調べるべきか」のクエリを出す）
  ③ 材料を取得し、格付けと突き合わせを行い、全員に同じ材料を配る
     格付け  : 各ソースに信頼度ラベル（公式/通信社/ニュース/集計）を付ける
     突き合わせ: 1件のみ言及の主張に「（未確認）」の札を付けて渡す
  ④ 村の各モデルが独立に確率(0〜100)＋基準率＋根拠を出す（根拠なしの数字は出せない）
  ⑤ 中央値・最小・最大を計算
  ⑥ ~/workspace/yosouya/ledger.jsonl に1行追記
  ⑦ 結果dictを返す（画面表示用）

設計メモ（2026-11-03「予想屋さん まとめ」）準拠。
"""
from __future__ import annotations

import os
import re
import json
import time
import datetime
import statistics
import concurrent.futures

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

LEDGER_PATH = os.environ.get(
    "YOSOUYA_LEDGER",
    r"C:\Users\USER\workspace\yosouya\ledger.jsonl"
)

DEFAULT_MEMBERS = [
    {"kote": "[村]巌",   "model": "deepseek-r1:32b"},
    {"kote": "[村]理明", "model": "gemma3:4b"},
    {"kote": "[村]否問", "model": "gemma3:4b"},
    {"kote": "[村]夕凪", "model": "sougoshitsumon:latest"},
    {"kote": "[村]神楽", "model": "gemma3:4b"},
    {"kote": "[村]銀風", "model": "sougoshitsumon:latest"},
]

# 材料つきシステムプロンプト。基準率と根拠を必須化。
_SYSTEM_WITH_MATERIALS = """\
あなたは村の『予想屋』の一員です。
以下の【材料】をすべて読んでから、問いに対する確率を独立した見立てで見積もってください。

【手順・厳守】
1. まず「基準率」を述べる：過去の似た事例では何%くらいで起きているか（1文で）。
2. 次に「上げる要因」と「下げる要因」を材料から1つずつ挙げる。
3. 最後の行に「確率: NN%」の形で0〜100の整数を1つだけ書く（他の行に%を書かない）。
材料が足りない場合は「材料が足りないため不確かだが」と明示して見積もること。
根拠なしで数字だけ書くことは禁止。\
"""

# 材料なし（フォールバック）— Tavily未設定時に表示するためだけに使う
_SYSTEM_NO_MATERIALS = """\
あなたは村の『予想屋』の一員です。
【注意】今回は外部の材料が取得できませんでした。
自分の学習知識の中の基準率（似た過去事例の頻度）のみで見積もってください。
「材料なし・学習データのみ」と最初に明示し、不確かさを大きめに取ること。

【手順・厳守】
1. 基準率を述べる（1文）。
2. 上げる要因・下げる要因を各1つ。
3. 最後の行に「確率: NN%」の形で整数を1つだけ書く（他の行に%を書かない）。\
"""

_PROB_RE = re.compile(r"確率[\s:：]*?(\d{1,3})\s*[%％]")
_FALLBACK_RE = re.compile(r"(\d{1,3})\s*[%％]")

# 材料取得パラメータ
_RESULTS_PER_QUERY = 5

# 調べ物リスト生成プロンプト
_SYSTEM_RESEARCH_LIST = """\
あなたは『調査担当』です。
以下の【問い】に答えるために何を調べるべきか、検索クエリのリストを出してください。

【ルール】
- クエリは英語で書く（国際的な情報を取るため）
- 1行1クエリ、最大5個
- 「世論調査・最新の数字」「過去の似た事例の基準率」「現状の情勢変化」の3つを軸にする
- 番号や箇条書き記号は不要。クエリの文字列だけを1行ずつ書く
- 予測市場の価格は調べない（採点用なので材料に入れない）\
"""


def _generate_research_queries(question: str, model: str = "") -> list[str]:
    """
    LLMに「この問いに答えるために何を調べるか」を聞き、
    検索クエリのリストを返す。

    LLMが応答しない場合はルールベースのフォールバックを返す。
    """
    if not model:
        # 軽量なモデルを選ぶ（調べ物リスト生成は推論より整形力が大事）
        model = "gemma3:4b"  # インストール済みの軽量モデル

    prompt = (f"[SYSTEM]\n{_SYSTEM_RESEARCH_LIST}\n"
              f"[USER]\n【問い】\n{question}\n[ASSISTANT]\n")
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 200}},
            timeout=15,
        )
        raw = re.sub(r"<think>.*?</think>", "", r.json().get("response", ""), flags=re.DOTALL)
        lines = [ln.strip() for ln in raw.strip().splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        # 番号付き・箇条書きの先頭記号を除去
        lines = [re.sub(r"^[\d\.\-\*・]+\s*", "", ln) for ln in lines]
        queries = [ln for ln in lines if ln][:5]
        if queries:
            print(f"[yosouya] 調べ物リスト生成: {queries}")
            return queries
    except Exception as e:
        print(f"[yosouya] 調べ物リスト生成失敗（フォールバックへ）: {e}")

    # フォールバック: ルールベースで最低限のクエリを作る
    q = question
    fb: list[str] = []
    if "選挙" in q or "midterm" in q.lower():
        import re as _re
        year_m = _re.search(r"(20\d\d)", q)
        year = year_m.group(1) if year_m else "2026"
        fb = [
            f"{year} midterm election forecast poll latest",
            f"{year} House Democrats Republicans generic ballot",
            f"US midterm election historical base rate incumbent party",
        ]
    else:
        fb = [q[:60] + " forecast poll", q[:60] + " historical base rate"]
    print(f"[yosouya] 調べ物リスト（フォールバック）: {fb}")
    return fb


# ── 材料取得 ──────────────────────────────────────────────────────

def gather_materials(question: str, research_queries: list[str] | None = None) -> dict:
    """
    材料を収集し、格付け・突き合わせを行い、全村人に配れる形にまとめる。

    research_queries : LLMが生成した調べ物リスト。None の場合はルールベースで補完。

    情報源の優先順位（設計メモ 2026-11-03 準拠）:
      Layer 1: Wikipedia 世論調査表（選挙トピックのみ）
      Layer 2: Google News RSS（最新ニュース）— research_queries のクエリを使う
      Layer 3: Tavily Web検索（フォールバック / API キーがある場合）

    ※ 予測市場価格は材料に含めない（採点時の比較相手として横に置く）。

    戻り値:
      {
        "available": bool,
        "sources": [{"title", "url", "snippet", "credibility"}],
        "summary": str,           # 格付け・突き合わせ済みの村人配布テキスト
        "queries": [str],
        "layers_used": [str],
        "unconfirmed_count": int, # 「未確認」の札を付けたソース数
      }
    """
    from wikipedia_polls import fetch_polls, guess_election_page
    from google_news_rss import fetch_news

    parts: list[str] = []
    sources: list[dict] = []
    queries_used: list[str] = []
    layers_used: list[str] = []

    # ── Layer 1: Wikipedia 世論調査表（選挙のみ） ────────────────
    wiki_page = guess_election_page(question)
    if wiki_page:
        print(f"[yosouya] Layer1: Wikipedia '{wiki_page}'")
        wiki = fetch_polls(wiki_page)
        if wiki.get("ok") and wiki.get("tables_found", 0) > 0:
            parts.append(f"【Wikipedia 世論調査表: {wiki['page_title']}】\n{wiki['text']}")
            sources.append({
                "title": wiki["page_title"], "url": wiki["source_url"],
                "snippet": f"世論調査表 {wiki['tables_found']}個",
                "credibility": "集計（Wikipedia）",
            })
            queries_used.append(f"Wikipedia: {wiki_page}")
            layers_used.append("Wikipedia")
        else:
            print(f"[yosouya] Layer1: 表なし or エラー: {wiki.get('error')}")

    # ── Layer 2: Google News RSS（調べ物リストのクエリを使う） ────
    # research_queries がなければルールベースで補完
    if research_queries:
        news_queries = research_queries[:3]  # 上位3件をニュース検索に使う
    else:
        from google_news_rss import build_election_queries
        news_queries = build_election_queries(question)

    for nq in news_queries:
        print(f"[yosouya] Layer2: Google News '{nq}'")
        news = fetch_news(nq, max_items=6)
        if news.get("ok") and news.get("items"):
            parts.append(news["summary"])
            for it in news["items"]:
                sources.append({
                    "title":   it["title"],
                    "url":     it["link"],
                    "snippet": it.get("description", ""),
                    "credibility": "ニュース（Google News）",
                })
            queries_used.append(f"GoogleNews: {nq}")
            if "GoogleNews" not in layers_used:
                layers_used.append("GoogleNews")
        else:
            print(f"[yosouya] Layer2: 記事なし or エラー: {news.get('error')}")

    # ── Layer 3: Tavily（フォールバック。Layer1+2 で材料がない場合） ──
    if not parts:
        print("[yosouya] Layer3: Tavily フォールバック")
        try:
            from web_search import tavily_search
            tq = (research_queries[0] if research_queries
                  else question.replace("アメリカ", "US").replace("中間選挙", "midterm election"))
            r = tavily_search(tq, max_results=_RESULTS_PER_QUERY)
            if r.get("available"):
                if r.get("answer"):
                    parts.append("【Tavily 要約】\n" + r["answer"])
                for item in r.get("results", []):
                    url = item.get("url", "")
                    if url:
                        sources.append({
                            "title":   item.get("title", ""),
                            "url":     url,
                            "snippet": (item.get("content", "") or "")[:400],
                            "credibility": "Web検索（Tavily）",
                        })
                        parts.append(f"[Tavily] {item.get('title', '')}\nURL: {url}\n"
                                     f"{(item.get('content', '') or '')[:400]}")
                queries_used.append(f"Tavily: {tq}")
                layers_used.append("Tavily")
        except ImportError:
            pass

    # ── 格付け・突き合わせ ────────────────────────────────────────
    rated_parts, unconfirmed_count = _rate_and_crosscheck(parts, sources)
    summary = "\n\n".join(rated_parts)

    available = bool(rated_parts)
    return {
        "available": available,
        "sources": sources,
        "summary": summary,
        "queries": queries_used,
        "layers_used": layers_used,
        "unconfirmed_count": unconfirmed_count,
    }


def _rate_and_crosscheck(parts: list[str], sources: list[dict]) -> tuple[list[str], int]:
    """
    材料に格付けラベルと突き合わせ（未確認の札）を付ける。

    突き合わせのロジック:
    - 各 URL のドメインを独立した「出どころ」とみなす
    - 同じ主張（スニペット内のキーフレーズ）が1件のみなら「（未確認）」と注記
    - 現在は保守的な実装: 1ソースのみのレイヤーのテキストに注記を付加する

    戻り値: (格付け済み parts リスト, 未確認件数)
    """
    import urllib.parse

    # ドメイン別のソース数を数える
    domains: dict[str, int] = {}
    for s in sources:
        url = s.get("url", "")
        try:
            domain = urllib.parse.urlparse(url).netloc
        except Exception:
            domain = url
        domains[domain] = domains.get(domain, 0) + 1

    # layers ごとに独立ソース数を数える
    layer_source_count: dict[str, int] = {}
    for s in sources:
        cred = s.get("credibility", "不明")
        layer_source_count[cred] = layer_source_count.get(cred, 0) + 1

    # 格付けラベルの定義
    credibility_label = {
        "集計（Wikipedia）":      "★★★ 集計（公式調査まとめ）",
        "ニュース（Google News）": "★★☆ 報道（Google News）",
        "Web検索（Tavily）":       "★☆☆ Web検索",
        "不明":                    "★☆☆",
    }

    rated_parts: list[str] = []
    unconfirmed_count = 0

    for part in parts:
        # この part がどのレイヤーから来たかを判定
        label = ""
        if "Wikipedia 世論調査表" in part:
            label = credibility_label["集計（Wikipedia）"]
        elif "Google News" in part:
            label = credibility_label["ニュース（Google News）"]
            # Google News 結果が1クエリのみで記事数が少ない場合は未確認フラグ
            news_count = layer_source_count.get("ニュース（Google News）", 0)
            if news_count <= 1:
                part = part + "\n※ この情報は現時点で1件の報道のみが出典です（未確認）"
                unconfirmed_count += 1
        elif "Tavily" in part:
            label = credibility_label["Web検索（Tavily）"]
            # Tavily のみの材料は未確認扱い（Layer1+2 が取れなかった場合）
            part = part + "\n※ Web検索のみで取得した情報です。他の出どころで確認できていません（未確認）"
            unconfirmed_count += 1

        if label:
            rated_parts.append(f"[{label}]\n{part}")
        else:
            rated_parts.append(part)

    return rated_parts, unconfirmed_count


# ── 村人への問い合わせ ──────────────────────────────────────────

def _ask_one(member: dict, question: str, materials_summary: str, timeout: float = 90.0) -> dict:
    """1モデルに独立見積もりを依頼し、{kote, prob, reason, model, ok} を返す。"""
    kote = member.get("kote", "?")
    model = member.get("model", "")

    if materials_summary:
        system = _SYSTEM_WITH_MATERIALS
        user_block = f"【材料】\n{materials_summary}\n\n【問い】\n{question}"
    else:
        system = _SYSTEM_NO_MATERIALS
        user_block = f"【問い】\n{question}"

    prompt = f"[SYSTEM]\n{system}\n[USER]\n{user_block}\n[ASSISTANT]\n"

    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 180}},
            timeout=timeout,
        )
        text = (r.json().get("response", "") or "").strip()
    except Exception as e:
        return {"kote": kote, "model": model, "prob": None, "reason": "", "ok": False, "error": str(e)}

    prob = _parse_prob(text)
    reason = _clean_reason(text)
    return {"kote": kote, "model": model, "prob": prob, "reason": reason,
            "ok": prob is not None, "raw": text[:500]}


def _parse_prob(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _PROB_RE.search(text)
    if not m:
        m = _FALLBACK_RE.search(text)
    if not m:
        return None
    v = int(m.group(1))
    return max(0, min(100, v))


def _clean_reason(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    keep = [ln for ln in lines if not _PROB_RE.search(ln)]
    reason = " ".join(keep).strip()
    return reason[:300]


def _fetch_market_ref(question: str) -> dict:
    """
    Polymarket と Kalshi から採点用の市場参照データを取得する。
    村人への材料には使わない。エラーは握りつぶして空を返す。

    戻り値:
      {
        "markets": bool,           # 1件でも取れたか
        "polymarket": [...],
        "kalshi": [...],
        "retrieved_at": str,
      }
    """
    from polymarket import search_markets as pm_search, build_query as pm_query
    from kalshi import search_markets as ka_search

    query = pm_query(question)
    pm_result: list[dict] = []
    ka_result: list[dict] = []

    try:
        r = pm_search(query, limit=3)
        if r.get("ok"):
            pm_result = r.get("markets", [])
    except Exception as e:
        print(f"[yosouya] Polymarket 取得失敗: {e}")

    try:
        r = ka_search(query, limit=3)
        if r.get("ok"):
            ka_result = r.get("markets", [])
    except Exception as e:
        print(f"[yosouya] Kalshi 取得失敗: {e}")

    return {
        "markets":    bool(pm_result or ka_result),
        "query":      query,
        "polymarket": pm_result,
        "kalshi":     ka_result,
        "retrieved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _append_ledger(entry: dict) -> str:
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return LEDGER_PATH


# ── メインの実行関数 ──────────────────────────────────────────────

def run(question: str, members: list | None = None, deadline: str = "") -> dict:
    """
    予想屋を1周回す。

    question : 『〜までに〜が起きる』の形が望ましい
    deadline : 結果が確定する日（YYYY-MM-DD）。空の場合は「未設定」として記録。
               ★ 投票日≠確定日。下院なら投票日の数日後を指定すること。
    """
    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "問いが空です"}
    members = members or DEFAULT_MEMBERS

    # ── 1. 調べ物リスト生成（LLMがクエリを出す） ────────────────
    print(f"[yosouya] 調べ物リストを生成中: 「{question}」")
    research_queries = _generate_research_queries(question)

    # ── 2. 材料収集（格付け・突き合わせ込み・全員に同じ材料を配る） ─
    print(f"[yosouya] 材料を収集中: 「{question}」")
    materials = gather_materials(question, research_queries=research_queries)
    materials_summary = materials.get("summary", "")
    materials_available = materials.get("available", False)

    if not materials_available:
        print("[yosouya] 材料取得できず — 材料なしで見積もります（精度は低下します）")
    elif not materials_summary:
        print("[yosouya] 材料取得はできたが結果が空でした")
    else:
        n_sources = len(materials.get("sources", []))
        unconf = materials.get("unconfirmed_count", 0)
        print(f"[yosouya] 材料取得完了: {n_sources}件 (未確認:{unconf}件) / "
              f"クエリ: {materials.get('queries', [])}")

    # ── 3. 独立見積もり（軽量村人を先に並列、重量モデルを後出し） ──
    _HEAVY_MODELS = {"deepseek-r1:32b", "deepseek-r1:70b"}

    light_members = [m for m in members if m.get("model") not in _HEAVY_MODELS]
    heavy_members = [m for m in members if m.get("model") in _HEAVY_MODELS]

    estimates: list[dict] = []

    # 軽い村人を先に並列実行（Ollamaキューを重量モデルに占有させない）
    if light_members:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(light_members)) as ex:
            futs = {ex.submit(_ask_one, m, question, materials_summary, 90.0): m for m in light_members}
            for fut in concurrent.futures.as_completed(futs, timeout=120):
                estimates.append(fut.result())
        print(f"[yosouya] 軽量村人完了: {len(estimates)}体")

    # 重量モデルを後出しで順番に実行（全員分の材料のみ渡す、他の村人の意見は渡さない）
    for m in heavy_members:
        print(f"[yosouya] 重量モデル開始: {m['kote']} ({m['model']})")
        result = _ask_one(m, question, materials_summary, 600.0)
        estimates.append(result)
        print(f"[yosouya] 重量モデル完了: {m['kote']} prob={result.get('prob')}")

    order = {m["kote"]: i for i, m in enumerate(members)}
    estimates.sort(key=lambda e: order.get(e["kote"], 999))

    probs = [e["prob"] for e in estimates if e.get("prob") is not None]
    median = statistics.median(probs) if probs else None
    lo = min(probs) if probs else None
    hi = max(probs) if probs else None

    # ── 4. 市場参照データを取得（採点用・村人には見せない） ────────
    market_ref = _fetch_market_ref(question)
    if market_ref.get("markets"):
        print(f"[yosouya] 市場参照: Polymarket {len(market_ref.get('polymarket', []))}件, "
              f"Kalshi {len(market_ref.get('kalshi', []))}件")

    # ── 5. 台帳に記録 ──────────────────────────────────────────
    entry = {
        "id": "y_" + time.strftime("%Y%m%d%H%M%S"),
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "deadline": deadline or "",
        "materials_available": materials_available,
        "research_queries": research_queries,          # LLMが生成した調べ物リスト
        "search_queries": materials.get("queries", []),
        "layers_used": materials.get("layers_used", []),
        "unconfirmed_count": materials.get("unconfirmed_count", 0),
        "n_sources": len(materials.get("sources", [])),
        "sources": [{"title": s["title"], "url": s["url"]} for s in materials.get("sources", [])],
        # 応答なし（prob=None）も含めて全員を記録する
        "estimates": [
            {"kote": e["kote"], "prob": e["prob"], "reason": e.get("reason", ""),
             "ok": e.get("ok", e.get("prob") is not None)}
            for e in estimates
        ],
        "median": median,
        "min": lo,
        "max": hi,
        "n": len(probs),
        "source": "LA-village",
        "resolved": None,
        # 採点用の市場参照（村人には渡していない）
        "market_ref": market_ref,
    }
    ledger_path = _append_ledger(entry)

    return {
        "ok": True,
        "question": question,
        "median": median,
        "min": lo,
        "max": hi,
        "n": len(probs),
        "estimates": entry["estimates"],
        "materials_available": materials_available,
        "n_sources": entry["n_sources"],
        "market_ref": market_ref,
        "ledger": ledger_path,
        "id": entry["id"],
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "2026年アメリカ中間選挙で下院は民主党が過半数を取る"
    # 直接実行のデモ：判定期限は投票日の5日後（結果確定の目安）
    deadline = sys.argv[2] if len(sys.argv) > 2 else "2026-11-08"
    res = run(q, deadline=deadline)
    print(json.dumps(res, ensure_ascii=False, indent=2))
