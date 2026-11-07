"""
AI_001の「学習」機能。

学習ボタンが押されたら：
  1. 直近の会話・生成失敗例から、router役のAI(lfm2.5)に
     「今AI_001が学ぶべきこと」を推測させ、検索クエリを自分で立てさせる
     （社訓「良い問いを立てよ」の実践）
  2. arXiv・Wikipedia・e-Gov（法令）・Qiita（技術記事）の4ソースから
     無料APIでテキストを取得する。e-GovとQiitaはテーマに関係がある場合のみ
     AIが自分でクエリを立て、関係なければ空文字を返してスキップする。
  3. 取得した生テキストをそのまま shobo_knowledge コレクションに保存する
     （要約・加工はしない。category="learning" として後から区別できるようにする）
  4. 1回の学習サイクルごとに1つのテキストログファイルを
     learning_logs/ に書き出す（クエリ・取得結果・エラー/成功件数を含む）
  5. shobo_knowledge のうち3か月（90日）を超えたデータを削除する
     （無限に肥大化させないため）

ネットワークアクセスについて：
  - arXiv API: https://export.arxiv.org/api/query （無料・無制限・APIキー不要）
  - Wikipedia API: https://ja.wikipedia.org/w/api.php （無料・無制限・APIキー不要）
  - e-Gov法令API: egov_source.py 経由（無料・APIキー不要）
  - Qiita API: https://qiita.com/api/v2/items （無料・認証不要、レート制限あり）
  いずれも取得失敗時は例外を投げず、エラーとして記録し処理を継続する
  （1ソースの失敗で学習サイクル全体を止めないため）。
"""

import os
import re
import time
import json
import egov_source
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from ai import ask_ai
from peer_review import evaluate_with_real_teacher
from memory_store import (
    save_shobo_knowledge, _get_knowledge_col,
    save_medical_knowledge, _get_medical_col,
)

LEARNING_LOGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learning_logs"
)

# ★ 消防の同心円状拡張（建材・空調排煙設備・電気設備・気象災害・医学など）の学習ログは、
# 通常の自己改善学習（learning_logs/直下）とは分けて learning_logs/expansion/ に整理する。
# category が指定された学習サイクルはすべて「拡張学習」とみなし、こちらに書き出す。
EXPANSION_LOG_SUBDIR = "expansion"

# shobo_knowledge に保存したデータを何日で削除するか
KNOWLEDGE_RETENTION_DAYS = 90

ARXIV_API_URL = "https://export.arxiv.org/api/query"
WIKIPEDIA_API_URL = "https://ja.wikipedia.org/w/api.php"
QIITA_API_URL = "https://qiita.com/api/v2/items"

REQUEST_TIMEOUT_SECONDS = 20


# ─────────────────────────────────────────
# ① 検索クエリの立案（「問いを立てる力」の実装）
# ─────────────────────────────────────────

def _build_query_planning_prompt(recent_failures: list, recent_conversation: str, recently_learned: str = "", theme: str = "") -> str:
    failures_text = ""
    if recent_failures:
        lines = []
        for f in recent_failures[:5]:
            lines.append(f"- タスク: {f.get('task', '')[:100]} / 理由: {f.get('reason', '')[:100]}")
        failures_text = "直近の生成失敗例：\n" + "\n".join(lines)
    else:
        failures_text = "直近の生成失敗例：特になし"

    conv_text = recent_conversation.strip() or "（直近の会話ログなし）"

    recently_learned_block = ""
    if recently_learned.strip():
        recently_learned_block = f"""
直前の学習サイクルで学んだこと（同じ内容を繰り返し調べないこと）：
{recently_learned.strip()}
"""

    # ★ テーマ指定ブロック。ユーザーが入力欄にテーマを書いた場合のみ表示する。
    theme_block = ""
    if theme.strip():
        theme_block = f"""
★ ユーザーが指定した学習テーマ（最優先で従うこと）：
{theme.strip()}

arxiv_query と wikipedia_query は、このテーマに沿った内容にすること。
"""

    return f"""
あなたはAI_001という名前のAIエージェントです。社訓は「良い問いを立てよ、自分の答えを評価・選択せよ」です。

これから自分自身の知識を補うために、外部から情報を学習します。
以下の情報をもとに、「今のAI_001に最も必要な知識は何か」を考え、
検索に使うクエリを立ててください。
{theme_block}
{failures_text}

直近の会話の様子：
{conv_text}
{recently_learned_block}
ルール：
- arxiv_query は学術論文検索（arXiv）用の英語キーワード。
  テーマが指定されている場合はそのテーマに関する英語キーワードにすること。
  テーマが指定されていない場合は、AIエージェント設計・コード生成・LLM活用など
  AI_001自身の能力向上に関係するテーマにすること。
  「AI_001」のような固有名詞・プロジェクト名はそのままクエリに使わないこと。
  実在する論文がヒットしやすい、一般的な英語の技術用語にすること。
- wikipedia_query は日本語のWikipedia検索用キーワード。
  テーマが指定されている場合はそのテーマに沿った日本語キーワードにすること。
  テーマが指定されていない場合は、プログラミング・React・TypeScript・
  ソフトウェア設計など一般的な技術知識を補うテーマにすること。
- egov_query は法令名（一部でよい）。テーマがshobo-appの消防・建築法令に
  関連する場合のみ、関連する法令名（例：「消防法」「建築基準法」）を入れること。
  関係ないテーマなら空文字("")にすること。
- qiita_query はQiita検索用のキーワード。テーマがプログラミング・React・
  TypeScript・Electron等の技術力向上に関連する場合のみキーワードを入れること。
  関係ないテーマなら空文字("")にすること。
- いずれも1〜4語程度の短いキーワードにすること。
- ★重要：直前の学習サイクルで使ったarXivクエリと全く同じクエリは使わないこと。
  直前に学んだことがある場合は、それとは違うテーマや角度でクエリを立てること。
- reasonには、なぜそのクエリを選んだのか一言で書くこと。

以下のJSONのみ返してください：
{{
  "arxiv_query": "string",
  "wikipedia_query": "string",
  "egov_query": "string",
  "qiita_query": "string",
  "reason": "string"
}}
""".strip()


def _plan_learning_queries(recent_failures: list, recent_conversation: str, recently_learned: str = "", theme: str = "") -> dict:
    """
    router役のAIに、今学ぶべき検索クエリを自分で立てさせる。
    theme が渡された場合はそのテーマに沿ったクエリを立てさせる。
    失敗した場合は固定のフォールバッククエリを使う（学習サイクル自体は止めない）。
    """
    prompt = _build_query_planning_prompt(recent_failures, recent_conversation, recently_learned, theme)
    try:
        result = ask_ai(prompt, model="router")
        arxiv_query = result.get("arxiv_query") or "AI agent code generation"
        wikipedia_query = result.get("wikipedia_query") or "React"
        egov_query = result.get("egov_query", "")
        qiita_query = result.get("qiita_query", "")
        reason = result.get("reason", "")
        return {
            "arxiv_query": arxiv_query,
            "wikipedia_query": wikipedia_query,
            "egov_query": egov_query,
            "qiita_query": qiita_query,
            "reason": reason,
        }
    except Exception as e:
        print(f"[LEARNING] クエリ立案に失敗しました。フォールバックを使用します: {e}")
        return {
            "arxiv_query": "AI agent code generation",
            "wikipedia_query": "React",
            "egov_query": "",
            "qiita_query": "",
            "reason": f"クエリ立案失敗のためフォールバック（{e}）",
        }


# ─────────────────────────────────────────
# ② 各ソースからの取得
# ─────────────────────────────────────────

def _fetch_arxiv(query: str, max_results: int = 3) -> dict:
    """
    arXiv APIから論文のタイトル・要約を取得する。
    戻り値: {"ok": bool, "items": [{"title":..., "summary":..., "url":...}], "error": str|None}
    """
    try:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        })
        url = f"{ARXIV_API_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "AI_001-learning/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as res:
            raw = res.read()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        items = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary = (summary_el.text or "").strip() if summary_el is not None else ""
            url_ = (id_el.text or "").strip() if id_el is not None else ""
            if title:
                items.append({"title": title, "summary": summary, "url": url_})

        return {"ok": True, "items": items, "error": None}
    except Exception as e:
        print(f"[LEARNING] arXiv取得に失敗しました: {e}")
        return {"ok": False, "items": [], "error": str(e)}


def _fetch_wikipedia(query: str, max_results: int = 2) -> dict:
    """
    Wikipedia(日本語)から検索結果の抜粋（extract）を取得する。
    戻り値: {"ok": bool, "items": [{"title":..., "extract":..., "url":...}], "error": str|None}
    """
    try:
        # ①検索してページタイトルを得る
        search_params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        })
        search_url = f"{WIKIPEDIA_API_URL}?{search_params}"
        req = urllib.request.Request(search_url, headers={"User-Agent": "AI_001-learning/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as res:
            search_data = json.loads(res.read().decode("utf-8"))

        titles = [hit["title"] for hit in search_data.get("query", {}).get("search", [])]
        if not titles:
            return {"ok": True, "items": [], "error": None}

        # ②各ページの要約(extract)を取得
        extract_params = urllib.parse.urlencode({
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": "|".join(titles),
            "format": "json",
        })
        extract_url = f"{WIKIPEDIA_API_URL}?{extract_params}"
        req2 = urllib.request.Request(extract_url, headers={"User-Agent": "AI_001-learning/1.0"})
        with urllib.request.urlopen(req2, timeout=REQUEST_TIMEOUT_SECONDS) as res2:
            extract_data = json.loads(res2.read().decode("utf-8"))

        items = []
        pages = extract_data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            extract = page.get("extract", "")
            if title and extract:
                page_url = "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                items.append({"title": title, "extract": extract, "url": page_url})

        return {"ok": True, "items": items, "error": None}
    except Exception as e:
        print(f"[LEARNING] Wikipedia取得に失敗しました: {e}")
        return {"ok": False, "items": [], "error": str(e)}


def _fetch_qiita(query: str, max_results: int = 3) -> dict:
    """
    Qiita APIから記事のタイトル・本文冒頭を取得する（認証不要、レート制限あり）。
    戻り値: {"ok": bool, "items": [{"title":..., "content":..., "url":...}], "error": str|None}
    """
    try:
        params = urllib.parse.urlencode({"query": query, "per_page": max_results})
        url = f"{QIITA_API_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "AI_001-learning/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as res:
            data = json.loads(res.read().decode("utf-8"))

        items = []
        for entry in data:
            title = entry.get("title", "")
            body = entry.get("body", "") or ""
            url_ = entry.get("url", "")
            if title:
                items.append({"title": title, "content": body[:500], "url": url_})
        return {"ok": True, "items": items, "error": None}
    except Exception as e:
        print(f"[LEARNING] Qiita取得に失敗しました: {e}")
        return {"ok": False, "items": [], "error": str(e)}


# ─────────────────────────────────────────
# ③ 直近の失敗例・会話ログの収集（学習対象の「問い」を立てる材料）
# ─────────────────────────────────────────

def _collect_recent_failures() -> list:
    """
    shobo_knowledge とは別の la_experience コレクションから、
    直近の失敗(outcome=failure)を集める。
    取得に失敗しても空リストを返すだけにして、学習サイクルは止めない。
    """
    try:
        from memory_store import _get_experience_col
        col = _get_experience_col()
        count = col.count()
        if count == 0:
            return []
        # 直近の失敗例を広めに取得するため、空クエリに近い形で全件走査は重いので
        # 直近らしい代表的なキーワードで検索する簡易実装（最初の一歩）
        results = col.query(query_texts=["失敗"], n_results=min(5, count))
        failures = []
        for meta in results.get("metadatas", [[]])[0]:
            if meta.get("outcome") == "failure":
                failures.append({
                    "task": meta.get("task", ""),
                    "reason": meta.get("reflection", "") or meta.get("learned", ""),
                })
        return failures
    except Exception as e:
        print(f"[LEARNING] 直近の失敗例の収集に失敗しました（無視して続行）: {e}")
        return []


# ─────────────────────────────────────────
# ④ ChromaDBへの保存とテキストログ出力
# ─────────────────────────────────────────

def _save_items_to_chroma(items: list, source: str, category: str = "learning", target: str = "shobo") -> int:
    """
    取得したアイテム（arXiv/Wikipedia/e-Gov/Qiita）をそのまま保存する。
    要約・加工はしない（生のタイトル＋本文をそのまま保存）。

    target: "shobo"（デフォルト、shobo_knowledge） | "medical"（medical_knowledge）
    category: 保存時のカテゴリ。拡張学習（建材・空調排煙設備・電気設備・気象災害・医学 等）
      の場合は呼び出し元からカテゴリ名がそのまま渡される。未指定時は従来通り"learning"。
    戻り値: 保存できた件数
    """
    save_fn = save_medical_knowledge if target == "medical" else save_shobo_knowledge
    saved = 0
    for item in items:
        title = item.get("title", "")
        content = item.get("summary") or item.get("extract") or item.get("content") or ""
        if not content.strip():
            continue
        try:
            save_fn(
                content=content,
                title=title,
                source=source,
                category=category,
            )
            saved += 1
        except Exception as e:
            print(f"[LEARNING] 保存に失敗しました（{source} / {title}）: {e}")
    return saved
def _save_items_to_chroma_with_review(items, source, category="learning", target="shobo", theme=""):
    save_fn = save_medical_knowledge if target == "medical" else save_shobo_knowledge
    saved = 0
    review_log = []
    for item in items:
        title = item.get("title", "")
        content = item.get("summary") or item.get("extract") or item.get("content") or ""
        if not content.strip():
            continue
        
        # ★狛犬先生が100点満点で採点
        verdict = evaluate_with_real_teacher(title, content, theme, content[:2000])
        score = verdict.get("score", 0)
        review_log.append(f"{title}: {score}点 - {verdict.get('good','')} / 直し:{verdict.get('fix','')}")
        
        # 90点未満はゴミ箱行き
        if score < 90:
            print(f"[狛犬教室 不採用] {score}点なので保存スキップ: {title[:50]}")
            continue
            
        try:
            save_fn(content=content, title=title, source=source, category=category)
            saved += 1
            print(f"[狛犬教室 採用] {score}点 合格！保存: {title[:50]}")
        except Exception as e:
            print(f"[LEARNING] 保存に失敗しました（{source} / {title}）: {e}")
    return saved, review_log


def _write_learning_log(
    queries: dict,
    arxiv_result: dict,
    wikipedia_result: dict,
    egov_result: dict,
    qiita_result: dict,
    saved_counts: dict,
    category: str = "",
    log_subdir: str = "",
    review_logs: list = None,
) -> str:
    """
    1回の学習サイクルの内容をテキストファイルに書き出す。
    ファイル名: learning_YYYYMMDD_HHMMSS.txt（category指定時は末尾に付与）

    log_subdir: 拡張学習（category指定あり）の場合は EXPANSION_LOG_SUBDIR を渡し、
    learning_logs/expansion/ 配下に書き出す。空文字なら従来通りlearning_logs/直下。
    戻り値: 書き出したファイルのフルパス
    """
    logs_dir = os.path.join(LEARNING_LOGS_DIR, log_subdir) if log_subdir else LEARNING_LOGS_DIR
    os.makedirs(logs_dir, exist_ok=True)

    now = datetime.now()
    suffix = f"_{category}" if category else ""
    filename = f"learning_{now.strftime('%Y%m%d_%H%M%S')}{suffix}.txt"
    filepath = os.path.join(logs_dir, filename)

    lines = []
    lines.append(f"=== AI_001 学習ログ ===")
    lines.append(f"日時: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    if category:
        lines.append(f"カテゴリ（拡張学習）: {category}")
    lines.append("")
    lines.append("--- 検索クエリ（AIが自分で立てた問い） ---")
    lines.append(f"arXivクエリ: {queries.get('arxiv_query', '')}")
    lines.append(f"Wikipediaクエリ: {queries.get('wikipedia_query', '')}")
    lines.append(f"e-Govクエリ: {queries.get('egov_query', '') or '（今回は対象外）'}")
    lines.append(f"Qiitaクエリ: {queries.get('qiita_query', '') or '（今回は対象外）'}")
    lines.append(f"クエリ選定理由: {queries.get('reason', '')}")
    lines.append("")

    lines.append("--- 生徒同士の相互評価ログ ---")
    if review_logs:
        for entry in review_logs:
            lines.append(f"・{entry}")
    else:
        lines.append("（評価対象がありませんでした）")
    lines.append("")

    lines.append("--- arXiv 取得結果 ---")
    if arxiv_result["ok"]:
        if arxiv_result["items"]:
            for item in arxiv_result["items"]:
                lines.append(f"・{item['title']}")
                lines.append(f"  URL: {item['url']}")
                lines.append(f"  要約: {item['summary'][:300]}")
                lines.append("")
        else:
            lines.append("（該当する論文が見つかりませんでした）")
    else:
        lines.append(f"取得失敗: {arxiv_result['error']}")
    lines.append("")

    lines.append("--- Wikipedia 取得結果 ---")
    if wikipedia_result["ok"]:
        if wikipedia_result["items"]:
            for item in wikipedia_result["items"]:
                lines.append(f"・{item['title']}")
                lines.append(f"  URL: {item['url']}")
                lines.append(f"  抜粋: {item['extract'][:300]}")
                lines.append("")
        else:
            lines.append("（該当するページが見つかりませんでした）")
    else:
        lines.append(f"取得失敗: {wikipedia_result['error']}")
    lines.append("")

    lines.append("--- e-Gov（法令） 取得結果 ---")
    if not queries.get("egov_query"):
        lines.append("（今回のテーマでは対象外のためスキップしました）")
    elif egov_result["ok"]:
        if egov_result["items"]:
            for item in egov_result["items"]:
                lines.append(f"・{item['title']}")
                lines.append(f"  本文: {item['content'][:300]}")
                lines.append("")
        else:
            lines.append("（該当する条文が見つかりませんでした）")
    else:
        lines.append(f"取得失敗: {egov_result['error']}")
    lines.append("")

    lines.append("--- Qiita 取得結果 ---")
    if not queries.get("qiita_query"):
        lines.append("（今回のテーマでは対象外のためスキップしました）")
    elif qiita_result["ok"]:
        if qiita_result["items"]:
            for item in qiita_result["items"]:
                lines.append(f"・{item['title']}")
                lines.append(f"  URL: {item['url']}")
                lines.append(f"  本文冒頭: {item['content'][:300]}")
                lines.append("")
        else:
            lines.append("（該当する記事が見つかりませんでした）")
    else:
        lines.append(f"取得失敗: {qiita_result['error']}")
    lines.append("")

    lines.append("--- エラー・成功件数まとめ ---")
    lines.append(f"arXiv: 取得 {'成功' if arxiv_result['ok'] else '失敗'} / 件数 {len(arxiv_result['items'])} / DB保存 {saved_counts.get('arxiv', 0)}件")
    lines.append(f"Wikipedia: 取得 {'成功' if wikipedia_result['ok'] else '失敗'} / 件数 {len(wikipedia_result['items'])} / DB保存 {saved_counts.get('wikipedia', 0)}件")
    lines.append(f"e-Gov: 取得 {'成功' if egov_result['ok'] else '失敗'} / 件数 {len(egov_result['items'])} / DB保存 {saved_counts.get('egov', 0)}件")
    lines.append(f"Qiita: 取得 {'成功' if qiita_result['ok'] else '失敗'} / 件数 {len(qiita_result['items'])} / DB保存 {saved_counts.get('qiita', 0)}件")
    failed_sources = []
    if not arxiv_result["ok"]:
        failed_sources.append("arXiv")
    if not wikipedia_result["ok"]:
        failed_sources.append("Wikipedia")
    if queries.get("egov_query") and not egov_result["ok"]:
        failed_sources.append("e-Gov")
    if queries.get("qiita_query") and not qiita_result["ok"]:
        failed_sources.append("Qiita")
    if failed_sources:
        lines.append(f"失敗したソース: {', '.join(failed_sources)}")
    else:
        lines.append("失敗したソース: なし")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[LEARNING] 学習ログを書き出しました: {filepath}")
    return filepath


# ─────────────────────────────────────────
# ⑤ 3か月（90日）を超えたデータの削除
# ─────────────────────────────────────────

def cleanup_old_logs(retention_days: int = 3) -> int:
    """
    learning_logs フォルダ内のテキストログファイルのうち、
    作成から retention_days 日を超えたものを削除する。
    ChromaDBへの保存は既に完了しているため、ログファイルを消しても
    学習データは失われない。
    戻り値: 削除したファイル数
    """
    try:
        if not os.path.exists(LEARNING_LOGS_DIR):
            return 0
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        for fname in os.listdir(LEARNING_LOGS_DIR):
            if not fname.startswith("learning_") or not fname.endswith(".txt"):
                continue
            fpath = os.path.join(LEARNING_LOGS_DIR, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                deleted += 1
        if deleted:
            print(f"[LEARNING] {retention_days}日を超えたログファイルを{deleted}件削除しました。")
        return deleted
    except Exception as e:
        print(f"[LEARNING] ログファイルの削除に失敗しました（無視して続行）: {e}")
        return 0


def cleanup_old_knowledge(retention_days: int = KNOWLEDGE_RETENTION_DAYS) -> int:
    """
    shobo_knowledge コレクションのうち、保存から retention_days 日を
    超えたデータを削除する。
    戻り値: 削除した件数
    """
    try:
        col = _get_knowledge_col()
        count = col.count()
        if count == 0:
            return 0

        cutoff_timestamp = int((datetime.now() - timedelta(days=retention_days)).timestamp())

        # ChromaDBは大量データのwhere絞り込みgetをサポートしている
        all_data = col.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            ts = meta.get("timestamp", 0)
            if ts and ts < cutoff_timestamp:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            print(f"[LEARNING] {retention_days}日を超えた知識を{len(ids_to_delete)}件削除しました。")

        return len(ids_to_delete)
    except Exception as e:
        print(f"[LEARNING] 古い知識の削除に失敗しました（無視して続行）: {e}")
        return 0


def cleanup_old_medical_knowledge(retention_days: int = KNOWLEDGE_RETENTION_DAYS) -> int:
    """
    medical_knowledge コレクションのうち、保存から retention_days 日を
    超えたデータを削除する（cleanup_old_knowledgeのmedical_knowledge版）。
    戻り値: 削除した件数
    """
    try:
        col = _get_medical_col()
        count = col.count()
        if count == 0:
            return 0

        cutoff_timestamp = int((datetime.now() - timedelta(days=retention_days)).timestamp())

        all_data = col.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
            ts = meta.get("timestamp", 0)
            if ts and ts < cutoff_timestamp:
                ids_to_delete.append(doc_id)

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            print(f"[LEARNING] {retention_days}日を超えた医学知識を{len(ids_to_delete)}件削除しました。")

        return len(ids_to_delete)
    except Exception as e:
        print(f"[LEARNING] 古い医学知識の削除に失敗しました（無視して続行）: {e}")
        return 0


# ─────────────────────────────────────────
# 学習サイクル本体
# ─────────────────────────────────────────

def _run_one_cycle(
    recent_conversation: str,
    generation,
    recently_learned: str = "",
    progress_offset: tuple = (0, 100),
    theme: str = "",
    category: str = "",
    target: str = "shobo",
) -> dict:
    """
    学習サイクル1回分の本体処理。
    run_learning_cycle（単発）と run_continuous_learning_cycle（連続）の
    両方から呼ばれる共通ロジック。

    progress_offset: (start_percent, end_percent) で、連続学習中に
    1サイクルあたりの進捗をその範囲内に収めるための引数。
    単発実行時は (0, 100) のまま使う。

    category: 拡張学習（建材・空調排煙設備・電気設備・気象災害・医学 等）のタグ。
      指定するとログはlearning_logs/expansion/に、保存カテゴリもこの値になる。
      空文字なら従来通り（learning_logs/直下、category="learning"）。
    target: "shobo"（shobo_knowledge） | "medical"（medical_knowledge）。
      保存先コレクションと、古いデータのクリーンアップ対象を決める。
    """
    start_pct, end_pct = progress_offset
    span = end_pct - start_pct

    def _progress(ratio: float, step: str):
        # ratio は0.0〜1.0の、このサイクル内での進み具合
        if generation is not None:
            try:
                import progress as _p
                percent = start_pct + int(span * ratio)
                _p.update(generation, percent, step)
            except Exception:
                pass

    print("[LEARNING] 学習サイクルを開始します")

    _progress(0.1, "学ぶべきことを考えています")
    recent_failures = _collect_recent_failures()
    queries = _plan_learning_queries(recent_failures, recent_conversation, recently_learned, theme)
    print(f"[LEARNING] クエリ: arxiv='{queries['arxiv_query']}' wikipedia='{queries['wikipedia_query']}' egov='{queries.get('egov_query')}' qiita='{queries.get('qiita_query')}' reason='{queries['reason']}'")

    _progress(0.25, f"arXivで「{queries['arxiv_query']}」を検索しています")
    arxiv_result = _fetch_arxiv(queries["arxiv_query"])

    _progress(0.4, f"Wikipediaで「{queries['wikipedia_query']}」を検索しています")
    wikipedia_result = _fetch_wikipedia(queries["wikipedia_query"])

    _progress(0.55, "e-Govで法令を調べています")
    egov_result = {"ok": True, "items": [], "error": None}
    if queries.get("egov_query"):
        law = egov_source.find_law_by_query(queries["egov_query"])
        egov_result = egov_source.fetch_law_articles(law["id"], law["name"])

    _progress(0.65, f"Qiitaで「{queries.get('qiita_query')}」を検索しています")
    qiita_result = {"ok": True, "items": [], "error": None}
    if queries.get("qiita_query"):
        qiita_result = _fetch_qiita(queries["qiita_query"])

    _progress(0.8, "学んだ内容を評価・保存しています")
    save_category = category or "learning"
    review_logs = []

    saved_arxiv, log_arxiv = _save_items_to_chroma_with_review(arxiv_result["items"], source="arxiv", category=save_category, target=target, theme=theme) if arxiv_result["ok"] else (0, [])
    saved_wikipedia, log_wikipedia = _save_items_to_chroma_with_review(wikipedia_result["items"], source="wikipedia", category=save_category, target=target, theme=theme) if wikipedia_result["ok"] else (0, [])
    saved_egov, log_egov = _save_items_to_chroma_with_review(egov_result["items"], source="egov", category=save_category, target=target, theme=theme) if egov_result["ok"] else (0, [])
    saved_qiita, log_qiita = _save_items_to_chroma_with_review(qiita_result["items"], source="qiita", category=save_category, target=target, theme=theme) if qiita_result["ok"] else (0, [])

    saved_counts = {
        "arxiv": saved_arxiv,
        "wikipedia": saved_wikipedia,
        "egov": saved_egov,
        "qiita": saved_qiita,
    }
    review_logs = log_arxiv + log_wikipedia + log_egov + log_qiita

    _progress(0.9, "学習ログを書き出しています")
    log_subdir = EXPANSION_LOG_SUBDIR if category else ""
    log_path = _write_learning_log(queries, arxiv_result, wikipedia_result, egov_result, qiita_result, saved_counts, category=category, log_subdir=log_subdir, review_logs=review_logs)

    _progress(0.97, "古い知識・ログを整理しています")
    deleted_count = cleanup_old_medical_knowledge() if target == "medical" else cleanup_old_knowledge()
    cleanup_old_logs()  # ログファイルは3日で自動削除（ChromaDBへの保存は済んでいるため安全）

    total_saved = saved_counts["arxiv"] + saved_counts["wikipedia"] + saved_counts["egov"] + saved_counts["qiita"]

    return {
        "status": "done",
        "queries": queries,
        "arxiv": {"ok": arxiv_result["ok"], "count": len(arxiv_result["items"]), "saved": saved_counts["arxiv"], "error": arxiv_result["error"], "titles": [i["title"] for i in arxiv_result["items"]]},
        "wikipedia": {"ok": wikipedia_result["ok"], "count": len(wikipedia_result["items"]), "saved": saved_counts["wikipedia"], "error": wikipedia_result["error"], "titles": [i["title"] for i in wikipedia_result["items"]]},
        "egov": {"ok": egov_result["ok"], "count": len(egov_result["items"]), "saved": saved_counts["egov"], "error": egov_result["error"], "titles": [i["title"] for i in egov_result["items"]]},
        "qiita": {"ok": qiita_result["ok"], "count": len(qiita_result["items"]), "saved": saved_counts["qiita"], "error": qiita_result["error"], "titles": [i["title"] for i in qiita_result["items"]]},
        "total_saved": total_saved,
        "deleted_old_count": deleted_count,
        "log_path": log_path,
    }


def _summarize_for_next_cycle(cycle_result: dict) -> str:
    """
    1サイクルの結果から、次のサイクルのクエリ立案に渡す
    「直前に学んだこと」の要約テキストを作る。
    （要約用に別モデルを呼ぶとコストが増えるため、タイトル一覧をそのまま渡す簡易実装）
    """
    parts = []
    arxiv_titles = cycle_result.get("arxiv", {}).get("titles", [])
    wiki_titles = cycle_result.get("wikipedia", {}).get("titles", [])
    egov_titles = cycle_result.get("egov", {}).get("titles", [])
    qiita_titles = cycle_result.get("qiita", {}).get("titles", [])
    if arxiv_titles:
        parts.append("arXiv: " + " / ".join(arxiv_titles[:3]))
    if wiki_titles:
        parts.append("Wikipedia: " + " / ".join(wiki_titles[:3]))
    if egov_titles:
        parts.append("e-Gov: " + " / ".join(egov_titles[:3]))
    if qiita_titles:
        parts.append("Qiita: " + " / ".join(qiita_titles[:3]))
    return "\n".join(parts)


def run_learning_cycle(recent_conversation: str = "", generation: int = None, theme: str = "", category: str = "", target: str = "shobo") -> dict:
    """
    学習ボタンが押されたときに呼ばれるメイン関数（1回だけ実行する版）。
    theme が渡された場合はそのテーマに沿ったクエリを立てる。

    category/target: 建材・空調排煙設備・電気設備・気象災害・医学 等の拡張学習を行う場合に指定する。
    """
    return _run_one_cycle(recent_conversation, generation, recently_learned="", progress_offset=(0, 100), theme=theme, category=category, target=target)


def run_continuous_learning_cycle(generation: int, recent_conversation: str = "", theme: str = "") -> dict:
    """
    「連続」チェックボックスがONの状態で学習ボタンが押されたときに呼ばれる。
    中断ボタン（progress.abort()）が押されるまで、学習サイクルを無限に繰り返す。
    theme が渡された場合は全サイクルを通じてそのテーマで学習する。
    """
    import time
    import progress as _p

    print(f"[LEARNING] 連続学習モードを開始します theme='{theme}'")

    recently_learned = ""
    cycle_count = 0
    last_result = None

    while True:
        if not _p.is_current_generation(generation):
            break
        cycle_count += 1
        _p.update(generation, 0, f"{cycle_count}回目を実行中")
        try:
            result = _run_one_cycle(
                recent_conversation,
                generation,
                recently_learned=recently_learned,
                progress_offset=(0, 100),
                theme=theme,
            )
        except Exception as e:
            print(f"[LEARNING] {cycle_count}回目のサイクルでエラーが発生しました（無視して続行）: {e}")
            time.sleep(30)
            continue

        last_result = result
        last_result["cycle_count"] = cycle_count
        recently_learned = _summarize_for_next_cycle(result)

        if not _p.is_current_generation(generation):
            print(f"[LEARNING] 中断を検知しました（{cycle_count}サイクル実行後）。連続学習を終了します。")
            break

        # ★ サイクル間に休憩を入れる。older hardware前提なのでこれくらい空けたい
        for _ in range(60):  # 60秒待つ間も中断チェックできるようにする
            if not _p.is_current_generation(generation):
                break
            time.sleep(1)

    if last_result is None:
        last_result = {
            "status": "aborted",
            "queries": {"arxiv_query": "", "wikipedia_query": "", "egov_query": "", "qiita_query": "", "reason": ""},
            "arxiv": {"ok": False, "count": 0, "saved": 0, "error": "中断されました", "titles": []},
            "wikipedia": {"ok": False, "count": 0, "saved": 0, "error": "中断されました", "titles": []},
            "egov": {"ok": False, "count": 0, "saved": 0, "error": "中断されました", "titles": []},
            "qiita": {"ok": False, "count": 0, "saved": 0, "error": "中断されました", "titles": []},
            "total_saved": 0,
            "deleted_old_count": 0,
            "log_path": "",
            "cycle_count": 0,
        }

    return last_result