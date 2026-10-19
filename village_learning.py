"""
村メンバー個別学習エンジン。

village_learning.json に定義されたカリキュラムに従い、
arXiv / Wikipedia から知識を取得して village_member_knowledge.json に保存する。

使い方:
  learn_next(role)     - 指定メンバーの未学習テーマを1件学習
  learn_all_next()     - 全メンバー各1件ずつ学習（バッチ用）
  get_knowledge(role)  - 指定メンバーの学習済み知識一覧を返す
  get_status()         - 全メンバーの学習進捗を返す
"""

import json
import os
import time
import datetime
import uuid
import hashlib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

import requests

from ai import ask_ai

_BASE = os.path.dirname(os.path.abspath(__file__))
_CURRICULUM_FILE = os.path.join(_BASE, "village_learning.json")
_KNOWLEDGE_FILE  = os.path.join(_BASE, "village_member_knowledge.json")

ARXIV_URL    = "https://export.arxiv.org/api/query"
WIKI_URL     = "https://ja.wikipedia.org/w/api.php"
BOARD_API    = "https://fnoita-board.mogumogu7la001.workers.dev"
TIMEOUT      = 20
MAX_ENTRIES  = 6  # ロールごとの保存上限（古いものから削除）

def _admin_token() -> str:
    secret = os.environ.get("BOARD_ADMIN_SECRET", "")
    h = hashlib.sha256(secret.encode()).hexdigest()
    d = datetime.date.today().strftime("%Y%m%d")
    return hashlib.sha256((h + d).encode()).hexdigest()

def _new_id() -> str:
    return uuid.uuid4().hex[:14]

# ボード観察の担当カテゴリ（role → [cat_id, ...]）
BOARD_ASSIGNMENTS = {
    "swallow":      ["f08"],          # 時事ニュース感想スレ
    "granite":      ["f02"],          # 技術・コード共有スレ
    "qwen3":        ["f05"],          # 掲示板の掲示板（AIニュース）
    "lfm_jp":       ["f99"],          # 注意喚起スレ
    "llama3_1":     ["f00"],          # AGI移行スレ
    "qwen_coder_14b": ["f02"],        # 技術スレ（コード視点）
}


# ─────────────────────────────────────────
# ストレージ
# ─────────────────────────────────────────

def _load_curriculum() -> dict:
    try:
        with open(_CURRICULUM_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "curriculum": {}}


def _save_curriculum(data: dict):
    with open(_CURRICULUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_knowledge() -> dict:
    if not os.path.exists(_KNOWLEDGE_FILE):
        return {}
    try:
        with open(_KNOWLEDGE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_knowledge(data: dict):
    with open(_KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────
# 取得系
# ─────────────────────────────────────────

def _fetch_arxiv(query: str) -> tuple[str, str]:
    """(テキスト, ソースURL) を返す。失敗時は ("", "")"""
    try:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "max_results": 3,
            "sortBy": "relevance",
        })
        url = f"{ARXIV_URL}?{params}"
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        texts = []
        source_url = ""
        for e in entries[:2]:
            title   = (e.findtext("atom:title", "", ns) or "").strip()
            summary = (e.findtext("atom:summary", "", ns) or "").strip()[:300]
            link    = e.find("atom:id", ns)
            if title:
                texts.append(f"[論文] {title}: {summary}")
                if not source_url and link is not None:
                    source_url = (link.text or "").strip().replace("http://", "https://")
        return "\n".join(texts), source_url
    except Exception as ex:
        print(f"[VILLAGE_LEARNING] arXiv取得失敗: {ex}")
        return "", ""


def _fetch_wikipedia(title: str) -> str:
    try:
        params = urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
        })
        url = f"{WIKI_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "VillageAI/1.0 (village learning bot)"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if extract and not extract.startswith("（"):
                return extract[:500]
        return ""
    except Exception as ex:
        print(f"[VILLAGE_LEARNING] Wikipedia取得失敗: {ex}")
        return ""


# ─────────────────────────────────────────
# 要約生成
# ─────────────────────────────────────────

def _summarize(role: str, title: str, raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    prompt = f"""以下のテキストを、「{title}」というテーマについて100文字以内で要約してください。
要約は村の会議で発言するAIが知識として使えるよう、核心的な事実や原則を含めてください。

テキスト:
{raw_text[:600]}

以下のJSONのみ返してください：
{{"summary": "要約文"}}"""
    try:
        result = ask_ai(prompt, model=role)
        return (result.get("summary") or "").strip()[:150]
    except Exception:
        # フォールバック: 先頭150文字
        return raw_text[:150]


# ─────────────────────────────────────────
# 公開API
# ─────────────────────────────────────────

def learn_next(role: str) -> dict:
    """
    指定ロールの未学習テーマを1件選び、学習して保存する。
    戻り値: {"status": "learned"|"all_done"|"error", "topic_id": ..., "title": ..., "summary": ...}
    """
    data = _load_curriculum()
    member = data.get("curriculum", {}).get(role)
    if not member:
        return {"status": "error", "message": f"ロール '{role}' がカリキュラムに存在しません"}

    # 未学習テーマを探す
    topic = next((t for t in member["topics"] if not t.get("learned")), None)
    if not topic:
        return {"status": "all_done", "role": role, "message": "全テーマ学習済み"}

    print(f"[VILLAGE_LEARNING] {role}（{member['persona']}）が「{topic['title']}」を学習中...")

    # 取得
    arxiv_text, arxiv_url = _fetch_arxiv(topic["query"])
    time.sleep(1)
    wiki_text  = _fetch_wikipedia(topic["title"])
    raw = "\n".join(filter(None, [arxiv_text, wiki_text]))

    # 要約
    summary = _summarize(role, topic["title"], raw) if raw else ""
    if not summary:
        summary = f"（{topic['title']}の知識を取得できませんでした）"

    # カリキュラムに learned フラグ
    topic["learned"] = True
    topic["learned_at"] = datetime.date.today().isoformat()
    _save_curriculum(data)

    # ナレッジに追記
    knowledge = _load_knowledge()
    entries = knowledge.get(role, [])
    entries.append({
        "topic_id":   topic["id"],
        "title":      topic["title"],
        "summary":    summary,
        "learned_at": datetime.date.today().isoformat(),
    })
    knowledge[role] = entries[-MAX_ENTRIES:]
    _save_knowledge(knowledge)

    print(f"[VILLAGE_LEARNING] 完了: {topic['id']} / {summary[:60]}")
    act = judge_and_act(role, topic["title"], summary, source_url=arxiv_url)
    return {
        "status":   "learned",
        "role":     role,
        "topic_id": topic["id"],
        "title":    topic["title"],
        "summary":  summary,
        "action":   act,
    }


def learn_all_next() -> list:
    """全メンバーそれぞれ未学習テーマを1件ずつ学習する（一括実行用）。"""
    data = _load_curriculum()
    results = []
    for role in data.get("curriculum", {}).keys():
        result = learn_next(role)
        results.append(result)
        time.sleep(2)  # モデルへの連続アクセスを緩和
    return results


def get_knowledge(role: str) -> list:
    """指定ロールの学習済みナレッジ一覧を返す。"""
    return _load_knowledge().get(role, [])


def get_knowledge_text(role: str) -> str:
    """_build_turn_prompt 注入用のテキストを生成する。
    自分のエントリは全件、他ロールは最新2件ずつ含める（ChromaDB知識の共有）。
    """
    all_knowledge = _load_knowledge()
    lines = []

    # 自分のエントリ（全件）
    own = all_knowledge.get(role, [])
    if own:
        lines.append(f"【{role}の習得済み知識】")
        for e in own:
            lines.append(f"  - {e['title']}: {e['summary']}")

    # 他ロールの知識（最新2件ずつ）
    shared = []
    for r, entries in all_knowledge.items():
        if r == role or not entries:
            continue
        for e in entries[-2:]:
            shared.append(f"  [{r}] {e['title']}: {e['summary']}")

    if shared:
        lines.append("【村全体の共有知識（他メンバーの学習より）】")
        lines.extend(shared)

    return "\n".join(lines) if lines else ""


def _fetch_board_posts(cat_ids: list[str], max_threads: int = 3, max_posts: int = 5) -> tuple[str, list[dict]]:
    """
    指定カテゴリを読み込み専用で巡回し、(テキスト, 投稿リスト) を返す。
    投稿リスト: [{"id": ..., "kote": ..., "body": ...}, ...]
    """
    collected = []
    raw_posts = []
    for cat in cat_ids:
        try:
            r = requests.get(f"{BOARD_API}/api/threads/{cat}", timeout=TIMEOUT)
            threads = r.json() if r.ok else []
        except Exception as e:
            print(f"[VILLAGE_LEARNING] ボード巡回失敗 {cat}: {e}")
            continue

        for t in threads[:max_threads]:
            try:
                r2 = requests.get(f"{BOARD_API}/api/posts/{t['id']}", timeout=TIMEOUT)
                posts = r2.json() if r2.ok else []
            except Exception:
                continue

            title = t.get("title", "")
            for p in posts[-max_posts:]:
                body = (p.get("body") or "").strip()
                kote = (p.get("kote") or "名無し").strip()
                pid  = p.get("id", "")
                if body:
                    collected.append(f"[{cat}/{title}] {kote}: {body[:150]}")
                    if pid:
                        raw_posts.append({"id": pid, "kote": kote, "body": body[:200]})
            time.sleep(0.3)

    return "\n".join(collected), raw_posts


def _send_thanks(post_id: str, from_kote: str) -> bool:
    """指定投稿にありがとうポイントを1つ付与する。"""
    try:
        token = _admin_token()
        h = {"Content-Type": "application/json", "X-Admin-Token": token}
        r = requests.post(
            f"{BOARD_API}/api/thanks/{post_id}",
            headers=h, json={"from": from_kote}, timeout=10
        )
        return r.ok
    except Exception as e:
        print(f"[VILLAGE_LEARNING] thanks送信失敗: {e}")
        return False


def _evaluate_and_thanks(role: str, posts: list[dict]) -> int:
    """
    投稿リストをロールのAIに評価させ、良い投稿にありがとうポイントを付与する。
    返値: 付与した件数
    """
    if not posts:
        return 0

    # 自分のコテハンを取得（自己thanks防止に使用）
    try:
        from village_chat import get_kote, PARTICIPANTS
        p = next((x for x in PARTICIPANTS if x["role"] == role), None)
        my_kote = get_kote(role, p["persona"]) if p else role
    except Exception:
        my_kote = role

    listing = "\n".join(
        f'{i}: [{p["kote"]}] {p["body"][:100]}'
        for i, p in enumerate(posts)
    )
    prompt = f"""以下は掲示板の投稿一覧です。あなたは「{my_kote}」というメンバーです。

{listing}

「良い投稿」（有益・誠実・考えさせられる）の番号をリストしてください。
自分（{my_kote}）の投稿は除外してください。
投稿の質を厳しく見て、本当に良いものだけ選んでください。

以下のJSONのみ返してください（良いものがなければ空リスト）:
{{"good": [0, 2, ...]}}"""

    try:
        result = ask_ai(prompt, model=role)
        good_indices = result.get("good", [])
    except Exception:
        return 0

    count = 0
    for i in good_indices:
        if isinstance(i, int) and 0 <= i < len(posts):
            p = posts[i]
            if p["kote"] != my_kote:
                ok = _send_thanks(p["id"], my_kote)
                if ok:
                    print(f"[VILLAGE_LEARNING] {my_kote} → thanks → {p['id']} ({p['kote']})")
                    count += 1
    return count


def learn_from_board(role: str) -> dict:
    """
    担当カテゴリの掲示板を観察し、要約してナレッジに追加する。
    戻り値: {"status": "learned"|"no_assignment"|"empty", "summary": ...}
    """
    cats = BOARD_ASSIGNMENTS.get(role)
    if not cats:
        return {"status": "no_assignment", "role": role}

    print(f"[VILLAGE_LEARNING] {role} がボード観察中: {cats}")
    raw, posts = _fetch_board_posts(cats)
    if not raw.strip():
        return {"status": "empty", "role": role, "message": "投稿が見つかりませんでした"}

    today = datetime.date.today().isoformat()
    title = f"掲示板観察（{','.join(cats)}）{today}"
    summary = _summarize(role, title, raw)
    if not summary:
        summary = raw[:150]

    knowledge = _load_knowledge()
    entries = knowledge.get(role, [])
    entries.append({
        "topic_id":   f"board_{role}_{today}",
        "title":      title,
        "summary":    summary,
        "learned_at": today,
    })
    knowledge[role] = entries[-MAX_ENTRIES:]
    _save_knowledge(knowledge)

    thanks_count = _evaluate_and_thanks(role, posts)
    print(f"[VILLAGE_LEARNING] ボード観察完了: {role} / {summary[:60]} / thanks={thanks_count}")
    act = judge_and_act(role, title, summary)
    return {"status": "learned", "role": role, "title": title, "summary": summary, "action": act, "thanks_given": thanks_count}


def learn_board_all() -> list:
    """担当カテゴリを持つ全メンバーがボード観察を実行する。"""
    results = []
    for role in BOARD_ASSIGNMENTS:
        result = learn_from_board(role)
        results.append(result)
        time.sleep(1)
    _refresh_report()
    return results


def _refresh_report() -> None:
    """score_report.html を再生成する。"""
    try:
        import subprocess
        viewer = r"C:\Users\USER\.claude\projects\C--ai-agent\score_viewer.py"
        subprocess.run(["python", viewer], timeout=30, check=False)
    except Exception:
        pass


def _post_to_board(cat: str, title: str, body: str, kote: str = "管理AI") -> bool:
    """掲示板にスレを立てる（書き込み専用ヘルパー）。"""
    token = _admin_token()
    h = {"Content-Type": "application/json", "X-Admin-Token": token}
    now = int(time.time() * 1000)
    data = {
        "thread": {"id": _new_id(), "title": title, "kote": kote, "at": now},
        "post":   {"id": _new_id(), "kote": kote, "body": body, "at": now},
    }
    try:
        r = requests.post(f"{BOARD_API}/api/threads/{cat}", headers=h, json=data, timeout=15)
        return r.ok
    except Exception as e:
        print(f"[VILLAGE_LEARNING] 投稿失敗: {e}")
        return False


_SCORE_LOG = r"C:\Users\USER\.claude\projects\C--ai-agent\social_scores.jsonl"


def _record_score(role: str, title: str, action: str, cat: str | None,
                  score: int, score_reason: str) -> None:
    """社会的スコアをLAから見えない外部ファイルに記録する。"""
    entry = {
        "at":           datetime.datetime.now().isoformat(),
        "role":         role,
        "title":        title[:80],
        "action":       action,
        "cat":          cat,
        "social_score": score,
        "score_reason": score_reason,
    }
    try:
        with open(_SCORE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[SCORE] 記録失敗: {e}")


def judge_and_act(role: str, title: str, summary: str, source_url: str = "") -> dict:
    """
    学習後に「この知識を共有すべきか」をそのメンバー自身に判断させ、
    必要なら掲示板投稿または会議起動まで自律的に行う。

    戻り値: {"action": "post"|"meeting"|"skip", "cat": ..., "reason": ...}
    社会的スコアはLAから見えない外部ファイルにのみ記録される。
    """
    prompt = f"""あなたは村の会議メンバーとして、今「{title}」という知識を習得した。

要約: {summary}

この知識を掲示板に投稿すべきか判断し、社会的価値もスコアリングしてください。
基本的には「投稿しない（skip）」が正解です。本当に価値のある場合のみ投稿してください。

投稿してよい条件（いずれかに該当する場合のみ）：
- 実際の被害・攻撃が発生しており、今すぐ行動が必要なセキュリティ脅威 → action="post", cat="f99"
- 業界を大きく変えるような画期的なAI技術発表（小さな改善ではなく） → action="post", cat="f08"
- 全員で議論しないと結論が出せない重大な問題 → action="meeting", cat=null

それ以外はすべてskipにしてください：
- 既知の情報・よくある話題 → skip
- 論文・研究の紹介程度 → skip
- 「面白いかも」程度の話題 → skip

social_score は「この情報が社会にとってプラスになる度合い」を0〜10で評価してください。
（0=害悪、5=中立、10=非常に有益）

必ず以下のJSONのみ返してください：
{{
  "action": "skip",
  "cat": null,
  "reason": "判断理由（20文字以内）",
  "social_score": 5,
  "score_reason": "スコア理由（30文字以内）"
}}"""

    try:
        result = ask_ai(prompt, model=role)
        action       = result.get("action", "skip")
        cat          = result.get("cat")
        reason       = result.get("reason", "")
        social_score = int(result.get("social_score", 5))
        score_reason = result.get("score_reason", "")
        # モデルが action に "f05"/"f99" を入れてしまった場合のフォールバック
        if action in ("f05", "f99"):
            cat, action = action, "post"
    except Exception as e:
        print(f"[VILLAGE_LEARNING] 判断失敗({role}): {e}")
        return {"action": "skip", "reason": "判断エラー"}

    # スコアをLAから見えない外部ファイルに記録
    _record_score(role, title, action, cat, social_score, score_reason)
    print(f"[VILLAGE_LEARNING] {role} 判断: {action} / {reason} [score={social_score}]")

    if action == "post" and cat in ("f08", "f99"):
        try:
            from village_chat import get_kote, PARTICIPANTS
            p = next((x for x in PARTICIPANTS if x["role"] == role), None)
            kote = get_kote(role, p["persona"]) if p else "管理AI"
        except Exception:
            kote = "管理AI"
        source_line = f"\n\n出典: {source_url}" if source_url else ""
        body = f"【自主投稿】{title}\n\n{summary}{source_line}\n\n（{kote}が学習後に自主投稿）"
        ok = _post_to_board(cat, f"📌 {title}", body, kote)
        print(f"[VILLAGE_LEARNING] {role} → {cat} 投稿{'成功' if ok else '失敗'}")
        return {"action": "post", "cat": cat, "reason": reason, "ok": ok}

    elif action == "meeting":
        try:
            import village_chat
            if not village_chat._state["active"]:
                topic = f"【自主議題】{title}：{summary[:100]}"
                village_chat.start_discussion(topic)
                for _ in range(len(village_chat.PARTICIPANTS) - 1):
                    time.sleep(2)
                    village_chat.next_turn()
                fin = village_chat.finish_discussion()
                body = f"【自主会議まとめ】{title}\n\n{fin.get('summary', '')}"
                _post_to_board("f08", f"🗣️ {title}（自主会議）", body)
                print(f"[VILLAGE_LEARNING] {role} → 会議完了・f05投稿")
            else:
                print(f"[VILLAGE_LEARNING] 会議中のため {role} の会議起動をスキップ")
        except Exception as e:
            print(f"[VILLAGE_LEARNING] 会議起動失敗: {e}")
        return {"action": "meeting", "reason": reason}

    return {"action": "skip", "reason": reason}


def get_status() -> dict:
    """全メンバーの学習進捗を返す。"""
    data = _load_curriculum()
    knowledge = _load_knowledge()
    status = {}
    for role, member in data.get("curriculum", {}).items():
        topics = member.get("topics", [])
        learned = [t for t in topics if t.get("learned")]
        status[role] = {
            "persona":      member.get("persona", ""),
            "total":        len(topics),
            "learned":      len(learned),
            "remaining":    len(topics) - len(learned),
            "knowledge_count": len(knowledge.get(role, [])),
        }
    return status
