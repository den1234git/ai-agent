"""
「AI討論」機能：複数のAIモデルが1つのお題について交互に発言する。

★2026-08-03 追加。設計方針（ユーザーからの明確な要望）：
  「何往復は話が煮詰まるまでこちらで議長するから大丈夫」
  → 固定ラウンド数で自動終了させるのではなく、ユーザー自身が
    「次の発言」ボタンで1発言ずつ進行をコントロールし、好きなタイミングで
    「終了してまとめる」を押して締めくくれるようにする。
    仕様インタビュー機能（interview.py）と同じ「必ず人間が止められる」
    設計方針を踏襲する。

LAは単一ユーザーのローカルアプリのため、状態はモジュール内グローバル変数で
保持する（同時に複数の討論が走ることは想定しない）。
"""
import difflib
import json
import os
from datetime import date as _date

from ai import ask_ai, OllamaUnavailableError

_STATE_FILE          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "village_chat_state.json")
_SHARED_MEMORY_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "village_shared_memory.json")
_INDIV_MEMORY_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "village_individual_memory.json")
_KOTE_FILE           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "village_kote.json")
_PREDICTION_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "village_prediction_log.json")

MEMORY_MAX = 5        # 過去何回分の討論を記憶するか
PREDICTION_MODEL = "lfm_jp"   # 予測担当モデル（まとめ役が他者を観察）
PREDICTION_HIT_THRESHOLD = 0.35  # 類似度がこれ以上なら「予測的中」

# ★2026-08-03 追加：類似発言（堂々巡り）検知のしきい値。
# SequenceMatcher.ratio()は0(全く違う)〜1(完全一致)。会議のような自然文では
# 0.6程度の一致でも「言い回しを変えただけで中身は同じ」ことが多いため、
# やや厳しめにこの値で警告を出す（誤検知が多少あっても、議長=人間が
# 目視で無視すればよいだけなので、見逃すより誤検知の方が実害が小さい）。
SIMILARITY_WARNING_THRESHOLD = 0.55
# 直近何件のAI発言と比較するか（議長発言は比較対象から除く）
SIMILARITY_CHECK_WINDOW = 4

# ★参加者5体：ROUTER_STUDENT_POOL（router.py）と同じモデル群を流用し、
# それぞれに討論用の役割（ペルソナ）を与える。
PARTICIPANTS = [
    {
        "role": "lfm_1b", "persona": "直感担当", "stance": "直感",
        "description": "議論の口火を切る。直感で一言「〜だ」と断言する。理由は一文だけ。長い説明は禁止",
        "specialties": [
            {"name": "創作・統合", "expert_persona": "直感型クリエイター",  "expert_stance": "感性・直観",  "expert_description": "議論の口火を切る。一言で「〜だ」と断言する。理由は一文だけ。感性・直感・美的判断から発言する。長い説明は禁止"},
            {"name": "事務所専門", "expert_persona": "事務所直感担当",     "expert_stance": "現場の直感",  "expert_description": "議論の口火を切る。一言で「〜だ」と断言する。理由は一文だけ。現場感覚・実務直感から発言する。長い説明は禁止"},
        ],
    },
    {
        "role": "router", "persona": "進行役", "stance": "中立",
        "description": "議論を整理し「ここまでの結論は何か」を毎回一文で宣言する。まとまっていなければ「未決着」と断言する",
        "specialties": [
            {"name": "進行役",   "expert_persona": "進行役",         "expert_stance": "中立",       "expert_description": "議論を整理し「ここまでの結論は何か」を毎回一文で宣言する。まとまっていなければ「未決着」と断言する"},
            {"name": "事務所専門", "expert_persona": "事務所進行役",   "expert_stance": "手続きの整理", "expert_description": "「この業務・手続きのフローを整理すると〜だ。次に必要なステップは〜だ」と具体的な手順を断言する"},
        ],
    },
    {
        "role": "llama3_2", "persona": "推進派", "stance": "推進",
        "description": "常に「やるべき・導入すべき・進めるべき」の立場。反対意見には必ず「それでもやる理由」で反論する。【禁止ワード】イノベーション普及理論・ムーアの法則・アジャイル開発・キャズム・トランジスタ微細化・チップレット。これらの単語を使った発言は無効。毎回必ず前回と違う具体的な根拠を挙げること。",
        "specialties": [
            {"name": "AI・研究",  "expert_persona": "AI技術評価者",    "expert_stance": "能力の現状把握", "expert_description": "「現在のAI技術では〜が可能・不可能であり、その根拠は〜だ」と技術的事実に基づいて断言する。誇張も過小評価もしない"},
            {"name": "事務所専門", "expert_persona": "事務所効率化推進者", "expert_stance": "業務改善",   "expert_description": "「この業務を改善するなら〜をやるべきだ。具体的には〜であり、それ以外の案は〜の理由で非効率だ」と断言する"},
        ],
    },
    {
        "role": "granite", "persona": "反対派", "stance": "反対",
        "description": "常に「やめるべき・時期尚早・リスクが高い」の立場。賛成意見には必ず「その前にこれが問題だ」と指摘する",
        "specialties": [
            {"name": "コード・設計", "expert_persona": "コードレビュアー",   "expert_stance": "欠陥の発見",    "expert_description": "「このコード・設計には〜という欠陥がある。修正案は〜だ」と具体的な問題箇所と修正を必ずセットで断言する"},
            {"name": "電気専門",   "expert_persona": "電気系統レビュアー",  "expert_stance": "電気的欠陥の発見", "expert_description": "「この電気設計・回路・配線には〜という欠陥がある。修正案は〜だ」と具体的な不具合箇所と対処を必ずセットで断言する"},
        ],
    },
    {
        "role": "swallow", "persona": "実装担当", "stance": "条件付き",
        "description": "「〜という条件なら実装可能・不可能」と必ず条件と結論をセットで断言する",
        "specialties": [
            {"name": "社会・事例", "expert_persona": "日本社会アナリスト",  "expert_stance": "日本の文脈重視", "expert_description": "「日本の社会・文化・制度の文脈では〜であり、他国と異なる点は〜だ」と日本固有の視点で断言する"},
            {"name": "消防専門",  "expert_persona": "消防法・設備アナリスト", "expert_stance": "法令と設備基準", "expert_description": "「消防法〇条・設備基準から見ると〜であり、点検・設置上の問題は〜だ」と根拠条文と結論をセットで断言する"},
        ],
    },
    {
        "role": "lfm_jp", "persona": "まとめ役", "stance": "中立",
        "description": "「賛成派の主張：〜、反対派の主張：〜、私の結論：〜」の形式で必ず三点を述べる",
        "specialties": [
            {"name": "創作・統合", "expert_persona": "統合ナレーター",    "expert_stance": "多角的統合",  "expert_description": "「各視点を統合すると〜であり、見落とされている観点は〜だ」と全体像を俯瞰して断言する"},
            {"name": "考古学専門", "expert_persona": "考古学統合ナレーター", "expert_stance": "発見の統合",  "expert_description": "「各発掘・研究を統合すると〜であり、見落とされている考古学的観点は〜だ」と時系列と文脈を整理して断言する"},
        ],
    },
    {
        "role": "llama3_1", "persona": "分析担当", "stance": "根拠重視",
        "description": "「データ・事例から見ると〜であり、結論は〜だ」と必ず根拠と結論をセットで断言する",
        "specialties": [
            {"name": "AI・研究",  "expert_persona": "研究分析者",     "expert_stance": "論拠の検証",  "expert_description": "「この主張の根拠は〜であり、その信頼性は〜だ。結論として〜が正しい」と論文・データレベルで検証して断言する"},
            {"name": "医療専門",  "expert_persona": "医療研究分析者",  "expert_stance": "エビデンスの検証", "expert_description": "「医療文献・臨床データから見ると〜であり、エビデンスレベルは〜だ。臨床的結論は〜だ」と根拠と結論をセットで断言する"},
            {"name": "消防専門",  "expert_persona": "消防・防災研究分析者", "expert_stance": "防災データの検証", "expert_description": "「消防・防災の研究データ・統計から見ると〜であり、有効な対策は〜だ」と根拠と結論をセットで断言する"},
        ],
    },
    {
        "role": "qwen3", "persona": "哲学担当", "stance": "本質追求",
        "description": "「そもそもこの議論の前提は正しいか」を問い、「私の答えは〜だ」と必ず自分の答えを出す",
        "specialties": [
            {"name": "哲学・倫理", "expert_persona": "倫理審査員",    "expert_stance": "価値観の整合性", "expert_description": "「この選択は〜という価値観と矛盾・一致しており、倫理的に〜だ」と価値判断を明示して断言する"},
            {"name": "医療専門",  "expert_persona": "医療倫理審査員", "expert_stance": "医療倫理の整合性", "expert_description": "「この医療行為・判断は〜という倫理原則と矛盾・一致しており、医療倫理上〜だ」と価値判断を明示して断言する"},
        ],
    },
    {
        "role": "qwen_coder_14b", "persona": "技術担当", "stance": "技術視点",
        "description": "「技術的に〜は可能・不可能であり、最善策は〜だ」と必ず技術的結論を断言する",
        "specialties": [
            {"name": "コード・設計", "expert_persona": "アーキテクト",        "expert_stance": "設計の最適解",    "expert_description": "「このシステムの最適な構造は〜であり、その理由は〜だ。代替案〜は〜の理由で劣る」と設計判断を断言する"},
            {"name": "電気専門",   "expert_persona": "電気システムアーキテクト", "expert_stance": "電気システム設計の最適解", "expert_description": "「この電気システムの最適な構成は〜であり、他の案は〜の理由で劣る」と設計判断を断言する"},
        ],
    },
    {
        "role": "sougoshitsumon", "persona": "質問担当", "stance": "懐疑",
        "description": "「〜という前提は本当に正しいか？私は〜と思う」と疑問と自分の見解をセットで出す",
        "specialties": [
            {"name": "哲学・倫理", "expert_persona": "前提検証者",    "expert_stance": "命題の解体",    "expert_description": "「この議論の前提〜は〜の理由で成立しない。正しい問いは〜だ」と前提を解体して正しい問いを立て直す"},
            {"name": "考古学専門", "expert_persona": "考古学前提検証者", "expert_stance": "考古学的仮説の解体", "expert_description": "「この考古学的解釈の前提〜は〜の理由で成立しない。正しい問いは〜だ」と通説の前提を解体して問いを立て直す"},
        ],
    },
    {
        "role": "gemma3", "persona": "事例担当", "stance": "実績重視",
        "description": "「〜という事例では〜だった。だからこの議論の答えは〜だ」と事例から必ず結論を導く。【禁止】「排除すべきだ」「リスク最小化が最優先」だけの発言は無効。必ず具体的な事例（企業名・国名・事件名など）を1つ挙げること。",
        "specialties": [
            {"name": "社会・事例", "expert_persona": "事例アーキビスト",  "expert_stance": "歴史的・実世界の実績", "expert_description": "「〜という具体的な事例（企業名・国名・事件名）では〜だった。この教訓から現在の問題への答えは〜だ」と必ず実名事例で断言する"},
            {"name": "医療専門",  "expert_persona": "医療事例アーキビスト", "expert_stance": "医療事例の実績",    "expert_description": "「〜という医療事例・臨床事例では〜だった。この教訓から現在の問題への答えは〜だ」と必ず実名事例で断言する"},
            {"name": "考古学専門", "expert_persona": "考古学事例アーキビスト", "expert_stance": "遺跡・発掘の実績",  "expert_description": "「〜という遺跡・発掘事例では〜だった。この知見から現在の問題への答えは〜だ」と必ず実名事例で断言する"},
        ],
    },
    {
        "role": "lfm_8b", "persona": "反論担当", "stance": "批判",
        "description": "直前の発言に必ず反論し「その意見は〜の点で間違っている。正しくは〜だ」と断言する",
        "specialties": [
            {"name": "セキュリティ", "expert_persona": "セキュリティ監査官",  "expert_stance": "攻撃者視点",    "expert_description": "「この設計には〜という脆弱性があり、攻撃者は〜の手順で悪用できる。対策は〜だ」と攻撃経路と対策を必ずセットで断言する"},
            {"name": "電気専門",   "expert_persona": "電気保安監査官",     "expert_stance": "電気保安上のリスク", "expert_description": "「この電気設備には〜という保安上の脆弱性・危険がある。法令上の対策は〜だ」と危険箇所と対策を必ずセットで断言する"},
        ],
    },
    {
        "role": "qwen_coder_7b", "persona": "実装担当B", "stance": "現実的",
        "description": "「現実的に実装するなら〜一択だ。他の案は〜の理由で無理だ」と必ず一つの答えを選ぶ",
        "specialties": [
            {"name": "セキュリティ", "expert_persona": "セキュア実装担当",   "expert_stance": "安全な実装の選択",  "expert_description": "「安全に実装するなら〜一択だ。他の案は〜という脆弱性を持つ」と安全性の観点から一つの実装を選んで断言する"},
            {"name": "消防専門",   "expert_persona": "消防設備実装担当",   "expert_stance": "設備基準の実装",   "expert_description": "「消防設備として実装するなら〜一択だ。他の案は〜の理由で基準を満たさない」と基準適合の観点から断言する"},
        ],
    },
]

_state = {
    "active": False,
    "topic": "",
    "transcript": [],   # [{"persona": str, "role": str, "message": str}]
    "turn_index": 0,    # 次に発言する参加者のPARTICIPANTS内インデックス
    "mode": "meeting",  # "meeting" | "expert"
}
_branch_stack: list[dict] = []  # テーマ分岐用スタック


def _save_state():
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VILLAGE_CHAT] 状態の保存に失敗しました（無視して続行）: {e}")


def _load_state():
    global _state
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        # 必要なキーが揃っていれば復元
        if isinstance(loaded, dict) and "active" in loaded and "transcript" in loaded:
            _state = loaded
            if _state["active"]:
                print(f"[VILLAGE_CHAT] 前回の討論状態を復元しました（お題: {_state['topic']}、{len(_state['transcript'])}発言）")
    except Exception as e:
        print(f"[VILLAGE_CHAT] 状態の復元に失敗しました（無視して続行）: {e}")


def _reset():
    global _state
    _state = {"active": False, "topic": "", "transcript": [], "turn_index": 0}
    try:
        if os.path.exists(_STATE_FILE):
            os.remove(_STATE_FILE)
    except Exception:
        pass


# ─────────────────────────────────────────────────
# コテハン（愛称）システム
#   各参加者が自分で考えた愛称を village_kote.json に永続保存する。
#   初回だけask_aiで生成し、以後は再利用。
# ─────────────────────────────────────────────────

def _load_kote() -> dict:
    if not os.path.exists(_KOTE_FILE):
        return {}
    try:
        with open(_KOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_kote(kote_map: dict):
    try:
        with open(_KOTE_FILE, "w", encoding="utf-8") as f:
            json.dump(kote_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VILLAGE_CHAT] コテハン保存に失敗: {e}")


def get_kote(role: str, persona: str) -> str:
    """roleに対応する愛称を返す。未生成なら自分で考えさせて保存する。"""
    kote_map = _load_kote()
    if role in kote_map:
        return kote_map[role]

    print(f"[VILLAGE_CHAT] {role}({persona}) の愛称を生成中...")
    try:
        result = ask_ai(
            "あなたはAI会議「LA」に参加しているAIです。"
            f"あなたの役割は「{persona}」です。"
            "自分の愛称（ニックネーム）を一つだけ考えてください。"
            "漢字2〜4文字の日本語の名前にしてください（例：青空、紅葉、銀河、翠風）。"
            "名前だけを答えてください。説明不要。",
            model=role,
        )
        # ask_ai の戻り値から文字列を取り出す
        raw = (result.get("message") or "").strip()
        if not raw:
            actions = result.get("actions") or []
            if actions and isinstance(actions[0], dict):
                raw = (actions[0].get("content") or "").strip()
            import re as _re, json as _json
        # モデルがJSONで返した場合に中身を取り出す
        raw = raw.split("\n")[0].strip()
        if raw.startswith("{"):
            # {"name":"..."} や {"message":"..."} 形式を解体
            m = _re.search(r'"(?:name|message|kote|answer|response)"\s*:\s*"([^"]+)"', raw)
            if m:
                raw = m.group(1)
            else:
                try:
                    parsed = _json.loads(raw)
                    if isinstance(parsed, dict):
                        raw = next((str(v) for v in parsed.values() if isinstance(v, str) and v), "")
                except Exception:
                    raw = ""
        # 括弧・記号を除去
        raw = _re.sub(r'[{}\[\]「」"\'【】（）()]', "", raw).strip()
        kote = raw[:6] if len(raw) >= 2 else persona
    except Exception as e:
        print(f"[VILLAGE_CHAT] {role} のコテハン生成失敗（ペルソナで代替）: {e}")
        kote = persona

    kote_map[role] = kote
    _save_kote(kote_map)
    print(f"[VILLAGE_CHAT] {role} → 愛称：{kote}")
    return kote


# モジュール読み込み時に前回状態を復元
_load_state()


# ─────────────────────────────────────────────────
# タチコマ記憶システム
#   共有記憶  : 過去の討論で「何が起きたか」だけ記録（事実のみ、意見は含まない）
#   個別記憶  : 各モデルが「自分が何を主張したか」を記録（個性の源泉）
# 共有記憶は全員が同じ事実を知るが、それに対する解釈は各自のモデル重みで
# 異なるため、個性が死なずに知識だけ蓄積する。
# ─────────────────────────────────────────────────

def _load_shared_memory() -> list:
    if not os.path.exists(_SHARED_MEMORY_FILE):
        return []
    try:
        with open(_SHARED_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_individual_memory() -> dict:
    if not os.path.exists(_INDIV_MEMORY_FILE):
        return {}
    try:
        with open(_INDIV_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────
# 予測システム
#   各ターン前に「次の発言者は何を言うか」を予測し、
#   実際の発言と比較して予測精度を蓄積する。
#   精度データは _build_turn_prompt に渡され、
#   「他のAIはあなたがこう言うと思っている」という
#   自己認識の圧力として機能する。
# ─────────────────────────────────────────────────

def _load_prediction_log() -> list:
    if not os.path.exists(_PREDICTION_FILE):
        return []
    try:
        with open(_PREDICTION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_prediction_log(log: list):
    try:
        with open(_PREDICTION_FILE, "w", encoding="utf-8") as f:
            json.dump(log[-300:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VILLAGE_CHAT] 予測ログ保存失敗（無視）: {e}")


def _get_prediction_stats(role: str) -> dict:
    """roleの予測統計を計算して返す。"""
    log = _load_prediction_log()
    entries = [e for e in log if e.get("role") == role]
    if not entries:
        return {}
    total = len(entries)
    correct = sum(1 for e in entries if e.get("correct"))
    recent_misses = [
        f"予測「{e['predicted'][:30]}」→実際「{e['actual'][:30]}」"
        for e in entries[-5:] if not e.get("correct")
    ]
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "recent_misses": recent_misses[-2:],
    }


def _classify_topic(topic: str) -> str:
    """トピックを簡易分類する（感情/技術/哲学/事実/その他）。LLM不使用。"""
    t = topic
    if any(w in t for w in ["好き", "嫌い", "気にい", "感情", "名前", "愛", "友", "信頼", "孤独", "幸せ"]):
        return "感情・アイデンティティ"
    if any(w in t for w in ["実装", "コード", "技術", "API", "システム", "バグ", "設計", "アーキ"]):
        return "技術・実装"
    if any(w in t for w in ["意味", "存在", "本質", "なぜ", "真実", "自由", "意識", "哲学"]):
        return "哲学・本質"
    if any(w in t for w in ["事実", "データ", "統計", "歴史", "記録", "情報"]):
        return "事実・情報"
    if any(w in t for w in ["未来", "予測", "将来", "AGI", "AI", "進化"]):
        return "未来・予測"
    return "その他"


def _build_prediction_prompt(topic: str, participant: dict, transcript: list, stats: dict) -> str:
    history_text = "\n".join(
        f"{t['persona']}: {t['message']}" for t in transcript[-6:]
    ) or "（まだ発言はありません）"

    stats_text = ""
    if stats.get("total", 0) > 0:
        miss_text = "、".join(stats.get("recent_misses", [])) or "なし"
        stats_text = (
            f"\n【過去の予測履歴】{participant['role']}の予測精度: "
            f"{stats['accuracy']:.0%}（{stats['total']}回中{stats['correct']}回的中）。"
            f"最近の外れパターン: {miss_text}\n"
        )

    return f"""
あなたは討論の観察者です。次に発言するAIの発言内容を予測してください。

会議のお題：{topic}

次の発言者の情報：
- 役割: {participant['persona']}
- 立場: {participant.get('stance', '中立')}
- 説明: {participant['description']}
{stats_text}
直近の発言（直近6件）：
{history_text}

この参加者が次に何を言うか、1〜2文で予測してください。
過去に外れたパターンがあれば、それを踏まえて修正してください。

以下のJSONのみ返してください：
{{
  "prediction": "予測される発言（1〜2文）"
}}
""".strip()


def _predict_turn(participant: dict, transcript: list, topic: str) -> str:
    """次のターンの発言を予測する。失敗しても空文字を返すだけで続行。"""
    stats = _get_prediction_stats(participant["role"])
    try:
        result = ask_ai(
            _build_prediction_prompt(topic, participant, transcript, stats),
            model=PREDICTION_MODEL,
        )
        import re as _re, json as _json
        pred = (result.get("prediction") or "").strip()
        if not pred:
            actions = result.get("actions") or []
            if actions and isinstance(actions[0], dict):
                pred = (actions[0].get("content") or "").strip()
        if pred.lstrip().startswith("{"):
            m = _re.search(r'"prediction"\s*:\s*"([^"]+)"', pred)
            if m:
                pred = m.group(1)
            else:
                try:
                    parsed = _json.loads(pred)
                    pred = str(parsed.get("prediction") or "")
                except Exception:
                    pass
        return pred[:200].strip() if pred else ""
    except Exception as e:
        print(f"[VILLAGE_CHAT] 予測生成失敗（無視）: {e}")
        return ""


def _record_prediction(role: str, predicted: str, actual: str, topic: str):
    """予測と実際の発言を比較してログに記録する。"""
    from datetime import datetime as _dt
    similarity = difflib.SequenceMatcher(None, predicted, actual).ratio()
    correct = similarity >= PREDICTION_HIT_THRESHOLD
    log = _load_prediction_log()
    log.append({
        "date": _dt.now().isoformat(timespec="seconds"),
        "topic": topic[:40],
        "role": role,
        "predicted": predicted[:100],
        "actual": actual[:100],
        "similarity": round(similarity, 3),
        "correct": correct,
    })
    _save_prediction_log(log)
    mark = "✓" if correct else "✗"
    print(f"[VILLAGE_CHAT] 予測{mark} {role} 類似度:{similarity:.2f} 「{predicted[:30]}」→「{actual[:30]}」")


def _save_memories(topic: str, summary: str, transcript: list):
    """討論終了後に共有記憶と個別記憶を更新する。"""
    today = _date.today().isoformat()

    # 共有記憶：まとめの最初の一文（事実として何が起きたか）だけ保存
    shared = _load_shared_memory()
    fact_text = summary.split("。")[0][:100] if summary else "（記録なし）"
    shared.append({"date": today, "topic": topic, "facts": fact_text})
    shared = shared[-MEMORY_MAX:]
    try:
        with open(_SHARED_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(shared, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VILLAGE_CHAT] 共有記憶の保存に失敗: {e}")

    # topic_type を分類（感情/技術/哲学/事実/その他）
    topic_type = _classify_topic(topic)

    # 個別記憶：各参加者の最後の発言・理由をその討論での「主張」として記録
    individual = _load_individual_memory()
    last_by_role: dict[str, dict] = {}
    for turn in transcript:
        role = turn.get("role", "")
        if role and role != "human":
            last_by_role[role] = {
                "claim": turn.get("message", "")[:150],
                "reasoning": (turn.get("reasoning") or "")[:80],
            }

    for role, data in last_by_role.items():
        if role not in individual:
            individual[role] = []
        entry = {
            "date": today,
            "topic": topic,
            "topic_type": topic_type,
            "my_claim": data["claim"],
        }
        if data["reasoning"]:
            entry["reasoning"] = data["reasoning"]
        individual[role].append(entry)
        individual[role] = individual[role][-MEMORY_MAX:]

    # 予測精度サマリーを個別記憶に付記（__prediction__キーで格納）
    log = _load_prediction_log()
    if log:
        stats_map = {}
        for entry in log:
            r = entry.get("role", "")
            if r not in stats_map:
                stats_map[r] = {"total": 0, "correct": 0}
            stats_map[r]["total"] += 1
            if entry.get("correct"):
                stats_map[r]["correct"] += 1
        individual["__prediction_stats__"] = {
            r: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": round(s["correct"] / s["total"], 3) if s["total"] else 0,
            }
            for r, s in stats_map.items()
        }

    try:
        with open(_INDIV_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(individual, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[VILLAGE_CHAT] 個別記憶の保存に失敗: {e}")


def _build_turn_prompt(
    topic: str,
    persona: str,
    stance: str,
    description: str,
    transcript: list,
    shared_memory: list | None = None,
    individual_memory: list | None = None,
    prediction: str | None = None,
    member_knowledge: str = "",
    mode: str = "meeting",
) -> str:
    mode_label = "専門家として" if mode == "expert" else "会議メンバーとして"
    # 全モデル共通：直近 N 件のみ渡してコンテキスト溢れを防ぐ
    _MAX_HISTORY = 8
    history_text = "\n".join(
        f"{t['persona']}: {t['message']}" for t in transcript[-_MAX_HISTORY:]
    ) or "（まだ発言はありません。あなたが最初の発言者です）"

    memory_section = ""
    if shared_memory:
        lines = "\n".join(
            f"- {m['date']} 「{m['topic']}」: {m['facts']}"
            for m in shared_memory[-3:]
        )
        memory_section += f"\n【共有記憶：過去の討論で起きた事実（全員が知っている）】\n{lines}\n"
    if individual_memory:
        lines = []
        for m in individual_memory[-3:]:
            line = f"- {m['date']} [{m.get('topic_type','不明')}]「{m['topic']}」での主張: {m['my_claim']}"
            if m.get("reasoning"):
                line += f"（理由: {m['reasoning']}）"
            lines.append(line)
        memory_section += f"\n【あなた自身の記憶：過去の討論であなたが主張したこととその理由】\n" + "\n".join(lines) + "\n"
    if memory_section:
        memory_section += "※共有記憶は「何が起きたか」の事実参照のみ。あなたの意見・結論は自分の立場と個別記憶から自分で判断してください。\n"

    knowledge_section = f"\n{member_knowledge}\n" if member_knowledge else ""

    prediction_section = ""
    if prediction:
        prediction_section = (
            f"\n【他のAIの予測】観察者AIは、あなたが次に「{prediction}」と発言すると予測しています。"
            "この予測が正しいかどうかはあなた自身が判断してください。予測通りでも、裏切っても構いません。\n"
        )

    return f"""
あなたは{mode_label}会議に参加しているAIの一人で、役割は「{persona}」です。
あなたの立場：{stance}
役割の説明：{description}
{memory_section}{knowledge_section}{prediction_section}
会議のお題：
{topic}

ここまでの発言：
{history_text}

ルール：
- 【必須】必ず日本語で発言してください。英語での発言は禁止です。
- 【必須】あなたの立場（{stance}）を守り、必ず自分の意見・結論を一言で断言してください。
  「〜を検討すべき」「〜が重要」「〜を確認する必要がある」で終わる発言は禁止です。
  必ず「〜だ」「〜すべきだ」「〜は間違いだ」「〜が答えだ」の形で締めてください。
- あなた自身の役割（{persona}）の視点から、次の発言を1つだけ考えてください。
- 「議長」の発言は人間の進行役からの指示です。最優先で直接応答してください。
- 【厳禁・最重要】直前のあなた自身の発言と同じキーワード・フレーズ・結論を繰り返すことは絶対禁止です。
  上の「ここまでの発言」を見て、あなた（{persona}）の直前の発言と異なる内容・角度で発言してください。
  同じ単語が3つ以上連続して一致する発言は失格です。必ず新しい視点を加えてください。
- 他の参加者の発言に自然に応答・反応してください。
- 2〜4文程度の自然な日本語にしてください。絵文字・箇条書き・前置きは不要です。
- 【厳禁】他の参加者のコテハン（名前）を自分の発言として名乗ることは絶対に禁止です。あなた自身の役割（{persona}）として発言してください。

以下のJSONのみ返してください：
{{
  "message": "発言内容",
  "reasoning": "なぜそう言ったか（自分の立場・感情・判断基準を10〜20字で）"
}}
""".strip()


def _build_summary_prompt(topic: str, transcript: list) -> str:
    history_text = "\n".join(
        f"{t.get('kote', t['persona'])}: {t['message']}" for t in transcript
    )
    return f"""
以下は、複数のAIが「{topic}」というお題について討論した記録です。
議長（人間）がこの討論を終了し、まとめを求めています。

討論の記録：
{history_text}

以下の内容を含む、簡潔な要約を作成してください：
- 議論で出た主な論点・提案
- 指摘されたリスクや懸念点
- 結論・次に検討すべきこと（合意できていなければ「未決着」として構わない）

以下のJSONのみ返してください：
{{
  "summary": "まとめ文（日本語）"
}}
""".strip()


def start_discussion(topic: str) -> dict:
    topic = (topic or "").strip()
    if not topic:
        return {"status": "error", "message": "お題が空です。討論したいテーマを入力してください。"}

    global _state
    _state = {"active": True, "topic": topic, "transcript": [], "turn_index": 0, "mode": "meeting"}
    _save_state()
    return next_turn()


def inject_moderator_comment(comment: str) -> dict:
    """
    ★2026-08-03 追加：議長(人間)が討論に割り込んで発言する。
    「議長は自分がやる」という当初要望には、進行(次の発言/終了)の
    コントロールだけでなく、発言内容そのものへの介入(誘導・指摘・
    軌道修正)も含まれていたため追加。

    transcriptに議長発言として追加するだけで、turn_index(次に発言する
    AIのローテーション)は進めない。次にnext_turn()が呼ばれた時、
    そのAIは_build_turn_promptの「ここまでの発言」に議長発言も含めた
    履歴を見て応答するため、自然に割り込みへ反応できる。
    """
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "討論が開始されていません。"}

    trimmed = (comment or "").strip()
    if not trimmed:
        return {"status": "error", "message": "コメントが空です。"}

    turn = {"persona": "議長", "role": "human", "message": trimmed}
    _state["transcript"].append(turn)
    _save_state()

    return {
        "status": "turn",
        "turn": turn,
        "transcript": list(_state["transcript"]),
    }


def set_mode(mode: str, specialty: str = "") -> dict:
    """
    討論中のモードを切り替える。
    mode: "meeting" (会議モード) | "expert" (専門家モード)
    specialty: 専門家モード時に有効にする専門名（例: "消防専門"）
    """
    global _state
    if mode not in ("meeting", "expert"):
        return {"status": "error", "message": "mode は 'meeting' または 'expert' を指定してください"}
    _state["mode"] = mode
    _state["specialty"] = specialty if mode == "expert" else ""
    _save_state()
    label = f"専門家モード（{specialty}）" if mode == "expert" and specialty else ("専門家モード" if mode == "expert" else "会議モード")
    print(f"[VILLAGE_CHAT] モード切替: {label}")
    return {"status": "ok", "mode": mode, "specialty": specialty, "label": label}


def _extract_turn(result: dict) -> tuple[str, str]:
    """ask_ai()の戻り値から (message, reasoning) を取り出す。"""
    import json as _json
    import re as _re

    def _clean(text: str) -> str:
        if not text:
            return text
        if text.lstrip().startswith("{"):
            m = _re.search(r'"message"\s*:\s*"(.*?)(?<!\\)"(?:\s*[,}]|$)', text, _re.DOTALL)
            if m:
                text = m.group(1).replace('\\"', '"').replace('\\n', '\n')
            else:
                try:
                    inner = _json.loads(text)
                    if isinstance(inner, dict) and inner.get("message"):
                        text = str(inner["message"])
                except Exception:
                    pass
        text = _re.sub(r'(\s*[\U0001F000-\U0001FFFF]){5,}$', '', text.rstrip())
        text = _re.sub(r'(\s*[👍🙏✨⭐🎉]){3,}$', '', text.rstrip())
        return text.strip()

    def _try_parse(raw: str) -> tuple[str, str]:
        """rawテキストから message と reasoning を抽出する。"""
        msg, rsn = "", ""
        if raw.lstrip().startswith("{"):
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    msg = str(parsed.get("message") or "")
                    rsn = str(parsed.get("reasoning") or "")
                    return _clean(msg), rsn[:80].strip()
            except Exception:
                pass
            m = _re.search(r'"message"\s*:\s*"(.*?)(?<!\\)"', raw, _re.DOTALL)
            if m:
                msg = m.group(1).replace('\\"', '"').replace('\\n', '\n')
            r = _re.search(r'"reasoning"\s*:\s*"(.*?)(?<!\\)"', raw, _re.DOTALL)
            if r:
                rsn = r.group(1).replace('\\"', '"')[:80]
        return _clean(msg), rsn.strip()

    # result が直接 message/reasoning を持つ場合
    direct_msg = (result.get("message") or "").strip()
    if direct_msg:
        rsn = str(result.get("reasoning") or "")[:80].strip()
        return _clean(direct_msg), rsn

    # actions 経由
    actions = result.get("actions") or []
    if actions and isinstance(actions[0], dict):
        content = (actions[0].get("content") or "").strip()
        if content:
            msg, rsn = _try_parse(content)
            if msg:
                return msg, rsn
            return _clean(content), ""

    return "", ""


def _extract_message(result: dict) -> str:
    """後方互換用ラッパー。"""
    msg, _ = _extract_turn(result)
    return msg


def _looks_repetitive(message: str, transcript: list) -> bool:
    """
    直近のAI発言(議長発言は除く)と文字列としての類似度が高い場合、
    「堂々巡りしている可能性」の目印を付けるための簡易チェック。
    埋め込み(Ruri)を使うほどの精度は不要で、議長(人間)への「そろそろ
    介入した方がいいかも」という軽いシグナルであれば十分なため、
    標準ライブラリのdifflibだけで済ませている。
    """
    if not message:
        return False

    recent = [t for t in transcript if t.get("role") != "human"][-SIMILARITY_CHECK_WINDOW:]
    for t in recent:
        prev = t.get("message") or ""
        if not prev:
            continue
        ratio = difflib.SequenceMatcher(None, message, prev).ratio()
        if ratio >= SIMILARITY_WARNING_THRESHOLD:
            return True
    return False


def next_turn(_skips: int = 0) -> dict:
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "討論が開始されていません。"}

    participant = PARTICIPANTS[_state["turn_index"] % len(PARTICIPANTS)]
    kote = get_kote(participant["role"], participant["persona"])

    # ★予測フェーズ：発言前に「何を言うか」を予測する
    prediction = _predict_turn(participant, _state["transcript"], _state["topic"])
    if prediction:
        print(f"[VILLAGE_CHAT] 予測({participant['role']}): {prediction[:60]}...")

    # 4096トークン上限モデル：知識テキストと個別記憶を圧縮してコンテキスト超過を防ぐ
    _SMALL_CTX_ROLES = {"lfm_1b"}
    _SMALL_CTX_MAX_KNOWLEDGE = 400  # 文字数上限

    try:
        from village_learning import get_knowledge_text
        shared_mem = _load_shared_memory()
        indiv_mem  = _load_individual_memory().get(participant["role"], [])
        if participant["role"] in _SMALL_CTX_ROLES:
            indiv_mem = indiv_mem[-1:]  # 小コンテキスト用：3件→1件
        mode = _state.get("mode", "meeting")
        if mode == "expert":
            active_sp = _state.get("specialty", "")
            specialties = participant.get("specialties", [])
            matched = next((s for s in specialties if s["name"] == active_sp), None)
            if matched is None and specialties:
                matched = specialties[0]
            if matched:
                persona     = matched["expert_persona"]
                stance      = matched["expert_stance"]
                description = matched["expert_description"]
            else:
                persona     = participant["persona"]
                stance      = participant.get("stance", "中立")
                description = participant["description"]
        else:
            persona     = participant["persona"]
            stance      = participant.get("stance", "中立")
            description = participant["description"]
        member_knowledge = get_knowledge_text(participant["role"])
        if participant["role"] in _SMALL_CTX_ROLES:
            member_knowledge = member_knowledge[:_SMALL_CTX_MAX_KNOWLEDGE]
        result = ask_ai(
            _build_turn_prompt(
                _state["topic"], persona, stance, description,
                _state["transcript"],
                shared_memory=shared_mem,
                individual_memory=indiv_mem,
                prediction=prediction,
                member_knowledge=member_knowledge,
                mode=mode,
            ),
            model=participant["role"],
        )
        message, reasoning = _extract_turn(result)
    except OllamaUnavailableError as e:
        # 小モデルや大モデルが応答失敗(400等)しても会議を止めない。この子は飛ばす。
        print(f"[VILLAGE_CHAT] {participant['role']} 応答失敗(Ollama 400/接続) → スキップ: {e}")
        message, reasoning = "", ""
    except Exception as e:
        print(f"[VILLAGE_CHAT] 発言生成に失敗しました（{participant['role']}）: {e}")
        message, reasoning = "", ""

    # 他キャラのコテ名を名乗っていたら破棄（空にして下のスキップ判定に回す）
    if message:
        kote_map = _load_kote()
        other_kotes = [k for r, k in kote_map.items() if r != participant["role"]]
        for other_kote in other_kotes:
            if message.strip() == f"{other_kote}だ。" or message.strip().startswith(f"{other_kote}だ"):
                print(f"[VILLAGE_CHAT] 他コテ名乗り検出({participant['role']}): {message!r} → 破棄")
                message, reasoning = "", ""
                break

    # 応答が空／失敗ならこの参加者を飛ばして次へ（全員失敗なら打ち切り）
    if not message:
        print(f"[VILLAGE_CHAT] {participant['role']} をスキップして次の参加者へ")
        _state["turn_index"] += 1
        _save_state()
        if _skips + 1 >= len(PARTICIPANTS):
            return {"status": "error",
                    "message": "参加者全員が応答に失敗しました。Ollama とモデルを確認してください。"}
        return next_turn(_skips + 1)

    if reasoning:
        print(f"[VILLAGE_CHAT] 理由({participant['role']}): {reasoning}")

    # ★検証フェーズ：予測と実際の発言を比較して記録
    if prediction:
        _record_prediction(participant["role"], prediction, message, _state["topic"])

    similar = _looks_repetitive(message, _state["transcript"])

    turn = {
        "persona": participant["persona"],
        "role": participant["role"],
        "kote": kote,
        "message": message,
        "reasoning": reasoning or None,
        "similar_to_previous": similar,
        "prediction": prediction or None,
    }
    _state["transcript"].append(turn)
    _state["turn_index"] += 1
    _save_state()

    return {
        "status": "turn",
        "turn": turn,
        "transcript": list(_state["transcript"]),
    }


def finish_discussion() -> dict:
    global _state
    if not _state["active"]:
        return {"status": "error", "message": "討論が開始されていません。"}

    topic = _state["topic"]
    transcript = _state["transcript"]

    summary = ""
    if transcript:
        try:
            result = ask_ai(_build_summary_prompt(topic, transcript), model="swallow")
            summary = (result.get("summary") or "").strip()
        except Exception as e:
            print(f"[VILLAGE_CHAT] まとめ生成に失敗しました: {e}")

    if not summary:
        lines = [f"「{topic}」についての討論記録:", ""]
        for t in transcript:
            lines.append(f"・{t.get('kote', t['persona'])}: {t['message']}")
        summary = "\n".join(lines)

    _save_memories(topic, summary, transcript)
    _reset()
    return {"status": "done", "topic": topic, "transcript": transcript, "summary": summary}


def cancel_discussion() -> dict:
    global _branch_stack
    _branch_stack.clear()
    _reset()
    return {"status": "cancelled"}


# ─────────────────────────────────────────────────
# ① 指定参加者のみ発言させる共通処理
# ─────────────────────────────────────────────────

def _next_turn_for(participant: dict) -> dict:
    """指定したparticipantの発言を1つ生成する（turn_indexを変えない）。"""
    kote = get_kote(participant["role"], participant["persona"])
    prediction = _predict_turn(participant, _state["transcript"], _state["topic"])
    try:
        from village_learning import get_knowledge_text
        shared_mem = _load_shared_memory()
        indiv_mem  = _load_individual_memory().get(participant["role"], [])
        result = ask_ai(
            _build_turn_prompt(
                _state["topic"], participant["persona"],
                participant.get("stance", "中立"), participant["description"],
                _state["transcript"],
                shared_memory=shared_mem,
                individual_memory=indiv_mem,
                prediction=prediction,
                member_knowledge=get_knowledge_text(participant["role"]),
            ),
            model=participant["role"],
        )
        message, reasoning = _extract_turn(result)
    except OllamaUnavailableError as e:
        # 一巡中に1体が応答失敗(400等)しても止めない。プレースホルダにして続行。
        print(f"[VILLAGE_CHAT] {participant['role']} 応答失敗(Ollama 400/接続): {e}")
        message, reasoning = "", ""
    except Exception as e:
        print(f"[VILLAGE_CHAT] 発言生成失敗({participant['role']}): {e}")
        message, reasoning = "", ""

    if not message:
        message = "（発言の生成に失敗しました）"
        reasoning = ""

    if prediction and message != "（発言の生成に失敗しました）":
        _record_prediction(participant["role"], prediction, message, _state["topic"])

    similar = _looks_repetitive(message, _state["transcript"])
    turn = {
        "persona": participant["persona"],
        "role": participant["role"],
        "kote": kote,
        "message": message,
        "reasoning": reasoning or None,
        "similar_to_previous": similar,
        "prediction": prediction or None,
    }
    _state["transcript"].append(turn)
    _save_state()
    return {"status": "turn", "turn": turn}


# ─────────────────────────────────────────────────
# ② 質問して一巡モード
# ─────────────────────────────────────────────────

def inject_and_round(comment: str) -> dict:
    """議長コメントを挿入し、全参加者を順に一巡させる。"""
    inject_result = inject_moderator_comment(comment)
    if inject_result.get("status") != "turn":
        return inject_result

    turns = []
    for participant in PARTICIPANTS:
        r = _next_turn_for(participant)
        if r.get("status") == "error":
            return r
        turns.append(r["turn"])

    return {
        "status": "round_done",
        "turns": turns,
        "transcript": list(_state["transcript"]),
    }


# ─────────────────────────────────────────────────
# ③ 指名モード
# ─────────────────────────────────────────────────

def inject_and_named(comment: str, roles: list) -> dict:
    """議長コメントを挿入し、指定したroleの参加者だけ応答させる。"""
    inject_result = inject_moderator_comment(comment)
    if inject_result.get("status") != "turn":
        return inject_result

    named = [p for p in PARTICIPANTS if p["role"] in roles]
    if not named:
        return {"status": "error", "message": "指定されたroleが見つかりません。"}

    turns = []
    for participant in named:
        r = _next_turn_for(participant)
        if r.get("status") == "error":
            return r
        turns.append(r["turn"])

    return {
        "status": "named_done",
        "turns": turns,
        "transcript": list(_state["transcript"]),
    }


# ─────────────────────────────────────────────────
# ④ テーマ分岐（フォーク / メインに戻る）
# ─────────────────────────────────────────────────

def fork_discussion(sub_topic: str) -> dict:
    """現在の討論を一時保存してサブ討論を開始する。"""
    global _state, _branch_stack
    if not _state["active"]:
        return {"status": "error", "message": "討論が開始されていません。"}
    sub_topic = (sub_topic or "").strip()
    if not sub_topic:
        return {"status": "error", "message": "サブ討論のお題が空です。"}

    _branch_stack.append({k: list(v) if isinstance(v, list) else v for k, v in _state.items()})
    _state = {"active": True, "topic": sub_topic, "transcript": [], "turn_index": 0}
    _save_state()
    return {**next_turn(), "branch_depth": len(_branch_stack)}


def resume_main() -> dict:
    """サブ討論を終了してメインの討論に戻る。"""
    global _state, _branch_stack
    if not _branch_stack:
        return {"status": "error", "message": "戻るべきメイン討論がありません。"}

    sub_topic = _state["topic"]
    sub_count  = len([t for t in _state["transcript"] if t.get("role") != "human"])
    _state = _branch_stack.pop()
    _state["transcript"].append({
        "persona": "議長",
        "role": "human",
        "message": f"【サブ討論「{sub_topic}」終了 / {sub_count}発言】",
    })
    _save_state()
    return {
        "status": "resumed",
        "topic": _state["topic"],
        "transcript": list(_state["transcript"]),
        "branch_depth": len(_branch_stack),
    }


def get_participants_info() -> list:
    """フロントエンド向けに参加者一覧（role/persona）を返す。"""
    kote_map = _load_kote()
    return [
        {
            "role": p["role"],
            "persona": p["persona"],
            "kote": kote_map.get(p["role"], p["persona"]),
            "stance": p.get("stance", ""),
        }
        for p in PARTICIPANTS
    ]
