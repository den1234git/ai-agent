"""
router役候補モデル（router.pyのROUTER_STUDENT_POOL: router/llama3_2/gemma3/granite）を
比較するための自動評価スクリプト。

5軸（2026-07-26 合意）:
  1. 応答速度    : 平均レイテンシ（秒）。モデル間で相対正規化（最速=100）
  2. JSON妥当性  : 期待するJSONスキーマ（intent/model キー）を正しく返せた割合
  3. 分類精度    : 正解ラベル付きテストセットとの一致率
  4. 出力安定性  : 同一プロンプトを3回投げたときの多数決一致率
  5. トークン効率: Ollamaのeval_count/eval_durationから算出したtokens/sec（相対正規化）

router役は実運用でRAGも使わずコード生成もしない、純粋なJSON分類タスク
（_classify_intentの意図判定、_route_with_ollamaのモデル振り分け）専用なので、
評価もこの2タスクに絞る。router.py の _build_intent_prompt() / _build_routing_prompt() を
そのまま再利用し、本番と全く同じプロンプトでテストする。

ai.py の ask_ai() は eval_count 等のトークン統計を返さないため、
この評価ではOllamaの /api/chat を直接叩く（model_stress_test.pyと同じ方式）。
ただしsystemプロンプトはai.pyのjson_mode=True時と同じものを使う。

出力: server/model_comparison_result.json
"""

import json
import time
import requests

from router import _build_intent_prompt, _build_routing_prompt, ROUTER_STUDENT_POOL
from ai import MODEL_MAP, KEEP_ALIVE

OLLAMA_URL = "http://localhost:11434/api/chat"
JSON_SYSTEM_PROMPT = (
    "あなたはJSON形式のみで応答するAIエージェントです。"
    "説明文や前置きは一切書かず、最初の文字から必ず { で始めてください。"
)
TIMEOUT_SECONDS = 120
CONSISTENCY_REPEATS = 3  # 初回込みで合計3回

OUTPUT_PATH = "model_comparison_result.json"

# ─────────────────────────────────────────
# 正解ラベル付きテストセット
# router.pyの_classify_intentプロンプト内コメントに書かれている判定基準から
# 代表的なケースを抽出して作成。
# ─────────────────────────────────────────

INTENT_TEST_CASES = [
    {"user": "StaffListという名前のコンポーネントを作って", "expected": "shobo_app"},
    {"user": "お問い合わせフォームの画面を作って", "expected": "html_mockup"},
    {"user": "自己改善ループを実装してください", "expected": "python_module"},
    {"user": "ChromaDBに保存されている知識の件数を教えて", "expected": "other"},
    {"user": "Customers.tsxの中身を確認して教えて", "expected": "other"},
    {"user": "ログイン画面のHTMLを作って", "expected": "html_mockup"},
    {"user": "新しい判定ロジックを追加する関数を実装して", "expected": "python_module"},
    {"user": "こんにちは", "expected": "other"},
    {"user": "消防法の目的について教えてください", "expected": "other"},
    {"user": "Settings.tsxというファイルを作って", "expected": "shobo_app"},
]

ROUTING_TEST_CASES = [
    {"user": "Reactでカウンターコンポーネントを作って", "expected": "codellama"},
    {"user": "今日の天気について雑談しましょう", "expected": "swallow"},
    {"user": "Pythonでソートアルゴリズムを実装して", "expected": "codellama"},
    {"user": "消防法の歴史について教えてください", "expected": "swallow"},
    {"user": "この関数のバグを直して", "expected": "codellama"},
    {"user": "励ましの言葉をかけてください", "expected": "swallow"},
]

# 出力安定性チェック対象（インデックス指定、代表的なものだけ3回ずつ追加実行）
CONSISTENCY_INTENT_INDEXES = [0, 7]    # shobo_app確定ケース / other確定ケース
CONSISTENCY_ROUTING_INDEXES = [0, 1]   # codellama確定ケース / swallow確定ケース


def call_ollama_raw(prompt: str, model_name: str) -> dict:
    """Ollamaに直接リクエストし、内容・レイテンシ・トークン数を1回でまとめて取得する。"""
    start = time.time()
    res = requests.post(
        OLLAMA_URL,
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": JSON_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": KEEP_ALIVE,
        },
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.time() - start
    res.raise_for_status()
    data = res.json()
    return {
        "content": data.get("message", {}).get("content", ""),
        "latency": elapsed,
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
    }


def parse_json_field(content: str, field: str):
    """contentからJSONをできるだけ寛容にパースし、fieldの値を返す。失敗時はNone。"""
    text = content.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end]).get(field)
    except Exception:
        return None


def run_case(model_name: str, prompt: str, field: str) -> dict:
    try:
        raw = call_ollama_raw(prompt, model_name)
    except Exception as e:
        return {"ok": False, "value": None, "latency": None, "tokens_per_sec": None, "error": str(e)}

    value = parse_json_field(raw["content"], field)
    tokens_per_sec = None
    if raw["eval_duration_ns"]:
        tokens_per_sec = raw["eval_count"] / (raw["eval_duration_ns"] / 1e9)

    return {"ok": value is not None, "value": value, "latency": raw["latency"], "tokens_per_sec": tokens_per_sec}


def evaluate_model(role: str) -> dict:
    model_name = MODEL_MAP[role]
    print(f"\n[EVAL] {role} ({model_name}) の評価を開始します")

    all_results = []
    latencies = []
    tokens_per_sec_list = []
    intent_first_values = {}
    routing_first_values = {}

    for i, case in enumerate(INTENT_TEST_CASES):
        prompt = _build_intent_prompt(case["user"])
        r = run_case(model_name, prompt, "intent")
        r.update({"expected": case["expected"], "task": "intent", "user": case["user"]})
        all_results.append(r)
        if r["latency"] is not None:
            latencies.append(r["latency"])
        if r["ok"] and r["tokens_per_sec"] is not None:
            tokens_per_sec_list.append(r["tokens_per_sec"])
        intent_first_values[i] = r["value"]
        print(f"  [intent]  {case['user'][:24]:<24} -> {r['value']!r:<16} (expected={case['expected']}, ok={r['ok']})")

    for i, case in enumerate(ROUTING_TEST_CASES):
        prompt = _build_routing_prompt(case["user"])
        r = run_case(model_name, prompt, "model")
        r.update({"expected": case["expected"], "task": "routing", "user": case["user"]})
        all_results.append(r)
        if r["latency"] is not None:
            latencies.append(r["latency"])
        if r["ok"] and r["tokens_per_sec"] is not None:
            tokens_per_sec_list.append(r["tokens_per_sec"])
        routing_first_values[i] = r["value"]
        print(f"  [routing] {case['user'][:24]:<24} -> {r['value']!r:<16} (expected={case['expected']}, ok={r['ok']})")

    consistency_groups = []

    for idx in CONSISTENCY_INTENT_INDEXES:
        case = INTENT_TEST_CASES[idx]
        prompt = _build_intent_prompt(case["user"])
        values = [intent_first_values[idx]]
        for _ in range(CONSISTENCY_REPEATS - 1):
            r = run_case(model_name, prompt, "intent")
            values.append(r["value"])
            if r["latency"] is not None:
                latencies.append(r["latency"])
        consistency_groups.append(values)
        print(f"  [consistency/intent]  {case['user'][:24]:<24} -> {values}")

    for idx in CONSISTENCY_ROUTING_INDEXES:
        case = ROUTING_TEST_CASES[idx]
        prompt = _build_routing_prompt(case["user"])
        values = [routing_first_values[idx]]
        for _ in range(CONSISTENCY_REPEATS - 1):
            r = run_case(model_name, prompt, "model")
            values.append(r["value"])
            if r["latency"] is not None:
                latencies.append(r["latency"])
        consistency_groups.append(values)
        print(f"  [consistency/routing] {case['user'][:24]:<24} -> {values}")

    total = len(all_results)
    json_ok = sum(1 for r in all_results if r["ok"])
    correct = sum(1 for r in all_results if r["ok"] and r["value"] == r["expected"])

    agreement_rates = []
    for values in consistency_groups:
        counter = {}
        for v in values:
            counter[v] = counter.get(v, 0) + 1
        agreement_rates.append(max(counter.values()) / len(values))

    return {
        "role": role,
        "model_name": model_name,
        "raw": {
            "avg_latency_sec": (sum(latencies) / len(latencies)) if latencies else None,
            "json_validity_rate": (json_ok / total) if total else 0.0,
            "accuracy": (correct / total) if total else 0.0,
            "consistency_rate": (sum(agreement_rates) / len(agreement_rates)) if agreement_rates else 0.0,
            "avg_tokens_per_sec": (sum(tokens_per_sec_list) / len(tokens_per_sec_list)) if tokens_per_sec_list else 0.0,
        },
        "details": all_results,
    }


def normalize_scores(results: list) -> list:
    """5モデル間の相対値から0-100スコアを算出する。"""
    latencies = [r["raw"]["avg_latency_sec"] for r in results if r["raw"]["avg_latency_sec"] is not None]
    tps_list = [r["raw"]["avg_tokens_per_sec"] for r in results if r["raw"]["avg_tokens_per_sec"]]

    lat_min, lat_max = (min(latencies), max(latencies)) if latencies else (0, 0)
    tps_max = max(tps_list) if tps_list else 0

    for r in results:
        raw = r["raw"]

        if raw["avg_latency_sec"] is None:
            speed_score = 0.0
        elif lat_max == lat_min:
            speed_score = 100.0
        else:
            speed_score = 100.0 * (1 - (raw["avg_latency_sec"] - lat_min) / (lat_max - lat_min))

        efficiency_score = (raw["avg_tokens_per_sec"] / tps_max * 100.0) if tps_max > 0 else 0.0

        r["scores"] = {
            "応答速度": round(speed_score, 1),
            "JSON妥当性": round(raw["json_validity_rate"] * 100.0, 1),
            "分類精度": round(raw["accuracy"] * 100.0, 1),
            "出力安定性": round(raw["consistency_rate"] * 100.0, 1),
            "トークン効率": round(efficiency_score, 1),
        }

    return results


def main():
    results = [evaluate_model(role) for role in ROUTER_STUDENT_POOL]
    results = normalize_scores(results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== 結果サマリー ===")
    for r in results:
        print(f"{r['role']} ({r['model_name']}): {r['scores']}")
    print(f"\n詳細は {OUTPUT_PATH} に保存しました。")


if __name__ == "__main__":
    main()
