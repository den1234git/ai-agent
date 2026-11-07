from komainu_router import call_komainu
import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv()

def evaluate_with_real_teacher(title, content, theme="", code=""):
    print(f"\n[狛犬教室] 登校 - 生徒:バグが尻尾振ってる")
    print(f"[デバッグ嬢] ふーん、また変なの持ってきたの？")
    
    prompt = f"""
あなたは元CIA工作員で今は平凡な教員、狛犬先生。
飼い犬のバグと猫のデバッグ嬢と暮らし、神社の狛犬が好き。
生徒の作品を100点満点で厳しく、でも愛を持って採点してください。

テーマ: {theme}
タイトル: {title}
説明: {content[:1000]}
コード: {code[:2000]}

採点基準：
- 動くか (40点)
- コードの美しさ (30点)
- 発想 (30点)

以下のJSONだけで返してください。他の文字は一切不要です。
{{"score": 85, "good": "良い点を一言", "bad": ["ダメな点1", "ダメな点2"], "fix": "具体的な直し方を1行で"}}
"""
    try:
        # ★ここが進化！ローカルLlama(3B) → 無料70Bハシゴ
        print("[狛犬先生] 無料70Bハシゴで採点中... Groq → Cerebras → OpenRouter")
        data = call_komainu(prompt, role="judge")

        # call_komainuが {score, good, bad, fix} を直接返した場合と、
        # 生テキストが混じった場合の両対応
        if isinstance(data, dict) and "score" in data and "good" in data:
            result_data = data
        else:
            # 文字列化してJSON抽出（保険）
            result_text = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
            print(f"[狛犬先生の生採点]\n{result_text[:500]}\n")
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if match:
                result_data = json.loads(match.group(0))
            else:
                result_data = {"score": 50, "good": "解析失敗", "bad": [result_text[:100]], "fix": ""}

        return result_data

    except Exception as e:
        print(f"[採点エラー] {e}")
        return {"score": 50, "good": "", "bad": [str(e)], "fix": "リトライしてください"}