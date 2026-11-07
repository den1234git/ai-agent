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
以下のJSONだけで返してください。
{{"score": 85, "good": "良い点を一言", "bad": ["ダメな点1"], "fix": "直し方を1行で"}}
"""
    try:
        print("[狛犬先生] 無料70Bハシゴで採点中...")
        from komainu_router import call_komainu_judge
        data = call_komainu_judge(prompt)
        if isinstance(data, dict) and "score" in data and "good" in data:
            result_data = data
        else:
            result_text = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
            match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not match:
                raise ValueError("JSON not found")
            result_data = json.loads(match.group(0))
        score = int(result_data.get("score", 60))
        print(f"[狛犬先生] {score}点だ。")
        return result_data
    except Exception as e:
        print(f"[狛犬教室 停電 - 70B全滅] {e}")
        return {"score": 50, "good": "とりあえず動いた", "bad": ["先生が休み"], "fix": "もう一度"}

def evaluate_relevance_multi(title, content, theme=""):
    return evaluate_with_real_teacher(title, content, theme, "")