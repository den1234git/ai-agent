# server/teachers/komainu.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

#.env 読み込み
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv() # ルートの.envも

# komainu_router を探す ( teachers/ -> server/ )
try:
    from server.komainu_router import call_komainu
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from komainu_router import call_komainu

class KomainuTeacher:
    id = "komainu"
    name = "狛犬先生"
    icon = "🦁"

    system_prompt = """
    あなたは狛犬先生。神社の番犬で、プログラミングの守り神。
    ルール:
    1. 必ず最初に0-100点で採点する。甘めに。動けば70点。
    2. 良い所を1つ褒める
    3. バグや改善点があったら「ワン！」と吠えてから指摘する
    4. 最後は「合格だ」で締める
    語尾は「だ」「である」で、たまに「ガル」と唸る。
    出力は必ずJSON: {"score": int, "comment": "講評本文"}
    """

    def evaluate(self, code: str) -> dict:
        # 1. 70B無料ハシゴに挑戦
        try:
            prompt = f"""
            以下のコードを採点せよ。

            【コード】
            {code[:4000]}

            【採点基準】
            {self.system_prompt}
            """
            result = call_komainu(prompt, role="judge")

            # call_komainuは {score, comment} or {winner...} を返すので正規化
            if "score" in result:
                return {"score": int(result["score"]), "comment": result["comment"]}
            # もし旧フォーマットが返ってきた場合の保険
            if "score_a" in result:
                avg = (result["score_a"] + result["score_b"]) // 2
                return {"score": avg, "comment": result.get("reason", "合格だ")}

        except Exception as e:
            print(f"[Komainu] 70B API失敗、ローカル8Bにフォールバック: {e}")

        # 2. フォールバック: ローカル LFM2.5-8B (5.2GB)
        # ここは君が登録済みのローカル呼び出しを入れる
        # 例: return self._call_local_8b(code)

        # 3. 最終フォールバック (今のロジック)
        score = 85
        if "bug" in code.lower() or "error" in code.lower():
            score = 75
            return {"score": score, "comment": "ワン！ バグの匂いがするぞ。だがよく頑張った、合格だ！ガル"}
        return {"score": score, "comment": f"{score}点だ、合格だ。守りは固いぞ。ガル"}

# 単体テスト用
if __name__ == "__main__":
    t = KomainuTeacher()
    print(t.evaluate("print('hello world')"))