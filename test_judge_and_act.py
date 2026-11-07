"""judge_and_act() の単体テスト"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\ai_agent\server")

from village_learning import judge_and_act

# テスト1: IoT脅威系のニュース → f99への注意喚起を期待
print("=" * 50)
print("テスト1: IoT脅威ニュース")
result = judge_and_act(
    role="granite",
    title="Mirai亜種がルーターを大規模攻撃、初期PWのデバイスが標的に",
    summary="新たなMirai亜種が確認され、初期パスワードを変更していないIoTデバイスを自動スキャンして感染を広げている。"
            "特に家庭用ルーターへの攻撃が急増しており、ファームウェア更新と強固なパスワード設定が急務とされる。"
)
print(f"判断: {result.get('action')}")
print(f"詳細: {result}")

# テスト2: 研究論文系 → f05または会議を期待
print("\n" + "=" * 50)
print("テスト2: AI研究論文")
result2 = judge_and_act(
    role="swallow",
    title="大規模言語モデルの推論効率を3倍改善する新手法を発表",
    summary="MITの研究チームが、LLMの推論時計算コストを大幅削減する新アーキテクチャを発表した。"
            "既存モデルに後付けできる手法で、精度を維持しながら推論速度を約3倍向上させる。"
)
print(f"判断: {result2.get('action')}")
print(f"詳細: {result2}")
