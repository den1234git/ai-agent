"""
_typecheck_code() の動作確認用テストスクリプト（2026-07-31 追加）。

LAの生成フロー（Ollama経由）を通さず、_typecheck_code() を直接呼び出して
「壊れたコード」と「正しいコード」の両方を試す。Ollamaが別の処理で
埋まっていても実行できる（LLMを一切使わないため）。

使い方:
  cd G:\\ai_agent\\server
  python test_typecheck.py
"""

from router import _typecheck_code

COMPONENT_NAME = "TypecheckTestComponent"
FILE_PATH = f"src/pages/{COMPONENT_NAME}.tsx"

# ── ケース1: わざと壊したコード ──────────────────────────────
# ・存在しない型 NonExistentType を使用
# ・存在しない関数 thisFunctionDoesNotExist() を呼び出す
# ・number型のstateに文字列を代入
BROKEN_CODE = f"""import {{ useState }} from 'react'

interface {COMPONENT_NAME}Props {{
  label: NonExistentType
}}

export default function {COMPONENT_NAME}() {{
  const [count, setCount] = useState<number>(0)

  const handleClick = () => {{
    setCount('文字列を代入（型エラーになるはず）')
    thisFunctionDoesNotExist()
  }}

  return (
    <div>
      <button onClick={{handleClick}}>{{count}}</button>
    </div>
  )
}}
"""

# ── ケース2: 正しいコード（false positiveが出ないことの確認用） ──
GOOD_CODE = f"""import {{ useState }} from 'react'

export default function {COMPONENT_NAME}() {{
  const [count, setCount] = useState<number>(0)

  const handleClick = () => {{
    setCount(count + 1)
  }}

  return (
    <div>
      <button onClick={{handleClick}}>{{count}}</button>
    </div>
  )
}}
"""


def main():
    print("=" * 60)
    print("ケース1: 壊れたコードを型チェック（エラーが検出されるはず）")
    print("=" * 60)
    result = _typecheck_code(BROKEN_CODE, FILE_PATH)
    print(f"has_errors = {result['has_errors']}")
    for e in result["errors"]:
        print(f"  - {e}")
    if result["has_errors"]:
        print("✅ 期待通りエラーを検出しました。")
    else:
        print("⚠️ エラーが検出されませんでした（tsc未検出/タイムアウト等でスキップされた可能性があります。上のログを確認してください）。")

    print()
    print("=" * 60)
    print("ケース2: 正しいコードを型チェック（エラーが出ないはず）")
    print("=" * 60)
    result2 = _typecheck_code(GOOD_CODE, FILE_PATH)
    print(f"has_errors = {result2['has_errors']}")
    for e in result2["errors"]:
        print(f"  - {e}")
    if not result2["has_errors"]:
        print("✅ 期待通りエラーなしでした。")
    else:
        print("⚠️ 正しいコードなのにエラーが出ました（誤検知の可能性、上のエラー内容を確認してください）。")


if __name__ == "__main__":
    main()
