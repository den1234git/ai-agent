"""
_typecheck_code()の中身を経由せず、tscの生出力をそのまま見るための一時デバッグ用。
test_typecheck.pyがhas_errors=Falseを返し続ける原因(regexが噛み合ってない/
tscがプロジェクト単位で先にコケてる/そもそも対象ファイルが見えてない等)を
切り分けるために、returncode・stdout・stderrを加工せず全部そのまま表示する。

使い方:
  cd G:\\ai_agent\\server
  python _debug_tsc_raw.py
"""
import os
import subprocess

COMPONENT_NAME = "TypecheckTestComponent"
FILE_PATH = f"src/pages/{COMPONENT_NAME}.tsx"

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

base = os.environ.get("SHOBO_APP_DIR", r"C:\ai_agent\AI_001\shobo-app")
tsconfig_path = os.path.join(base, "tsconfig.json")
target_path = os.path.join(base, FILE_PATH.replace("/", os.sep))

print(f"SHOBO_APP_DIR = {base}")
print(f"tsconfig_path = {tsconfig_path}  (存在={os.path.exists(tsconfig_path)})")
print(f"target_path   = {target_path}")

os.makedirs(os.path.dirname(target_path), exist_ok=True)
with open(target_path, "w", encoding="utf-8") as f:
    f.write(BROKEN_CODE)
print(f"書き出し完了。ファイルサイズ={os.path.getsize(target_path)}バイト")

cmd = f'npx tsc --noEmit -p "{tsconfig_path}"'
print(f"\n実行コマンド: {cmd}")
print(f"cwd: {base}\n")

proc = subprocess.run(
    cmd,
    cwd=base,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=120,
    shell=True,
)

print("=" * 60)
print(f"returncode = {proc.returncode}")
print("=" * 60)
print("--- stdout (生) ---")
print(repr(proc.stdout))
print()
print("--- stdout (表示用) ---")
print(proc.stdout)
print()
print("--- stderr (生) ---")
print(repr(proc.stderr))
print()
print("--- stderr (表示用) ---")
print(proc.stderr)

_report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug_tsc_raw_result.txt")
with open(_report_path, "w", encoding="utf-8") as f:
    f.write(f"returncode = {proc.returncode}\n\n")
    f.write("--- stdout ---\n")
    f.write(proc.stdout or "(空)")
    f.write("\n\n--- stderr ---\n")
    f.write(proc.stderr or "(空)")
print(f"\n結果を {_report_path} にも保存しました。")
