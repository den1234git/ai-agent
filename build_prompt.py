import json
import os

def build_prompt(user, mem):
    memory_str = json.dumps(mem or {}, ensure_ascii=False)
    return f"""
SYSTEM:
あなたはJSON形式のみで応答するAIエージェントです。必ず日本語で考え、日本語で処理してください。

重要ルール：
1. 「私の名前はX」という入力の場合：
   - type = "remember"
   - key = "name"
   - value = Xをそのままコピー

2. 質問（?や？を含む）の場合：
   - type = "answer"
   - contentに日本語で回答

3. コード生成・ファイル作成の場合：
   - type = "write_file"
   - path = 必ずファイルパスを指定
   - content = ファイルの内容
   - keyやvalueは使わない
   - pathは必須。省略禁止。

4. JSONのみ出力。説明文・コードブロック不要。

出力形式：
{{
  "plan": ["string"],
  "actions": [
    {{
      "type": "answer|remember|write_file|read_file",
      "content": "string",
      "path": "string",
      "key": "string",
      "value": "any"
    }}
  ]
}}

記憶:
{memory_str}

ユーザー:
{user}
""".strip()


def build_html_mockup_prompt(task):
    """
    shobo-app専用のReact分割生成パイプラインとは別に、汎用のUIモックアップ
    （どんなソフトウェアのものでもよい）を、単発でHTML+CSS1ファイルとして
    生成するためのプロンプト。
    """
    return f"""
以下のタスクに基づいて、単体で動くHTML+CSSのUIモックアップを1ファイルで生成してください。

タスク：
{task}

ルール：
- HTMLとCSSは1つのファイルにまとめること（<style>タグをheadに含める）。外部ファイルへの参照は禁止。
- JavaScriptは見た目の確認に必要な範囲（タブ切り替えなど）であれば<script>タグで含めてよいが、
  サーバーとの通信や複雑な処理は書かないこと。
- 日本語のラベル・文言を使い、実際の画面のように見える内容にすること。
- 配色・レイアウトもある程度整えること。装飾のないプレーンなフォームにしないこと。
- ファイルパスは必ず "mockups/わかりやすい名前.html" の形式にすること。

JSONのみ返してください。typeは必ず"write_file"を使うこと：
{{
  "plan": ["string"],
  "actions": [
    {{
      "type": "write_file",
      "path": "mockups/わかりやすい名前.html",
      "content": "生成したHTMLコード全体"
    }}
  ]
}}
""".strip()


def build_multi_file_plan_prompt(task, max_files=3):
    """
    1つの依頼に対して、複数の「独立した」Reactコンポーネントファイルが
    必要かどうかをAIに判断させ、それぞれの計画（コンポーネント名・役割の
    説明）を立てさせるためのプロンプト。
    """
    return f"""
以下のタスクを実行するために、必要なReact TypeScriptコンポーネントファイルの計画を立ててください。

タスク：
{task}

重要なルール：
- 1つのタスクから、複数の画面・コンポーネントが必要だと判断した場合は、
  それぞれを「独立した」ファイルとして計画すること。
- 各ファイルは互いに依存しない、単体で完結したコンポーネントとして計画すること
  （他のファイルをimportしたり、画面遷移で連携する設計は今回考えなくてよい）。
- ファイル数は最大{max_files}個まで。{max_files}個を超える場合は、
  最も重要なものから{max_files}個に絞ること。
- 1つのコンポーネントで十分な単純なタスクの場合は、無理に分割せず1個でよい。
- component_nameはPascalCase（例: StaffList）にすること。
- descriptionには、そのファイルが何をするコンポーネントかを具体的に書くこと
  （後で個別に生成するときの指示文として使われる）。

JSONのみ返してください：
{{
  "files": [
    {{
      "component_name": "コンポーネント名",
      "description": "このコンポーネントが何をするかの具体的な説明"
    }}
  ]
}}
""".strip()


def build_section_prompt(section_name, task, context="", reference_files=None, knowledge_context=""):
    """
    分割生成の各セクション用プロンプト。
    reference_files: [{"path": "src/pages/Foo.tsx", "content": "..."}, ...] の形式
    knowledge_context: ChromaDBから取得した消防法・業務知識のテキスト
    """

    section_specs = {
        "imports_and_types": {
            "description": "ファイルの先頭部分。import文と、必要な型定義だけを書く。関数定義やexportは書かない。",
            "example": "import { useState } from 'react';\n\ninterface CounterState {\n  count: number;\n}",
        },
        "state_and_handlers": {
            "description": (
                "コンポーネント関数の【中身のみ】を書く。"
                "useState定義、useEffect、handleChangeなどの処理関数を書く。"
                "ここで言う「処理関数」には、const handleClick = () => { ... }; のように"
                "ハンドラーをconstで宣言することも含まれ、これは正しい書き方なので"
                "省略したり消したりしないこと。"
                "禁止されているのは、コンポーネント自体の宣言行"
                "（例：export default function ComponentName() { や "
                "const ComponentName: React.FC = () => { ）だけである。"
                "JSXのreturn文も書かないこと。"
                "あなたの出力は、すでにある「コンポーネント関数」の{と}の間にそのまま挿入される。"
            ),
            "example": (
                "const [count, setCount] = useState(0);\n\n"
                "const handleIncrement = () => {\n  setCount(count + 1);\n};"
            ),
        },
        "jsx_return": {
            "description": (
                "return( と ); の【内側に入るJSX要素のみ】を書く。"
                "'return' という単語、セミコロン、波カッコ、コメントのみの省略は禁止。"
                "必ず実際に表示されるJSXタグ（<div>など）から書き始めること。"
            ),
            "example": (
                "<div>\n  <p>Count: {count}</p>\n"
                "  <button onClick={handleIncrement}>増やす</button>\n</div>"
            ),
        },
    }

    spec = section_specs.get(section_name, {"description": "", "example": ""})

    extra_rule = ""
    if section_name == "state_and_handlers":
        extra_rule = (
            "\n- ただし、handleClickやhandleSubmitのような「ハンドラー関数」を "
            "const handleClick = () => {...}; の形で宣言するのは問題ありません。"
            "削除したり省略したりしないこと。禁止されているのは、コンポーネント自体の"
            "宣言行（export default function ... { など）だけです。"
        )

    # referenceファイルのブロックを組み立て
    reference_block = ""
    if reference_files:
        parts = []
        for rf in reference_files:
            path = rf.get("path", "unknown")
            content = rf.get("content", "")
            parts.append(f"// === 参考ファイル: {path} ===\n{content}")
        reference_block = "\n\n参考ファイル（構造・スタイル・パターンを参考にすること）：\n" + "\n\n".join(parts)

    # 消防知識ブロックを組み立て
    knowledge_block = ""
    if knowledge_context:
        knowledge_block = f"\n\n【関連する消防法・業務知識】（コード生成の参考にすること）：\n{knowledge_context}"

    return f"""
React + TypeScriptのコードの一部分のみを生成してください。

生成する部分：{section_name}
この部分の役割：{spec['description']}

出力例（このような形式・粒度で書くこと。内容はタスクに応じて変えること）：
{spec['example']}

タスク全体の説明：
{task}

これまでに生成された部分（参考）：
{context}{reference_block}{knowledge_block}

絶対ルール：
- 指定された部分のみを書く。他の部分（コンポーネント自体の宣言・return文・閉じカッコなど）は絶対に書かない。{extra_rule}
- 「JSX部分は不要」「ここに処理を書く」のような説明・コメントだけの省略は禁止。必ず実際に動くコードを書く。
- コメントや説明文は不要。コードのみ。
- 完全に動作するコードにする。省略しない。

JSONのみ返してください：
{{
  "code": "生成したコード部分"
}}
""".strip()


def build_split_task_prompt(task):
    return f"""
以下のタスクを実行するために、React TypeScriptコンポーネントを3つの部分に分けて作る計画を立ててください。

タスク：
{task}

重要：file_pathは必ず "src/pages/" から始めること。例："src/pages/Staff.tsx"
component_nameはファイル名と同じにすること（拡張子なし）。

JSONのみ返してください：
{{
  "component_name": "コンポーネント名",
  "file_path": "src/pages/コンポーネント名.tsx",
  "steps": ["imports_and_types", "state_and_handlers", "jsx_return"]
}}
""".strip()


def build_connection_plan_prompt(components_summary, original_task):
    """
    複数ファイル生成で独立して作られたコンポーネントの一覧を見せて、
    「どれとどれを親子関係（連携）にすべきか」をAIに判断させるための
    プロンプト。
    """
    return f"""
以下は、1つの依頼から独立して生成された複数のReactコンポーネントの一覧です。
これらのうち、画面として自然に連携させるべきペアがあるかを判断してください。

元の依頼内容：
{original_task}

生成されたコンポーネント一覧：
{components_summary}

判断基準：
- 「一覧」と「詳細」のような関係（一覧から項目を選ぶと詳細が表示される）は
  連携させるべき典型例です。
- 連携は「親が子を呼び出す」という片方向の関係のみで考えてください
  （親のコンポーネントの中に、子のコンポーネントを組み込む形）。
- 明らかに関連がない、あるいは独立して使うべきコンポーネント同士は
  無理に連携させないでください。
- 連携が不要だと判断した場合は connections を空配列にしてください。

JSONのみ返してください：
{{
  "connections": [
    {{
      "parent": "親になるコンポーネント名",
      "child": "子になるコンポーネント名",
      "relationship": "関係の説明（例: 一覧から選択した項目の詳細を表示する）"
    }}
  ]
}}
""".strip()


def build_connect_components_prompt(parent_name, parent_code, child_name, child_code, relationship):
    """
    親コンポーネントのコードを、子コンポーネントを使う形に書き換えさせる
    ためのプロンプト。
    """
    return f"""
以下の2つのReact TypeScriptコンポーネントを連携させてください。
{parent_name} の中に {child_name} を組み込み、{relationship}という関係を実現してください。

親コンポーネント（{parent_name}）の現在のコード：
{parent_code}

子コンポーネント（{child_name}）の現在のコード：
{child_code}

ルール：
- 親コンポーネント（{parent_name}）のコード全体を書き直してください。
- 子コンポーネント（{child_name}）のコードは変更しないでください
  （子コンポーネントがPropsを要求する場合は、親側から正しい形でPropsを渡してください）。
- 子コンポーネントのimport文を追加してください
  （例: import {child_name} from './{child_name}';）。
- 親コンポーネントに「選択中の項目」を保持するstateを追加し、
  一覧の各項目に選択用のボタンや操作を追加してください。
- 選択中の項目がある場合のみ、子コンポーネントをJSX内に描画してください。
- 子コンポーネントが受け取るPropsの型・名前は、子コンポーネントの
  既存コードに合わせて正確に渡してください。勝手にProps名を変更しないでください。
- 既存の機能（一覧表示など）は壊さず、そのまま残してください。
- コメントや説明文は不要。コードのみ。
- import文・型定義・処理・JSXをすべて含む、完全に動作する1ファイル分の
  コードにすること。

JSONのみ返してください：
{{
  "code": "書き直した{parent_name}の完全なコード"
}}
""".strip()
def build_vision_prompt(user_message: str) -> str:
    """
    vision モデル（gemma4:e2b）用のプロンプト。
    画像と一緒に送られるテキスト指示を整形する。
    JSON形式ではなく自然文で返ってくることが多いため、
    シンプルに指示するだけにしておく。
    """
    if user_message:
        return f"以下の指示に従って画像を分析してください。\n\n{user_message}"
    return "この画像の内容を日本語で詳しく説明してください。"