"""
Tavily Web検索APIのラッパー。

TAVILY_API_KEY は .env に設定し、app.py起動時の load_dotenv() で
環境変数に読み込まれている前提（.env の場所・作り方は .env.example 参照）。
キー未設定の場合は available=False を返し、呼び出し側（router.py の
run_web_search）が「検索機能が未設定です」という案内をユーザーに
返せるようにする（例外を投げてルーティング全体を落とさない）。
"""
import os
import requests

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 15


def tavily_search(query: str, max_results: int = 5) -> dict:
    """
    Tavily APIで検索を実行する。

    戻り値:
      {
        "available": bool,       # APIキーが設定されているか
        "error": str | None,     # 失敗時のエラー内容（available=Trueでも起きうる：通信失敗等）
        "answer": str | None,    # Tavilyが生成した簡潔な回答（include_answer=True時）
        "results": [
          {"title": str, "url": str, "content": str}, ...
        ],
      }
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {
            "available": False,
            "error": "TAVILY_API_KEYが設定されていません。.envを確認してください。",
            "answer": None,
            "results": [],
        }

    try:
        resp = requests.post(
            TAVILY_API_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": max_results,
            },
            timeout=TAVILY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[WEB_SEARCH] Tavily API呼び出しに失敗しました: {e}")
        return {
            "available": True,
            "error": f"Web検索に失敗しました: {e}",
            "answer": None,
            "results": [],
        }
    except Exception as e:
        print(f"[WEB_SEARCH] 予期しないエラー: {e}")
        return {
            "available": True,
            "error": f"Web検索中に予期しないエラーが発生しました: {e}",
            "answer": None,
            "results": [],
        }

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]

    return {
        "available": True,
        "error": None,
        "answer": data.get("answer"),
        "results": results,
    }
