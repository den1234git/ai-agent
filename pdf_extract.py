"""
チャットに添付されたPDFファイル（base64のdata URL文字列）からテキストを抽出する。

画像がFileReaderでbase64化されてvisionモデルに渡されるのと同じパターンで、
PDFもフロント側でbase64化して送ってもらい、サーバー側でpdfplumberを使って
テキスト抽出してから reference_files（コード生成の参考資料）に合流させる。

生成AIに直接PDFバイナリを読ませることはできないため、
「テキスト化してからreference_filesとして扱う」というアプローチを取っている。
"""

import base64
import io

import pdfplumber

MAX_PDF_PAGES = 30  # 巨大なPDFでプロンプトが肥大化しすぎないための上限


def extract_pdf_text_from_base64(data_url_or_base64: str, max_pages: int = MAX_PDF_PAGES) -> str:
    """
    data URL形式（"data:application/pdf;base64,...."）または素のbase64文字列から
    PDFのテキストを抽出して返す。

    戻り値：抽出したテキスト（失敗時は空文字ではなくエラーメッセージ文字列を返し、
    reference_filesの中身としてそのままLAに見えるようにする＝失敗が黙って消えないように）。
    """
    try:
        raw = data_url_or_base64
        if "," in raw and raw.strip().lower().startswith("data:"):
            raw = raw.split(",", 1)[1]

        pdf_bytes = base64.b64decode(raw)

        texts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = pdf.pages[:max_pages]
            for i, page in enumerate(pages_to_read):
                text = page.extract_text() or ""
                if text.strip():
                    texts.append(f"--- page {i + 1} ---\n{text.strip()}")

            if total_pages > max_pages:
                texts.append(f"\n（※ 全{total_pages}ページ中、先頭{max_pages}ページのみ抽出しています）")

        joined = "\n\n".join(texts).strip()
        if not joined:
            return "（PDFからテキストを抽出できませんでした。スキャン画像のPDFの可能性があります）"
        return joined

    except Exception as e:
        return f"（PDFの読み込みに失敗しました: {e}）"