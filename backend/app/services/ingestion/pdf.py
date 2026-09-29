from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list containing the page number and
    extracted text for each page.
    """

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return pages