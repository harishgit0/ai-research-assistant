import re


def clean_text(text: str, page_number: int | None = None) -> str:
    """
    Clean text extracted from a PDF while preserving
    meaningful content.
    """

    # Normalize newline styles
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        # Remove completely empty lines
        if not line:
            continue

        # Remove standalone extraction artifacts
        if line in {"•", "·", "-", "–"}:
            continue

        # Collapse multiple spaces/tabs
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    # Remove a page number if it appears as the
    # final line of the extracted page.
    if page_number is not None and cleaned_lines:
        if cleaned_lines[-1] == str(page_number):
            cleaned_lines.pop()

    return "\n".join(cleaned_lines)