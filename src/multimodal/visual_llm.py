from pathlib import Path
from typing import Optional


class VisualLLMAnalyzer:
    """
    Uses a multimodal LLM to describe rendered PDF pages.

    The analyzer is intentionally separate from the PDF
    structural detector. The detector decides which pages
    deserve visual analysis; this class performs semantic
    interpretation of those pages.
    """

    def __init__(
        self,
        llm,
    ) -> None:
        self.llm = llm

    def analyze_page(
        self,
        page_path: str | Path,
        page_number: int,
        surrounding_text: str = "",
    ) -> str:
        """
        Ask the multimodal LLM to describe visible visual
        information on a PDF page.
        """

        page_path = Path(page_path)

        if not page_path.exists():
            raise FileNotFoundError(
                f"Rendered page not found: {page_path}"
            )

        prompt = f"""
You are analyzing page {page_number} of an engineering document.

Inspect the provided page image carefully.

Identify ONLY visual information that is actually visible
or clearly readable on the page.

Report:

- figure numbers and figure titles
- diagrams
- illustrations
- charts
- tables
- labels
- connectors
- components
- ports
- symbols
- callouts
- visible relationships between labeled components
- other meaningful engineering visual information

Do not infer information that cannot be seen.

Do not invent component names, labels, figure numbers,
or relationships.

If the page contains mostly text and no meaningful visual
information, say so.

Return a concise but information-rich description suitable
for document retrieval.

Page number: {page_number}

Nearby extracted text, which may help identify captions
but must NOT override what is visually visible:

{surrounding_text[:4000]}
""".strip()

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return self.llm.generate_with_images(
            messages,
            [str(page_path)],
        ).strip()