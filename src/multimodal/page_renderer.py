from pathlib import Path

import pymupdf


def render_page(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path = "data/processed/page_images",
    dpi: int = 150,
) -> Path:
    """
    Render one PDF page as a PNG image.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_number:
        1-based PDF page number.
    output_dir:
        Directory where the rendered image will be saved.
    dpi:
        Rendering resolution.

    Returns
    -------
    Path
        Path to the generated PNG image.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as document:
        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} does not exist. "
                f"Document contains {len(document)} pages."
            )

        page = document[page_number - 1]

        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        output_path = (
            output_dir
            / f"{pdf_path.stem}_page_{page_number:03d}.png"
        )

        pixmap.save(output_path)

    return output_path

def render_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
    output_dir: str | Path = "data/processed/page_images",
    dpi: int = 150,
) -> list[Path]:
    """
    Render multiple selected PDF pages as PNG images.
    """

    rendered_paths: list[Path] = []

    for page_number in page_numbers:
        rendered_paths.append(
            render_page(
                pdf_path=pdf_path,
                page_number=page_number,
                output_dir=output_dir,
                dpi=dpi,
            )
        )

    return rendered_paths