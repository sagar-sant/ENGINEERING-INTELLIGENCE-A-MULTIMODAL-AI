from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class ExtractedImage:
    """Metadata for an image extracted from a PDF."""

    page_number: int
    image_index: int
    file_path: str
    width: int
    height: int
    extension: str


def extract_page_images(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path = "data/processed/embedded_images",
) -> list[ExtractedImage]:
    """
    Extract all embedded raster images from one PDF page.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF.
    page_number:
        1-based page number.
    output_dir:
        Directory where extracted images will be saved.

    Returns
    -------
    list[ExtractedImage]
        Metadata for each extracted image.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if page_number < 1:
        raise ValueError("page_number must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_images: list[ExtractedImage] = []

    with pymupdf.open(pdf_path) as document:
        if page_number > len(document):
            raise ValueError(
                f"Page {page_number} does not exist. "
                f"Document contains {len(document)} pages."
            )

        page = document[page_number - 1]
        images = page.get_images(full=True)

        for image_index, image_info in enumerate(images, start=1):
            xref = image_info[0]

            image_data = document.extract_image(xref)

            extension = image_data["ext"]
            image_bytes = image_data["image"]
            width = image_data["width"]
            height = image_data["height"]

            output_path = (
                output_dir
                / (
                    f"{pdf_path.stem}"
                    f"_page_{page_number:03d}"
                    f"_image_{image_index:02d}"
                    f".{extension}"
                )
            )

            output_path.write_bytes(image_bytes)

            extracted_images.append(
                ExtractedImage(
                    page_number=page_number,
                    image_index=image_index,
                    file_path=str(output_path),
                    width=width,
                    height=height,
                    extension=extension,
                )
            )

    return extracted_images