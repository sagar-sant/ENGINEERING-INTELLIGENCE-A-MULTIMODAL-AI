from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProcessedPage:
    """Represents one processed page in an engineering document."""

    page_number: int
    text: str
    extraction_method: str
    character_count: int
    ocr_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the page into a dictionary."""
        return asdict(self)


@dataclass
class ProcessedDocument:
    """Represents a complete processed engineering document."""

    file_name: str
    file_path: str
    page_count: int
    metadata: dict[str, Any]
    pages: list[ProcessedPage]
    visual_metadata: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the document into a dictionary."""
        return {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "pages": [page.to_dict() for page in self.pages],
            "visual_metadata": [
                asdict(item) for item in self.visual_metadata
            ],
        }