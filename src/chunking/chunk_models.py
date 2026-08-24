from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DocumentChunk:
    """Represents a searchable chunk of an engineering document."""

    chunk_id: str
    document_name: str
    page_number: int
    text: str
    character_count: int
    start_character: int
    end_character: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the chunk into a dictionary."""
        return asdict(self)