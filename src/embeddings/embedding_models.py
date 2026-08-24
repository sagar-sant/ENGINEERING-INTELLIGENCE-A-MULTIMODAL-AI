from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ChunkEmbedding:
    """Represents an embedding associated with a document chunk."""

    chunk_id: str
    model_name: str
    dimensions: int
    vector: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the embedding into a dictionary."""
        return asdict(self)