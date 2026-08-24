from dataclasses import dataclass


@dataclass
class ChunkQuality:
    """Quality indicators for a document chunk."""

    character_count: int
    word_count: int
    sentence_count: int
    has_heading: bool
    has_numbered_list: bool
    has_warning: bool
    has_table_like_content: bool

    @property
    def quality_score(self) -> float:
        """Calculate a simple heuristic quality score."""

        score = 0.0

        if self.character_count >= 200:
            score += 0.25

        if self.word_count >= 40:
            score += 0.20

        if self.sentence_count >= 2:
            score += 0.15

        if self.has_heading:
            score += 0.15

        if self.has_numbered_list:
            score += 0.10

        if self.has_warning:
            score += 0.10

        if self.has_table_like_content:
            score += 0.05

        return min(score, 1.0)


def analyze_chunk_quality(text: str) -> ChunkQuality:
    """
    Analyze basic structural characteristics of a text chunk.

    This is a heuristic quality assessment. It does not determine
    whether the underlying engineering information is correct.
    """

    stripped_text = text.strip()

    if not stripped_text:
        return ChunkQuality(
            character_count=0,
            word_count=0,
            sentence_count=0,
            has_heading=False,
            has_numbered_list=False,
            has_warning=False,
            has_table_like_content=False,
        )

    lines = [
        line.strip()
        for line in stripped_text.splitlines()
        if line.strip()
    ]

    words = stripped_text.split()

    sentence_count = sum(
        stripped_text.count(marker)
        for marker in [".", "!", "?"]
    )

    has_heading = False

    if lines:
        first_line = lines[0]

        if len(first_line) <= 120:
            if (
                first_line.isupper()
                or first_line.startswith("Chapter ")
                or first_line.startswith("Section ")
                or first_line[:1].isdigit()
            ):
                has_heading = True

    has_numbered_list = any(
        line[:2].rstrip(".").isdigit()
        or line[:3].rstrip(".").isdigit()
        for line in lines
    )

    warning_terms = (
        "WARNING",
        "CAUTION",
        "DANGER",
        "IMPORTANT",
        "NOTE",
    )

    upper_text = stripped_text.upper()

    has_warning = any(
        term in upper_text
        for term in warning_terms
    )

    has_table_like_content = any(
        line.count("|") >= 2
        or "\t" in line
        for line in lines
    )

    return ChunkQuality(
        character_count=len(stripped_text),
        word_count=len(words),
        sentence_count=sentence_count,
        has_heading=has_heading,
        has_numbered_list=has_numbered_list,
        has_warning=has_warning,
        has_table_like_content=has_table_like_content,
    )