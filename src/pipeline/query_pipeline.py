from src.rag.rag_pipeline import RAGPipeline


class DocumentQueryPipeline:
    """
    End-to-end query pipeline for engineering documents.

    Takes a user question, retrieves relevant document chunks,
    builds the RAG context, and optionally generates an answer.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
    ) -> None:
        self.rag_pipeline = rag_pipeline

    def query(
        self,
        question: str,
        top_k: int = 5,
    ):
        """
        Answer a question using the configured RAG pipeline.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        return self.rag_pipeline.generate(
            query=question,
            top_k=top_k,
        )