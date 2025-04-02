from typing import List, Tuple
import torch
from sentence_transformers import util


class RetrievalPipeline:
    """
    Handles retrieval of relevant text chunks based on query embeddings.
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(self, query: str, corpus_chunks: List[str], corpus_embeddings: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks based on cosine similarity.
        """
        query_embedding = self.embedder.embed([query])
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(similarities, k=top_k)

        retrieved_chunks = [(corpus_chunks[idx], similarities[idx].item()) for idx in top_results.indices]
        return retrieved_chunks
