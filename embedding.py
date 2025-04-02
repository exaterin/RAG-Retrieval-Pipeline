from sentence_transformers import SentenceTransformer
from typing import List
import torch


class EmbeddingModel:
    """
    Embedding model using SentenceTransformer.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a batch of texts.
        """
        return self.model.encode(texts, convert_to_tensor=True)
