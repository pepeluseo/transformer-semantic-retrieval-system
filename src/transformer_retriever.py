"""
Transformer-based Retrieval using Sentence Embeddings
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        print(f"🤖 Loading transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        print(f"✅ Model loaded: {model_name}")

    def build_index(self, corpus_texts: List[str]):
        """
        Build semantic index by encoding corpus texts.

        Args:
            corpus_texts: List of document texts.
        """
        print("🧠 Building semantic index...")
        self.corpus = corpus_texts

        # Encode corpus using sentence transformer
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")

    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """
        Retrieve top-k documents using semantic similarity.

        Args:
            query_texts: List of query texts.
            k: Number of documents to retrieve.

        Returns:
            Dict[int, List[int]]: Dictionary mapping query index to retrieved document indices.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")

        # Encode query texts using the transformer model
        query_embeddings = self.model.encode(
            query_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Calculate cosine similarities between queries and documents
        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)

        # Retrieve top-k documents for each query
        results = {}

        for query_idx, query_similarities in enumerate(similarities):
            top_k_indices = np.argsort(query_similarities)[::-1][:k]
            results[query_idx] = top_k_indices.tolist()

        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
