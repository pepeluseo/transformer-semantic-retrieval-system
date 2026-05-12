"""
BM25 Retriever for keyword-based information retrieval.
"""

import re
from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retriever using BM25.

    This retriever tokenizes documents, builds a BM25 index,
    and retrieves the top-k most relevant documents for each query.
    """

    def __init__(self):
        self.bm25 = None
        self.corpus = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase word tokens.

        Args:
            text: Input text.

        Returns:
            List[str]: Tokenized text.
        """
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_index(self, corpus_texts: List[str]):
        """
        Build BM25 index from corpus texts.

        Args:
            corpus_texts: List of document strings.
        """
        print("🔧 Building BM25 index...")

        self.corpus = corpus_texts
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus_texts]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"✅ BM25 index built for {len(corpus_texts):,} documents")

    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """
        Retrieve top-k documents for each query.

        Args:
            query_texts: List of query strings.
            k: Number of documents to retrieve.

        Returns:
            Dict[int, List[int]]: Dictionary mapping query index to retrieved document indices.
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running BM25 retrieval for {len(query_texts)} queries...")

        results = {}

        for query_idx, query in enumerate(query_texts):
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)

            top_k_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]

            results[query_idx] = top_k_indices

        print(f"✅ Retrieved top-{k} documents using BM25")
        return results
