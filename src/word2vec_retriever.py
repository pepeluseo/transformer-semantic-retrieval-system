"""
Word2Vec Retriever for static embedding-based information retrieval.
"""

import re
import numpy as np
from typing import List, Dict, Any
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class Word2VecRetriever:
    """
    Static embedding retriever using Word2Vec.

    This retriever:
    - tokenizes corpus texts
    - trains a Word2Vec model
    - converts documents and queries into average word vectors
    - retrieves top-k documents using cosine similarity
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 1
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg

        self.model = None
        self.corpus = None
        self.tokenized_corpus = None
        self.corpus_embeddings = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase word tokens.
        """
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _text_to_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Convert a list of tokens into a single vector by averaging Word2Vec vectors.
        """
        if self.model is None:
            raise ValueError("Word2Vec model not trained. Call build_index() first.")

        vectors = []

        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def build_index(self, corpus_texts: List[str]):
        """
        Build Word2Vec index from corpus texts.
        """
        print("🧠 Training Word2Vec model...")

        self.corpus = corpus_texts
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus_texts]

        self.model = Word2Vec(
            sentences=self.tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=self.sg
        )

        print("📄 Creating document vectors...")

        self.corpus_embeddings = np.vstack([
            self._text_to_vector(tokens)
            for tokens in self.tokenized_corpus
        ])

        print(f"✅ Word2Vec index built for {len(corpus_texts):,} documents")

    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """
        Retrieve top-k documents using cosine similarity between query and document vectors.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"🔍 Running Word2Vec retrieval for {len(query_texts)} queries...")

        tokenized_queries = [self._tokenize(query) for query in query_texts]

        query_embeddings = np.vstack([
            self._text_to_vector(tokens)
            for tokens in tokenized_queries
        ])

        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)

        results = {}

        for query_idx, query_similarities in enumerate(similarities):
            top_k_indices = np.argsort(query_similarities)[::-1][:k]
            results[query_idx] = top_k_indices.tolist()

        print(f"✅ Retrieved top-{k} documents using Word2Vec")
        return results

    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Return vocabulary statistics.

        Important:
        This method returns 'vocabulary_size', because the notebook expects that key.
        """
        if self.model is None:
            return {
                "vocabulary_size": 0,
                "vector_size": self.vector_size,
                "window": self.window,
                "min_count": self.min_count,
                "epochs": self.epochs
            }

        return {
            "vocabulary_size": len(self.model.wv),
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs
        }

    def optimize_hyperparameters(
        self,
        corpus_texts: List[str],
        query_texts: List[str],
        qrels_dict: Dict,
        evaluator_class,
        param_grid: List[Dict[str, Any]] = None,
        k: int = 20
    ) -> Dict[str, Any]:
        """
        Simple hyperparameter optimization for Word2Vec.

        Tries several configurations and keeps the one with the highest Recall@10.
        """
        if param_grid is None:
            param_grid = [
                {"vector_size": 100, "window": 5, "min_count": 1, "epochs": 5},
                {"vector_size": 200, "window": 10, "min_count": 1, "epochs": 10},
                {"vector_size": 300, "window": 10, "min_count": 1, "epochs": 10},
            ]

        best_score = -1.0
        best_params = None
        best_results = None

        for params in param_grid:
            print(f"Testing Word2Vec params: {params}")

            candidate = Word2VecRetriever(
                vector_size=params.get("vector_size", self.vector_size),
                window=params.get("window", self.window),
                min_count=params.get("min_count", self.min_count),
                epochs=params.get("epochs", self.epochs),
                workers=self.workers,
                sg=self.sg
            )

            candidate.build_index(corpus_texts)
            results = candidate.retrieve(query_texts, k=k)
            metrics = evaluator_class.evaluate_retrieval(results, qrels_dict)

            score = metrics.get("Recall@10", 0.0)

            if score > best_score:
                best_score = score
                best_params = params
                best_results = results

                self.vector_size = candidate.vector_size
                self.window = candidate.window
                self.min_count = candidate.min_count
                self.epochs = candidate.epochs
                self.model = candidate.model
                self.corpus = candidate.corpus
                self.tokenized_corpus = candidate.tokenized_corpus
                self.corpus_embeddings = candidate.corpus_embeddings

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_results": best_results
        }
