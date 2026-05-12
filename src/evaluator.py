"""
Evaluation Metrics for Information Retrieval
"""
import numpy as np
from typing import Dict, List


class IRMetrics:
    """Information Retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Recall@k: fraction of relevant documents found in top-k.

        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation

        Returns:
            Average Recall@k across all queries.
        """
        recall_scores = []

        for q_id, retrieved_docs in results.items():
            if q_id not in qrels:
                continue

            relevant_docs = set(qrels[q_id].keys())

            if len(relevant_docs) == 0:
                continue

            top_k_docs = retrieved_docs[:k]

            hits = 0
            for doc_id in top_k_docs:
                if doc_id in relevant_docs:
                    hits += 1

            recall = hits / len(relevant_docs)
            recall_scores.append(recall)

        return float(np.mean(recall_scores)) if recall_scores else 0.0

    @staticmethod
    def precision_at_k(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]], k: int) -> float:
        """
        Calculate Precision@k: fraction of top-k retrieved documents that are relevant.

        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}
            k: cutoff for top-k evaluation

        Returns:
            Average Precision@k across all queries.
        """
        precision_scores = []

        for q_id, retrieved_docs in results.items():
            if q_id not in qrels:
                continue

            relevant_docs = set(qrels[q_id].keys())

            if len(relevant_docs) == 0:
                continue

            top_k_docs = retrieved_docs[:k]

            hits = 0
            for doc_id in top_k_docs:
                if doc_id in relevant_docs:
                    hits += 1

            precision = hits / k
            precision_scores.append(precision)

        return float(np.mean(precision_scores)) if precision_scores else 0.0

    @staticmethod
    def mrr(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR is the average reciprocal rank of the first relevant document
        retrieved for each query.

        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}

        Returns:
            Mean Reciprocal Rank.
        """
        reciprocal_ranks = []

        for q_id, retrieved_docs in results.items():
            if q_id not in qrels:
                continue

            relevant_docs = set(qrels[q_id].keys())

            if len(relevant_docs) == 0:
                continue

            reciprocal_rank = 0.0

            for rank, doc_id in enumerate(retrieved_docs, start=1):
                if doc_id in relevant_docs:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def evaluate_retrieval(results: Dict[int, List[int]], qrels: Dict[int, Dict[int, int]]) -> Dict[str, float]:
        """
        Comprehensive evaluation with standard IR metrics.

        Args:
            results: {query_id: [doc_ids]}
            qrels: {query_id: {doc_id: relevance_score}}

        Returns:
            Dictionary with metric names and values.
        """
        metrics = {
            'Recall@1': IRMetrics.recall_at_k(results, qrels, 1),
            'Recall@5': IRMetrics.recall_at_k(results, qrels, 5),
            'Recall@10': IRMetrics.recall_at_k(results, qrels, 10),
            'Precision@5': IRMetrics.precision_at_k(results, qrels, 5),
            'MRR': IRMetrics.mrr(results, qrels)
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
        """Pretty print evaluation metrics."""
        print(f"\n📊 {title}")
        print("=" * 40)
        for metric, value in metrics.items():
            print(f"{metric:12}: {value:.4f}")
        print("=" * 40)