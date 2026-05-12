"""
Semantic Retrieval Project - Core Modules

This package contains the core modules for building a semantic retrieval system:
- data_loader: Load and preprocess datasets
- embedding_generator: Generate embeddings using transformer models
- retriever: Implement semantic search and retrieval
- evaluator: Evaluate retrieval performance
"""

from .data_loader import DataLoader
from .embedding_generator import EmbeddingGenerator, RECOMMENDED_MODELS
from .retriever import SemanticRetriever
from .evaluator import RetrievalEvaluator

__version__ = "1.0.0"
__author__ = "Udacity Deep Learning Nanodegree"

__all__ = [
    "DataLoader",
    "EmbeddingGenerator", 
    "SemanticRetriever",
    "RetrievalEvaluator",
    "RECOMMENDED_MODELS"
]
