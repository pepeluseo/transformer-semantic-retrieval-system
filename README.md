# Transformer-Based Semantic Retrieval System - Starter

## Welcome to Your Retrieval System Project! 🎯

This project will guide you through building and comparing different retrieval systems, from traditional keyword-based search to modern semantic search using transformers. You'll experience the evolution of information retrieval and understand the foundations of RAG (Retrieval-Augmented Generation) systems.

## Project Structure

```
project/starter/
├── src/                                       # Core retrieval modules
│   ├── data_loader.py                        # BeIR/Natural Questions dataset handler
│   ├── bm25_retriever.py                     # Traditional keyword-based retrieval
│   ├── word2vec_retriever.py                 # Static embedding retrieval
│   ├── transformer_retriever.py              # Semantic transformer retrieval
│   ├── evaluator.py                          # IR metrics calculation
│   └── utils.py                              # Utility functions
├── tests/                                     # Unit test suite for validation
│   ├── test_bm25_retriever.py                # BM25 retriever tests
│   ├── test_transformer_retriever.py         # Transformer retriever tests  
│   ├── test_word2vec_retriever.py            # Word2Vec retriever tests
│   └── test_evaluator.py                     # IR metrics evaluator tests
├── streamlit/                                 # Interactive RAG demo application
│   ├── app.py                                # Streamlit comparison interface
│   ├── rag_system.py                         # RAG implementation with OpenAI
│   ├── demo_data.py                          # TechFlow AI demo corpus
│   └── .env.example                          # Environment variables template
├── notebooks/
│   └── unified_retrieval_comparison.ipynb    # Main analysis notebook
├── dataset/                                   # Natural Questions test dataset
├── .venv/                                     # Python virtual environment
└── requirements.txt                           # All dependencies
```

## Quick Start Guide

### **🔧 Environment Setup (Already Done!)**
Your environment is ready with all dependencies installed. To verify:

```bash
# Navigate to starter directory
cd project/starter

# Check that virtual environment is working
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Verify installation
python -c "import sentence_transformers; print('✅ All dependencies ready!')"
```

### **🧪 Validate Your Implementation**
As you complete each component, run the corresponding tests to validate your work:

```bash
# Run all tests (initially many will fail - this is expected!)
python -m pytest tests/ -v

# Test individual components as you implement them
python tests/test_bm25_retriever.py        # BM25 keyword search implementation
python tests/test_evaluator.py             # IR metrics calculation
python tests/test_word2vec_retriever.py     # Word2Vec embedding implementation  
python tests/test_transformer_retriever.py  # Transformer semantic search

# Final validation - all tests should pass when complete
python -m pytest tests/ -v
```

**Expected progression:**
- **Initially**: Many tests fail with helpful error messages
- **During implementation**: Tests pass as you complete each section
- **Final result**: All tests pass, confirming your implementation

### **🎮 Try the Interactive Demo**
```bash
# Navigate to streamlit directory
cd streamlit

# Configure OpenAI API (optional)
cp .env.example .env
# Edit .env: OPENAI_API_KEY=your_openai_api_key_here

# Launch demo
streamlit run app.py
```

### **🔬 Run the Analysis**
```bash
# Start Jupyter for the main analysis
jupyter lab notebooks/unified_retrieval_comparison.ipynb
```

## Implementation Guide

### **📚 Source Files You'll Need to Implement**

Your task is to complete the missing implementations in the core retrieval modules. Each file contains detailed "YOUR CODE HERE" sections with guidance:

#### **🔧 `src/bm25_retriever.py` - Traditional Keyword Search**
**What you'll implement:**
- **Text tokenization** and preprocessing for BM25 scoring
- **BM25 index creation** using the rank-bm25 library
- **Retrieval logic** to find and rank top-k documents
- **Understanding**: Learn tf-idf concepts and traditional IR methods

**Key concepts:** Tokenization, BM25 scoring, keyword matching, document frequency

#### **🔤 `src/word2vec_retriever.py` - Static Word Embeddings**
**What you'll implement:**
- **Text preprocessing** for Word2Vec training (stopwords, tokenization)
- **Word2Vec model training** with gensim library
- **Document vectorization** by averaging word embeddings
- **Similarity search** using cosine similarity
- **Parameter optimization** through grid search experiments

**Key concepts:** Word embeddings, vector averaging, cosine similarity, hyperparameter tuning

#### **🤖 `src/transformer_retriever.py` - Modern Semantic Search**
**What you'll implement:**
- **Corpus encoding** using sentence transformers
- **Query encoding** with the same transformer model
- **Semantic similarity computation** and top-k retrieval
- **Understanding**: Experience state-of-the-art semantic search

**Key concepts:** Sentence embeddings, contextual understanding, transformer models, semantic similarity

#### **📊 `src/evaluator.py` - Information Retrieval Metrics**
**What you'll implement:**
- **Recall@k calculation**: Fraction of relevant documents found
- **Precision@k calculation**: Fraction of retrieved documents that are relevant  
- **Mean Reciprocal Rank (MRR)**: Quality of first relevant result
- **Understanding**: Learn how to measure retrieval system performance

**Key concepts:** Evaluation metrics, relevance judgments, performance measurement

#### **📁 `src/data_loader.py` - Already Complete!**
This file handles the complex BeIR dataset loading and preprocessing. **No implementation needed** - focus on the retrieval algorithms!

#### **🛠️ `src/utils.py` - Already Complete!**
Contains utility functions for text processing and system operations. **No implementation needed**.

### **🎯 Learning Progression**

**Start with:** `bm25_retriever.py` (traditional approach, simpler concepts)
**Continue with:** `evaluator.py` (understand how to measure success) 
**Progress to:** `word2vec_retriever.py` (static embeddings, more complex)
**Finish with:** `transformer_retriever.py` (modern semantic search)

### **🧪 Testing Your Implementation**

Each component has focused tests that will **fail until you implement the core functionality**:

```bash
# Test individual components as you implement them
python tests/test_bm25_retriever.py        # Should show 4/5 tests failing initially
python tests/test_evaluator.py             # Should show 4/5 tests failing initially  
python tests/test_word2vec_retriever.py     # Should show 4/5 tests failing initially
python tests/test_transformer_retriever.py  # Should show 4/5 tests failing initially
```

**Success indicators:**
- Tests pass once you complete the implementation
- Error messages guide you to missing functionality
- Each test validates core concepts you need to understand

## Learning Path

### **Phase 1: Understanding the Components** 🧭

**Explore the retrievers in `src/`:**
1. **`bm25_retriever.py`** - Traditional keyword matching
2. **`word2vec_retriever.py`** - Static word embeddings
3. **`transformer_retriever.py`** - Modern semantic search

**Key questions to explore:**
- How does each method represent documents and queries?
- What are the trade-offs between speed and semantic understanding?
- When would you use each approach in practice?

### **Phase 2: Interactive Exploration** 🎯

**Use the Streamlit demo to:**
- Compare how different methods retrieve documents
- See how retrieval quality affects RAG answers
- Test with your own queries about TechFlow AI


### **Phase 3: Quantitative Analysis** 📊

**In the Jupyter notebook:**
- Understand IR evaluation metrics (Recall@k, Precision@k, MRR)
- Compare performance on the Natural Questions dataset
- Analyze when each method succeeds or fails

## Test Suite Overview

### **Quality Assurance: Unit Tests** ✅

Your implementation is validated by a comprehensive test suite:

- **Core Functionality**: Indexing, retrieval, ranking for all methods
- **Semantic Understanding**: Synonym handling, context awareness
- **Edge Cases**: Empty queries, out-of-vocabulary words
- **Consistency**: Deterministic behavior, proper parameter handling
- **Integration**: Component interaction and data flow

### **Expected Test Results**
```bash
🎯 INITIAL STATE (before implementation):
❌ ~16-20 tests FAILING (expected - implementations incomplete)
✅ ~5 tests PASSING (initialization and setup tests)

🚀 FINAL STATE (after complete implementation):
✅ 20 tests PASSING across all components
❌ 0 tests FAILED

Components to implement:
- BM25Retriever: 4 implementation sections → 5 tests ✅
- TransformerRetriever: 2 implementation sections → 5 tests ✅
- Word2VecRetriever: 8 implementation sections → 5 tests ✅
- IRMetrics: 3 implementation sections → 5 tests ✅
```

## Key Concepts You'll Learn

### **🔍 Information Retrieval Evolution**
1. **Keyword-based (BM25)**: Fast, exact matching, good for technical terms
2. **Static Embeddings (Word2Vec)**: Basic semantic understanding, resource-efficient
3. **Contextual Embeddings (Transformers)**: Deep semantic understanding, best quality

### **📈 Evaluation Metrics**
- **Recall@k**: What fraction of relevant documents are retrieved?
- **Precision@k**: What fraction of retrieved documents are relevant?
- **MRR (Mean Reciprocal Rank)**: How quickly do we find the first relevant document?

### **🏗️ RAG Pipeline**
- **Retrieval**: Find relevant context documents
- **Augmentation**: Combine query + retrieved context  
- **Generation**: Use LLM to generate informed answers


### **Real-World Applications:**
- **Enterprise Search**: Internal knowledge bases
- **Customer Support**: AI chatbots with document retrieval
- **Research Tools**: Academic paper search
- **E-commerce**: Product search and recommendations
- **Developer Tools**: Code search and completion