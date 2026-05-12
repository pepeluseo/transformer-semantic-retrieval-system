"""
Simple RAG System using direct OpenAI API calls
"""
import os
import openai
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    """Simple RAG system for answer generation using different retrievers."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize RAG system."""
        self.api_key = openai_api_key
        self.model = model
        self.temperature = temperature
        
        # Set up OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=openai_api_key)
        except ImportError:
            # Fallback for older openai versions
            openai.api_key = openai_api_key
            self.client = None
    
    def generate_answer(self, query: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Generate answer using retrieved documents."""
        try:
            # Create context from retrieved documents
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
            
            # Create prompt
            prompt = f"""Based on the following documents, please answer the question.

Documents:
{context}

Question: {query}

Instructions:
- Use only information from the provided documents
- If the answer is not in the documents, say so
- Be concise and accurate
- Cite which documents you used

Answer:"""
            
            # Generate response
            if self.client:
                # New API style
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            else:
                # Legacy API style
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided documents."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'context_used': context,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Sorry, I encountered an error generating the answer: {str(e)}",
                'context_used': "",
                'success': False,
                'error': str(e)
            }
    
    def compare_rag_methods(self, query: str, retrievers_dict: Dict[str, Any], 
                           corpus_texts: List[str], k: int = 3) -> Dict[str, Dict]:
        """Compare RAG results across different retrieval methods."""
        results = {}
        
        for method_name, retriever in retrievers_dict.items():
            try:
                # Get retrieval results
                retrieval_results = retriever.retrieve([query], k=k)
                retrieved_doc_indices = retrieval_results[0] if retrieval_results else []
                
                # Get retrieved document texts
                retrieved_docs = [corpus_texts[idx] for idx in retrieved_doc_indices 
                                if idx < len(corpus_texts)]
                
                # Generate answer
                rag_result = self.generate_answer(query, retrieved_docs)
                
                results[method_name] = {
                    'answer': rag_result['answer'],
                    'retrieved_indices': retrieved_doc_indices,
                    'retrieved_docs': retrieved_docs,
                    'context_used': rag_result['context_used'],
                    'success': rag_result['success'],
                    'error': rag_result['error']
                }
                
            except Exception as e:
                logger.error(f"Error with {method_name}: {e}")
                results[method_name] = {
                    'answer': f"Error with {method_name}: {str(e)}",
                    'retrieved_indices': [],
                    'retrieved_docs': [],
                    'context_used': "",
                    'success': False,
                    'error': str(e)
                }
        
        return results
