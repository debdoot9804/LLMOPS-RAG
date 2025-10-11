"""
Simplified Document Retriever Module

This module provides a clean interface for retrieving and querying documents
using vector embeddings and RAG (Retrieval Augmented Generation).
It maintains session-specific document retrieval for multi-user applications.

Search Types:
- similarity: Standard cosine similarity search with score threshold filtering
- mmr: Maximal Marginal Relevance - balances relevance with diversity to avoid redundant results
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Azure OpenAI
from azure_clients import get_embedding_client, create_azure_openai_client
from langchain_openai import AzureChatOpenAI

# Custom modules
from multi_doc_chat.src.ingestion import DocumentIngestionPipeline
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.prompts.prompt_library import (
    RAG_QA_PROMPT,
    DOCUMENT_SUMMARIZATION_PROMPT,
    REASONING_QA_PROMPT,
    INFORMATION_EXTRACTION_PROMPT
)

# Initialize logger
logger = get_logger(__file__)


class DocumentRetriever:
    """
    A simplified retriever for accessing and querying document embeddings in a vector store.
    Maintains session-specific document retrieval and manages the RAG pipeline.
    """
    
    def __init__(
        self,
        vector_store_path: str = "vector_store",
        session_id: str = None,
        top_k: int = 4,
        score_threshold: float = 0.3,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store_path: Base path to the vector store directory
            session_id: Unique identifier for the user session
            top_k: Number of documents to retrieve for each query
            score_threshold: Minimum similarity score for retrieved documents
            search_type: Type of search to perform ("similarity" or "mmr")
                - "similarity": Standard similarity search
                - "mmr": Maximal Marginal Relevance (balances relevance with diversity)
            fetch_k: Number of documents to fetch before MMR reranking (only used for MMR)
            lambda_mult: Diversity factor for MMR (0=max diversity, 1=max relevance, only used for MMR)
        """
        # Set up session handling
        self.session_id = session_id or str(uuid.uuid4())[:12]
        
        # Determine vector store path based on session
        if session_id:
            self.vector_store_path = os.path.join(vector_store_path, f"session_{self.session_id}")
        else:
            self.vector_store_path = vector_store_path
        
        # Set up retrieval parameters
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.search_type = search_type if search_type in ["similarity", "mmr"] else "similarity"
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        
        # Initialize embedding model
        self.embedding_model = get_embedding_client()
        
        # Try to load the vector store
        self.vector_store = self._load_vector_store()
        
        # Initialize Azure OpenAI client for LLM
        self.llm = self._initialize_llm()
        
        logger.info(f"Initialized document retriever with session_id={self.session_id}, top_k={self.top_k}, search_type={self.search_type}")
    
    def _load_vector_store(self) -> Optional[FAISS]:
        """
        Load the vector store from disk.
        
        Returns:
            FAISS vector store if it exists, None otherwise
        """
        try:
            if os.path.exists(self.vector_store_path):
                vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded vector store from {self.vector_store_path}")
                return vector_store
            else:
                logger.warning("Vector store not found at %s", self.vector_store_path)
                return None
        except Exception as e:
            logger.error("Error loading vector store: %s", str(e))
            return None
    
    def _initialize_llm(self) -> AzureChatOpenAI:
        """
        Initialize the LLM for generating responses.
        
        Returns:
            Initialized LLM instance
        """
        try:
            # Using the Azure OpenAI client
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("API_VERSION"),
                deployment_name=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
                temperature=0.0
            )
            
            logger.info("Initialized LLM with model %s", os.getenv('OPENAI_CHAT_DEPLOYMENT_NAME'))
            return llm
            
        except Exception as e:
            logger.error("Error initializing LLM: %s", str(e))
            raise
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query using either similarity search or MMR.
        
        Args:
            query: The query string
            
        Returns:
            List of relevant document chunks
        """
        if self.vector_store is None:
            logger.warning("No vector store available. Please ingest documents first.")
            return []
        
        try:
            if self.search_type == "mmr":
                # Use Maximal Marginal Relevance for diversity
                docs = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=self.top_k,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult
                )
                
                # MMR doesn't return scores, so we'll add a placeholder
                for doc in docs:
                    doc.metadata["similarity_score"] = "N/A (MMR)"
                    doc.metadata["search_type"] = "mmr"
                
                logger.info("Retrieved %d documents using MMR (fetch_k=%d, lambda=%.1f)", 
                           len(docs), self.fetch_k, self.lambda_mult)
                return docs
                
            else:
                # Standard similarity search with scores
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
                
                # Filter by score threshold and extract just the documents
                filtered_docs = []
                for doc, score in docs_with_scores:
                    # Convert the score to similarity (if using cosine distance)
                    similarity = 1.0 - (score / 2.0)
                    
                    if similarity >= self.score_threshold:
                        # Add similarity score to metadata
                        doc.metadata["similarity_score"] = similarity
                        doc.metadata["search_type"] = "similarity"
                        filtered_docs.append(doc)
                
                logger.info("Retrieved %d relevant documents using similarity search (threshold=%.2f)", 
                           len(filtered_docs), self.score_threshold)
                return filtered_docs
            
        except Exception as e:
            logger.error("Error retrieving documents: %s", str(e))
            return []
    
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Format a list of documents into a single string.
        
        Args:
            docs: List of Document objects
            
        Returns:
            Formatted string containing document content
        """
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.replace('{', '{{').replace('}', '}}')  # Escape curly braces
            metadata = doc.metadata
            source = str(metadata.get("source", "Unknown")).replace('{', '{{').replace('}', '}}')  # Escape curly braces
            page = str(metadata.get("page", "")).replace('{', '{{').replace('}', '}}')  # Escape curly braces
            score = metadata.get("similarity_score", 0.0)
            # Ensure score is a float for formatting
            try:
                score_float = float(score)
            except (ValueError, TypeError):
                score_float = 0.0
            header = "Document {} (Source: {}, Page: {}, Relevance: {:.2f})".format(i, source, page, score_float)
            formatted_docs.append("{}\n{}\n".format(header, content))
        
        return "\n\n".join(formatted_docs)
    
    def query_documents(
        self, 
        query: str, 
        prompt_type: str = "standard",
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, List[Document]]:
        """
        A simple interface for querying documents using LangChain LCEL (LangChain Expression Language).
        
        Args:
            query: The query string
            prompt_type: Type of prompt to use ("standard", "reasoning", "summary", "extraction")
            chat_history: Optional list of (human_message, ai_message) tuples for conversation context
            
        Returns:
            A tuple containing (response_text, list_of_relevant_documents)
        """
        docs = self.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found to answer your query.", []
        
        # Select the appropriate prompt based on the prompt type
        prompt_map = {
            "reasoning": REASONING_QA_PROMPT,
            "summary": DOCUMENT_SUMMARIZATION_PROMPT,
            "extraction": INFORMATION_EXTRACTION_PROMPT,
            "standard": RAG_QA_PROMPT
        }
        prompt = prompt_map.get(prompt_type, RAG_QA_PROMPT)
        
        try:
            # Format chat history for the prompt
            formatted_history = []
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    # Escape curly braces in chat messages
                    safe_human_msg = human_msg.replace('{', '{{').replace('}', '}}')
                    safe_ai_msg = ai_msg.replace('{', '{{').replace('}', '}}')
                    formatted_history.extend([
                        ("human", safe_human_msg),
                        ("assistant", safe_ai_msg)
                    ])
            
            # Escape curly braces in query
            safe_query = query.replace('{', '{{').replace('}', '}}')
            
            # Create chain using RunnableParallel and RunnablePassthrough
            
            if prompt_type == "extraction":
                # Extraction prompt uses different variable name
                chain = (
                    RunnableParallel(
                        context=lambda x: self._format_documents(x["docs"]),
                        extraction_query=RunnablePassthrough()
                    )
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                response = chain.invoke({"docs": docs, "extraction_query": safe_query})
            else:
                # Standard, reasoning, and summary prompts
                chain = (
                    RunnableParallel(
                        context=lambda x: self._format_documents(x["docs"]),
                        question=lambda x: x["question"],
                        chat_history=lambda x: x.get("chat_history", [])
                    )
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                response = chain.invoke({
                    "docs": docs,
                    "question": safe_query,
                    "chat_history": formatted_history
                })
            
            return response, docs
            
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return "Error: {}".format(str(e)), docs


# Example usage
if __name__ == "__main__":
    # Get data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    vector_store_path = os.path.join(data_dir, "vector_store")
    
    # Create a session ID
    session_id = str(uuid.uuid4())[:12]
    print(f"Using session ID: {session_id}")
    
    # Check if we need to ingest documents first
    if not os.path.exists(os.path.join(vector_store_path, f"session_{session_id}")):
        print("No vector store found for this session. Ingesting documents first...")
        
        # Define paths to documents
        file_paths = [
            os.path.join(data_dir, "Mindtree_offer.pdf"),
            # Add more file paths as needed
        ]
        
        # Create the ingestion pipeline with the same session ID
        ingestion_pipeline = DocumentIngestionPipeline(
            chunk_size=2000,
            chunk_overlap=400,
            vector_store_path=vector_store_path,
            session_id=session_id
        )
        
        # Process documents
        ingestion_pipeline.process_documents(
            file_paths=file_paths,
            metadata={"source": "initial_ingestion", "date": "2025-10-09"}
        )
    
    # Create and use the retriever with the same session ID
    # Example 1: Using similarity search (default)
    retriever_similarity = DocumentRetriever(
        vector_store_path=vector_store_path,
        session_id=session_id,
        top_k=3,
        search_type="similarity",
        score_threshold=0.3
    )
    
    # Example 2: Using MMR for diverse results
    retriever_mmr = DocumentRetriever(
        vector_store_path=vector_store_path,
        session_id=session_id,
        top_k=3,
        search_type="mmr",
        fetch_k=20,  # Fetch 20 candidates before reranking
        lambda_mult=0.5  # Balance between relevance (1.0) and diversity (0.0)
    )
    
    # Choose which retriever to use
    retriever = retriever_mmr  # Change to retriever_similarity to test similarity search
    
    # Test different query types
    test_scenarios = [
        {"query": "What is the compensation package?", "prompt_type": "standard"},
        {"query": "What are the benefits provided?", "prompt_type": "reasoning"},
        {"query": "Summarize the key details of the offer", "prompt_type": "summary"},
        {"query": "Extract information about notice period and joining date", "prompt_type": "extraction"}
    ]
    
    for scenario in test_scenarios:
        query = scenario["query"]
        prompt_type = scenario["prompt_type"]
        
        print(f"\n\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"PROMPT TYPE: {prompt_type}")
        print(f"{'='*50}")
        
        response, docs = retriever.query_documents(query, prompt_type=prompt_type)
        
        print("\nRESPONSE:")
        print("-" * 40)
        print(response)
        
        print("\nRELEVANT DOCUMENTS:")
        print("-" * 40)
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else doc.page_content)
            print(f"Metadata: {doc.metadata}")
            print("-" * 30)
