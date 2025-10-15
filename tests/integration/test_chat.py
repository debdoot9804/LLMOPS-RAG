"""
Integration tests for chat functionality - Simple focused tests.
"""
import pytest
from unittest.mock import Mock, patch
from langchain.schema import Document


class TestDocumentRetrieval:
    """Test document retrieval works correctly."""
    
    @patch('azure_clients.create_azure_openai_client')
    @patch('azure_clients.get_embedding_client')
    @patch('multi_doc_chat.src.retriever.FAISS')
    def test_similarity_search_returns_scored_docs(self, mock_faiss, mock_embeddings, mock_llm):
        """Test similarity search adds relevance scores."""
        from multi_doc_chat.src.retriever import DocumentRetriever
        
        retriever = DocumentRetriever("/tmp/test", "test-session", search_type="similarity")
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="Test", metadata={"source": "test.txt"}), 0.95)
        ]
        retriever.vector_store = mock_vector_store
        
        docs = retriever.get_relevant_documents("query")
        assert docs[0].metadata["similarity_score"] == 0.95
    
    @patch('azure_clients.create_azure_openai_client')
    @patch('azure_clients.get_embedding_client')
    @patch('multi_doc_chat.src.retriever.FAISS')
    def test_mmr_search_called(self, mock_faiss, mock_embeddings, mock_llm):
        """Test MMR search is invoked."""
        from multi_doc_chat.src.retriever import DocumentRetriever
        
        retriever = DocumentRetriever("/tmp/test", "test-session", search_type="mmr")
        
        mock_vector_store = Mock()
        mock_vector_store.max_marginal_relevance_search.return_value = [
            Document(page_content="Test", metadata={"source": "test.txt"})
        ]
        retriever.vector_store = mock_vector_store
        
        retriever.get_relevant_documents("query")
        mock_vector_store.max_marginal_relevance_search.assert_called_once()


class TestIngestionPipeline:
    """Test document ingestion works."""
    
    @patch('azure_clients.get_embedding_client')
    @patch('multi_doc_chat.src.ingestion.load_documents')
    @patch('multi_doc_chat.src.ingestion.FAISS')
    def test_documents_are_chunked(self, mock_faiss, mock_load, mock_embeddings):
        """Test that documents are split into chunks."""
        from multi_doc_chat.src.ingestion import DocumentIngestionPipeline
        
        # Mock a document
        mock_load.return_value = [
            Document(page_content="Test content " * 100, metadata={"source": "test.txt"})
        ]
        mock_faiss.from_documents.return_value = Mock()
        
        pipeline = DocumentIngestionPipeline(
            chunk_size=100,
            chunk_overlap=10,
            vector_store_path="/tmp/test",
            session_id="test"
        )
        
        pipeline.process_documents(["test.txt"], {})
        
        # Verify FAISS was called (meaning chunks were created)
        mock_faiss.from_documents.assert_called_once()
