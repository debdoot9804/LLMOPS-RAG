"""
Unit tests for DocumentRetriever - Simple focused tests.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document


@patch('azure_clients.create_azure_openai_client', return_value=Mock())
@patch('azure_clients.get_embedding_client', return_value=Mock())
class TestDocumentFormatting:
    """Test document formatting logic."""
    
    def test_format_with_curly_braces(self, mock_embeddings, mock_llm):
        """Test that curly braces don't cause format errors."""
        with patch('multi_doc_chat.src.retriever.FAISS'):
            from multi_doc_chat.src.retriever import DocumentRetriever
            
            retriever = DocumentRetriever("/tmp/test", "test-session")
            
            docs = [Document(
                page_content="Code: {key: value}",
                metadata={"source": "test.txt", "page": 1, "similarity_score": 0.9}
            )]
            
            # Should not raise error
            result = retriever._format_documents(docs)
            assert "Document 1" in result
    
    def test_format_with_integer_page(self, mock_embeddings, mock_llm):
        """Test handling integer page numbers."""
        with patch('multi_doc_chat.src.retriever.FAISS'):
            from multi_doc_chat.src.retriever import DocumentRetriever
            
            retriever = DocumentRetriever("/tmp/test", "test-session")
            
            docs = [Document(
                page_content="Content",
                metadata={"source": "test.pdf", "page": 5, "similarity_score": 0.9}
            )]
            
            result = retriever._format_documents(docs)
            assert "5" in result


@patch('azure_clients.create_azure_openai_client', return_value=Mock())
@patch('azure_clients.get_embedding_client', return_value=Mock())
class TestSearchTypes:
    """Test different search strategies."""
    
    def test_similarity_search(self, mock_embeddings, mock_llm):
        """Test similarity search adds scores."""
        with patch('multi_doc_chat.src.retriever.FAISS'):
            from multi_doc_chat.src.retriever import DocumentRetriever
            
            retriever = DocumentRetriever("/tmp/test", "test-session", search_type="similarity")
            
            mock_vector_store = Mock()
            mock_vector_store.similarity_search_with_score.return_value = [
                (Document(page_content="Test", metadata={"source": "test.pdf"}), 0.95)
            ]
            retriever.vector_store = mock_vector_store
            
            docs = retriever.get_relevant_documents("query")
            assert docs[0].metadata["similarity_score"] == 0.95
    
    def test_mmr_search(self, mock_embeddings, mock_llm):
        """Test MMR search is called correctly."""
        with patch('multi_doc_chat.src.retriever.FAISS'):
            from multi_doc_chat.src.retriever import DocumentRetriever
            
            retriever = DocumentRetriever("/tmp/test", "test-session", search_type="mmr")
            
            mock_vector_store = Mock()
            mock_vector_store.max_marginal_relevance_search.return_value = [
                Document(page_content="Test", metadata={"source": "test.pdf"})
            ]
            retriever.vector_store = mock_vector_store
            
            docs = retriever.get_relevant_documents("query")
            mock_vector_store.max_marginal_relevance_search.assert_called_once()
