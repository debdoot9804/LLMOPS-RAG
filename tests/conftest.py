"""
PyTest configuration file for LLMOPS_RAG tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Mock Azure clients BEFORE any imports
sys.modules['azure_clients'] = MagicMock()


@pytest.fixture(scope="session", autouse=True)
def mock_azure_clients():
    """Mock Azure OpenAI clients for all tests."""
    import azure_clients
    
    # Create mock clients
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Test response")
    
    mock_embeddings = Mock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 1536]
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    # Set up the mocked functions
    azure_clients.create_azure_openai_client = Mock(return_value=mock_llm)
    azure_clients.get_embedding_client = Mock(return_value=mock_embeddings)
    
    return {
        'llm': mock_llm,
        'embeddings': mock_embeddings
    }


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
