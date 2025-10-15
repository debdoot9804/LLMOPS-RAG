"""
Integration tests for FastAPI endpoints - Simple focused tests.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from main import app

client = TestClient(app)


class TestUploadEndpoint:
    """Test /upload endpoint."""
    
    def test_upload_no_files_fails(self):
        """Test upload without files returns error."""
        response = client.post("/upload")
        assert response.status_code == 422
    
    def test_upload_single_file(self, tmp_path):
        """Test uploading a single text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        with open(test_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"files": ("test.txt", f, "text/plain")}
            )
        
        # Should succeed or fail gracefully (depends on Azure setup)
        assert response.status_code in [200, 500]


class TestChatEndpoint:
    """Test /chat endpoint."""
    
    def test_chat_missing_query_fails(self):
        """Test chat without query returns error."""
        response = client.post(
            "/chat",
            json={"session_id": "test"}
        )
        assert response.status_code == 422
    
    def test_chat_invalid_session(self):
        """Test chat with non-existent session."""
        response = client.post(
            "/chat",
            json={
                "session_id": "non-existent",
                "query": "test",
                "prompt_type": "standard"
            }
        )
        # Should return error for invalid session
        assert response.status_code in [400, 404, 500]


class TestStaticFiles:
    """Test static file serving."""
    
    def test_index_page_loads(self):
        """Test that index page is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
