# Testing Guide for LLMOPS_RAG

Simple tests to verify your RAG application works correctly.

## What's Included

- **Unit Tests** (`tests/unit/`): Test individual components
  - `test_retriever.py`: Test document formatting and search
  - `test_ingestion.py`: Test document loading and validation

- **Integration Tests** (`tests/integration/`): Test how components work together
  - `test_api.py`: Test API endpoints
  - `test_chat.py`: Test retrieval pipeline

## Quick Start

### 1. Install test dependencies
```bash
pip install pytest pytest-cov httpx
```

### 2. Run all tests
```bash
pytest
```

### 3. Run with coverage
```bash
pytest --cov=multi_doc_chat
```

## Running Specific Tests

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests  
pytest tests/integration/

# Run a specific file
pytest tests/unit/test_retriever.py

# Run with verbose output
pytest -v
```

## Understanding the Tests

### Unit Tests Example
Tests a single function in isolation:
```python
def test_format_with_curly_braces(self):
    """Test that curly braces don't cause format errors."""
    # Creates a document with curly braces
    # Verifies it formats without errors
```

### Integration Tests Example
Tests multiple components together:
```python
def test_upload_single_file(self):
    """Test uploading a text file."""
    # Uploads a file to the API
    # Verifies the response is correct
```

## What Gets Tested

✅ Document formatting handles special characters  
✅ Different search types (similarity, MMR) work  
✅ API endpoints accept correct inputs  
✅ Invalid inputs are rejected properly  
✅ Empty documents raise clear errors  

## Tips

- Tests use **mocks** to avoid needing real Azure credentials
- Tests are **fast** - they don't call real APIs
- Tests are **isolated** - each test runs independently
- The import errors you see are just editor warnings - tests will run fine

## Next Steps

Want to add more tests? Follow the existing patterns:
1. Create a test class
2. Write test methods starting with `test_`
3. Use `assert` to verify behavior

That's it!
