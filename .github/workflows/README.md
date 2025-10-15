# GitHub Actions CI/CD

This workflow automatically tests your RAG application on every push and pull request.

## What Gets Tested

✅ Python 3.10, 3.11, and 3.12 compatibility  
✅ Unit tests (fast component tests)  
✅ Integration tests (end-to-end tests)  
✅ Code coverage reporting  
✅ Code formatting and linting  

## System Dependencies

The workflow installs:
- **tesseract-ocr**: For OCR text extraction from images
- **poppler-utils**: For PDF to image conversion

## Viewing Results

1. Go to your GitHub repository
2. Click the "Actions" tab
3. See test results for each commit/PR

## Badges

Add to your README.md:

```markdown
![CI](https://github.com/debdoot9804/LLMOPS-RAG/workflows/CI-LLMOPS-RAG/badge.svg)
```

## Local Testing

Before pushing, test locally:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=multi_doc_chat

# Run only unit tests
pytest tests/unit/
```
