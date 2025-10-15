import os
from typing import List, Dict, Any
import uuid
import tempfile
import shutil
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from multi_doc_chat.src.ingestion import DocumentIngestionPipeline
from multi_doc_chat.src.retriever import DocumentRetriever
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exception.exception import CustomException
from multi_doc_chat.logger.logger import get_logger

from multi_doc_chat.model.models import upload_response, chat_request, chat_response

# Initialize logger
logger = get_logger(__file__)


app=FastAPI(title="Multidoc Chat API", description="API for multi-document chat application", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files BEFORE templates
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Store active ingestion pipelines and retrievers per session
sessions: Dict[str, List[Dict[str, Any]]] = {}
ingestion_pipelines: Dict[str, DocumentIngestionPipeline] = {}
retrievers: Dict[str, DocumentRetriever] = {}

# Base paths for data storage
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_STORE_BASE = os.path.join(DATA_DIR, "vector_store")
TEMP_UPLOAD_DIR = os.path.join(DATA_DIR, "temp_uploads")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_BASE, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health()-> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request)-> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=upload_response)
async def upload_file(files: List[UploadFile] = File(...)) -> upload_response:
    """Upload and index up to 2 document files."""
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Maximum 2 files allowed per upload")
    session_id = str(uuid.uuid4())[:12]
    temp_file_paths = []
    try:
        # Save uploaded files
        for file in files:
            temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{session_id}_{file.filename}")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_file_paths.append(temp_file_path)
        # Create pipeline and process documents
        pipeline = DocumentIngestionPipeline(
            chunk_size=1000,
            chunk_overlap=200,
            vector_store_path=VECTOR_STORE_BASE,
            session_id=session_id
        )
        pipeline.process_documents(
            file_paths=temp_file_paths,
            metadata={"filenames": [file.filename for file in files]}
        )
        # Store session
        ingestion_pipelines[session_id] = pipeline
        sessions[session_id] = []
        logger.info("Indexed %s for session %s", ', '.join([file.filename for file in files]), session_id)
        return upload_response(
            session_id=session_id,
            indexed=True,
            message="Successfully indexed: " + ', '.join([file.filename for file in files])
        )
    except Exception as e:
        logger.error("Upload error: %s", str(e))
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=chat_response)
async def chat(request: chat_request) -> chat_response:
    """Chat with documents using RAG."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get or create retriever
        if session_id not in retrievers:
            retrievers[session_id] = DocumentRetriever(
                vector_store_path=VECTOR_STORE_BASE,
                session_id=session_id,
                top_k=4,
                search_type="mmr",
                score_threshold=0.3
            )
        
        # Format chat history
        chat_history = sessions.get(session_id, [])
        formatted_history = [
            (msg["content"], chat_history[i+1]["content"] if i+1 < len(chat_history) else "")
            for i, msg in enumerate(chat_history) if msg["role"] == "user"
        ]
        
        # Query documents
        answer, source_docs = retrievers[session_id].query_documents(
            query=request.message,
            prompt_type="standard",
            chat_history=formatted_history
        )
        
        # Update history
        sessions[session_id].extend([
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": answer}
        ])
        
        # Format sources
        source_documents = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in source_docs
        ]
        
        return chat_response(
            session_id=session_id,
            answer=answer,
            source_documents=source_documents
        )
        
    except Exception as e:
        logger.error("Chat error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


