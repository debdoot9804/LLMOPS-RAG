from pydantic import BaseModel, Field
from typing import List, Optional,Annotated
from enum import Enum

class upload_response(BaseModel):
    session_id: str
    indexed:bool
    message: str| None

class chat_request(BaseModel):
    session_id: str
    message: str

class chat_response(BaseModel):
    session_id: str
    answer: str
    source_documents: Optional[List[dict]] = None