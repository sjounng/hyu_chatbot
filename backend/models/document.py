from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Document(BaseModel):
    id: str
    title: str
    content: str
    url: Optional[str] = None
    source: str
    created_at: datetime
    updated_at: datetime

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: dict
    created_at: datetime 