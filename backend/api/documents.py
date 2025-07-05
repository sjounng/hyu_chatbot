from fastapi import APIRouter, HTTPException
from typing import List
from ..models.document import Document, DocumentChunk
from ..services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

# 서비스 인스턴스 생성
document_service = DocumentService()

@router.get("/search/{query}", response_model=List[DocumentChunk])
async def search_documents(query: str, top_k: int = 5):
    """문서에서 쿼리와 유사한 청크들을 검색합니다."""
    try:
        chunks = await document_service.search_similar_chunks(query, top_k)
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류가 발생했습니다: {str(e)}")

@router.get("/context/{query}")
async def get_context(query: str):
    """쿼리에 대한 컨텍스트를 반환합니다."""
    try:
        context = await document_service.get_context_for_query(query)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"컨텍스트 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/stats")
async def get_document_stats():
    """문서 통계 정보를 반환합니다."""
    try:
        documents = document_service.get_all_documents()
        chunks = list(document_service.chunks.values())
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "created_at": doc.created_at
                }
                for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}") 