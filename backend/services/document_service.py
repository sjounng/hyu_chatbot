import uuid
from datetime import datetime
from typing import List, Optional
from ..models.document import Document, DocumentChunk
from .embedding_service import EmbeddingService

class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.documents = {}  # 임시 메모리 저장소 (실제로는 벡터 DB 사용)
        self.chunks = {}     # 임시 메모리 저장소 (실제로는 벡터 DB 사용)
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """쿼리와 유사한 청크들을 검색합니다."""
        
        if not self.chunks:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = await self.embedding_service.get_embedding(query)
        
        # 코사인 유사도 계산
        similarities = []
        for chunk in self.chunks.values():
            if chunk.embedding:
                similarity = self.embedding_service.cosine_similarity(query_embedding, chunk.embedding)
                similarities.append((similarity, chunk))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 k개 반환
        return [chunk for _, chunk in similarities[:top_k]]
    
    async def get_context_for_query(self, query: str) -> str:
        """쿼리에 대한 컨텍스트를 생성합니다."""
        
        similar_chunks = await self.search_similar_chunks(query, top_k=3)
        
        if not similar_chunks:
            return ""
        
        context_parts = []
        for chunk in similar_chunks:
            context_parts.append(f"출처: {chunk.metadata.get('document_title', '알 수 없음')}\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def get_all_documents(self) -> List[Document]:
        """모든 문서를 반환합니다."""
        return list(self.documents.values())
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """ID로 문서를 조회합니다."""
        return self.documents.get(document_id)
    
    # 개발/테스트용 메서드들
    def add_test_document(self, title: str, content: str, source: str = "test"):
        """테스트용 문서를 추가합니다."""
        document = Document(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            source=source,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.documents[document.id] = document
        return document
    
    async def add_test_chunk(self, document_id: str, content: str, metadata: dict = None):
        """테스트용 청크를 추가합니다."""
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=content,
            embedding=None,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # 임베딩 생성
        chunk.embedding = await self.embedding_service.get_embedding(chunk.content)
        
        self.chunks[chunk.id] = chunk
        return chunk 