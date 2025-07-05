import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self):
        # 한국어에 최적화된 모델 사용
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    async def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩을 생성합니다."""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"임베딩 생성 오류: {str(e)}")
            # 기본 임베딩 반환 (384차원)
            return [0.0] * 384
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """두 임베딩 간의 코사인 유사도를 계산합니다."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 정규화
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 코사인 유사도 계산
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"유사도 계산 오류: {str(e)}")
            return 0.0
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩을 배치로 생성합니다."""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"배치 임베딩 생성 오류: {str(e)}")
            # 기본 임베딩들 반환
            return [[0.0] * 384 for _ in texts] 