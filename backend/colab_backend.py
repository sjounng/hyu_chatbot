"""
Google Colab에서 실행할 수 있는 한양대학교 AI 챗봇 백엔드
"""

import os
import sys
import subprocess
import requests
import json
from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any

# Google Colab 환경 확인 및 조건부 import
try:
    from google.colab import output
    from IPython.display import display, HTML
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    # 로컬 환경용 대체 함수
    def display(html):
        print(html)
    def HTML(content):
        return content

# 필요한 패키지 설치
def install_requirements():
    """필요한 패키지들을 설치합니다."""
    packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "sentence-transformers==2.2.2",
        "numpy==1.24.3",
        "requests==2.31.0",
        "transformers==4.35.0",
        "torch==2.1.0",
        "accelerate==0.24.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")
        except:
            print(f"❌ {package} 설치 실패")

# 데이터 모델
class ChatRequest:
    def __init__(self, message: str, conversation_id: Optional[str] = None):
        self.message = message
        self.conversation_id = conversation_id

class ChatResponse:
    def __init__(self, response: str, conversation_id: str, timestamp: datetime):
        self.response = response
        self.conversation_id = conversation_id
        self.timestamp = timestamp

class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: datetime, conversation_id: str):
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.conversation_id = conversation_id

# 임베딩 서비스
class EmbeddingService:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 임베딩 모델 로드 완료")
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 실패: {e}")
            self.model = None
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩을 생성합니다."""
        if not self.model:
            return [0.0] * 384
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"임베딩 생성 오류: {str(e)}")
            return [0.0] * 384

# LLM 서비스 (Hugging Face 모델 사용)
class LLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Hugging Face 모델을 로드합니다."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # 한국어에 최적화된 모델 (더 작은 모델로 변경)
            model_name = "beomi/KoAlpaca-Polyglot-5.8B"
            
            print("🔄 모델 로딩 중... (시간이 걸릴 수 있습니다)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print("✅ LLM 모델 로드 완료")
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("대신 간단한 응답 생성기를 사용합니다.")
    
    def generate_response(self, prompt: str, max_length: int = 500) -> str:
        """텍스트 응답을 생성합니다."""
        if not self.model or not self.tokenizer:
            return self._simple_response(prompt)
        
        try:
            import torch
            
            # 프롬프트 구성
            full_prompt = f"""### 질문: {prompt}

### 답변:"""
            
            # 토큰화
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거
            response = response.replace(full_prompt, "").strip()
            
            return response if response else self._simple_response(prompt)
            
        except Exception as e:
            print(f"생성 오류: {e}")
            return self._simple_response(prompt)
    
    def _simple_response(self, prompt: str) -> str:
        """간단한 응답 생성기 (모델 로드 실패 시 사용)"""
        responses = [
            "한양대학교에 대해 질문해주셨네요. 더 구체적인 정보를 제공해드릴 수 있도록 도움이 필요합니다.",
            "한양대학교 관련 정보를 찾고 계시는군요. 어떤 부분에 대해 궁금하신가요?",
            "한양대학교에 대한 질문을 받았습니다. 학과, 캠퍼스, 입학 정보 등 구체적으로 말씀해주세요.",
            "한양대학교 정보를 제공해드리겠습니다. 어떤 정보가 필요하신가요?"
        ]
        
        import random
        return random.choice(responses)

# 문서 서비스
class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.documents = {}
        self.chunks = {}
        self._load_sample_data()
    
    def _load_sample_data(self):
        """샘플 한양대학교 데이터를 로드합니다."""
        sample_docs = [
            {
                "title": "한양대학교 개요",
                "content": "한양대학교는 1939년에 설립된 대한민국의 사립대학교입니다. 서울특별시 성동구에 위치한 서울캠퍼스와 경기도 안산시에 위치한 ERICA캠퍼스로 구성되어 있습니다. 공학, 의학, 경영학 등 다양한 분야에서 우수한 성과를 보이고 있으며, 특히 공학 분야에서 국내 최고 수준의 교육을 제공합니다."
            },
            {
                "title": "한양대학교 학과",
                "content": "한양대학교는 공과대학, 의과대학, 경영대학, 사범대학, 문과대학, 예술체육대학, 국제문화대학 등 다양한 단과대학을 운영하고 있습니다. 주요 학과로는 건축학과, 기계공학과, 전자공학과, 컴퓨터공학과, 의학과, 경영학과, 국어국문학과 등이 있습니다."
            },
            {
                "title": "한양대학교 캠퍼스",
                "content": "한양대학교는 서울캠퍼스와 ERICA캠퍼스 두 개의 캠퍼스를 운영합니다. 서울캠퍼스는 서울특별시 성동구 왕십리로에 위치하며, ERICA캠퍼스는 경기도 안산시 상록구 한양대학로에 위치합니다. 각 캠퍼스는 독립적인 교육 환경을 제공하며, 학생들은 자신의 전공에 따라 적절한 캠퍼스를 선택할 수 있습니다."
            }
        ]
        
        for i, doc in enumerate(sample_docs):
            doc_id = str(uuid.uuid4())
            self.documents[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "created_at": datetime.now()
            }
            
            # 청크 생성
            chunks = self._split_content(doc["content"])
            for j, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                self.chunks[chunk_id] = {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "content": chunk_text,
                    "embedding": self.embedding_service.get_embedding(chunk_text),
                    "metadata": {"title": doc["title"], "chunk_index": j}
                }
    
    def _split_content(self, content: str) -> List[str]:
        """내용을 청크로 분할합니다."""
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < 200:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """쿼리와 유사한 청크들을 검색합니다."""
        if not self.chunks:
            return []
        
        query_embedding = self.embedding_service.get_embedding(query)
        
        similarities = []
        for chunk in self.chunks.values():
            if chunk["embedding"]:
                similarity = self._cosine_similarity(query_embedding, chunk["embedding"])
                similarities.append((similarity, chunk))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도를 계산합니다."""
        import numpy as np
        
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"유사도 계산 오류: {str(e)}")
            return 0.0
    
    def get_context_for_query(self, query: str) -> str:
        """쿼리에 대한 컨텍스트를 생성합니다."""
        similar_chunks = self.search_similar_chunks(query, top_k=2)
        
        if not similar_chunks:
            return ""
        
        context_parts = []
        for chunk in similar_chunks:
            context_parts.append(f"출처: {chunk['metadata']['title']}\n{chunk['content']}")
        
        return "\n\n".join(context_parts)

# 채팅 서비스
class ChatService:
    def __init__(self):
        self.llm_service = LLMService()
        self.document_service = DocumentService()
        self.conversations = {}
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """사용자 메시지를 처리하고 AI 응답을 생성합니다."""
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        conversation = self.conversations.get(conversation_id, [])
        
        # 사용자 메시지 추가
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
        conversation.append(user_message)
        
        # RAG 응답 생성
        ai_response = self._generate_rag_response(request.message)
        
        # AI 메시지 추가
        assistant_message = ChatMessage(
            role="assistant",
            content=ai_response,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
        conversation.append(assistant_message)
        
        # 대화 기록 저장
        self.conversations[conversation_id] = conversation
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )
    
    def _generate_rag_response(self, user_message: str) -> str:
        """RAG를 사용하여 AI 응답을 생성합니다."""
        
        # 컨텍스트 검색
        context = self.document_service.get_context_for_query(user_message)
        
        # 프롬프트 구성
        if context:
            prompt = f"""한양대학교에 대한 질문에 답변해주세요.

참고 정보:
{context}

질문: {user_message}

답변:"""
        else:
            prompt = f"""한양대학교에 대한 질문에 답변해주세요.

질문: {user_message}

답변:"""
        
        # LLM 응답 생성
        response = self.llm_service.generate_response(prompt)
        
        return response

# FastAPI 앱
def create_app():
    """FastAPI 앱을 생성합니다."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    
    app = FastAPI(
        title="한양대학교 AI 챗봇 (Colab)",
        description="Google Colab에서 실행되는 한양대학교 정보 제공 AI 챗봇",
        version="1.0.0"
    )
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic 모델
    class ChatRequestModel(BaseModel):
        message: str
        conversation_id: Optional[str] = None
    
    class ChatResponseModel(BaseModel):
        response: str
        conversation_id: str
        timestamp: datetime
    
    # 서비스 인스턴스
    chat_service = ChatService()
    
    @app.get("/")
    async def root():
        return {
            "message": "한양대학교 AI 챗봇 (Colab)에 오신 것을 환영합니다!",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.post("/api/v1/chat/", response_model=ChatResponseModel)
    async def chat(request: ChatRequestModel):
        """채팅 메시지를 처리합니다."""
        try:
            chat_request = ChatRequest(
                message=request.message,
                conversation_id=request.conversation_id
            )
            response = chat_service.process_message(chat_request)
            
            return ChatResponseModel(
                response=response.response,
                conversation_id=response.conversation_id,
                timestamp=response.timestamp
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "서버가 정상적으로 작동 중입니다."}
    
    return app

# Colab 실행 함수
def run_colab_server():
    """Colab에서 서버를 실행합니다."""
    print("🚀 한양대학교 AI 챗봇 서버를 시작합니다...")
    
    # 패키지 설치
    print("📦 필요한 패키지를 설치합니다...")
    install_requirements()
    
    # 앱 생성
    app = create_app()
    
    # ngrok 설치 및 실행 (외부 접근용)
    try:
        print("🌐 ngrok을 설치하고 실행합니다...")
        subprocess.run(["pip", "install", "pyngrok"], check=True)
        
        from pyngrok import ngrok
        
        # FastAPI 서버 시작
        import uvicorn
        import threading
        
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8000)
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # 잠시 대기
        import time
        time.sleep(3)
        
        # ngrok 터널 생성
        public_url = ngrok.connect(8000)
        print(f"✅ 서버가 성공적으로 시작되었습니다!")
        print(f"🌍 공개 URL: {public_url}")
        print(f"📚 API 문서: {public_url}/docs")
        
        # URL을 HTML로 표시
        display(HTML(f"""
        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>🎉 서버 실행 완료!</h3>
            <p><strong>공개 URL:</strong> <a href="{public_url}" target="_blank">{public_url}</a></p>
            <p><strong>API 문서:</strong> <a href="{public_url}/docs" target="_blank">{public_url}/docs</a></p>
            <p><strong>헬스 체크:</strong> <a href="{public_url}/health" target="_blank">{public_url}/health</a></p>
        </div>
        """))
        
        return public_url
        
    except Exception as e:
        print(f"❌ ngrok 실행 실패: {e}")
        print("로컬에서만 실행됩니다.")
        return None

# 테스트 함수
def test_chat():
    """채팅 기능을 테스트합니다."""
    print("🧪 채팅 기능을 테스트합니다...")
    
    chat_service = ChatService()
    
    test_messages = [
        "한양대학교는 어떤 대학교인가요?",
        "한양대학교 학과는 어떤 것들이 있나요?",
        "한양대학교 캠퍼스는 어디에 있나요?"
    ]
    
    for message in test_messages:
        print(f"\n👤 사용자: {message}")
        
        request = ChatRequest(message=message)
        response = chat_service.process_message(request)
        
        print(f"🤖 AI: {response.response}")
        print("-" * 50)

if __name__ == "__main__":
    # Colab에서 실행할 때
    public_url = run_colab_server()
    
    if public_url:
        print(f"\n🎯 프론트엔드에서 다음 URL로 연결하세요: {public_url}")
    else:
        print("\n로컬에서만 실행 중입니다.") 