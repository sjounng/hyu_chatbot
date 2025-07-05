import os
import json
import pickle
import random
import logging
from collections import OrderedDict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from difflib import SequenceMatcher
from contextlib import asynccontextmanager

# LangChain imports (최신 버전으로 업데이트)
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 환경 변수 로드
load_dotenv()

# 로깅 설정
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# API 키 및 설정
api_key = os.getenv("OPENAI_API_KEY")

# --- 설정 및 문서 로딩 ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db_hyu")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
DOC_FILE_PATH = os.path.join(DATA_DIR, "document.json")
QA_FILE_PATH = os.path.join(DATA_DIR, "question_sample.json")

# OpenAI API 키 검증
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 전역 변수로 검색 시스템 초기화
embedding_model = None
vector_retriever = None
bm25_retriever = None
hybrid_retriever = None
title_to_doc_map = {}
all_titles = []
qa_samples = []

def initialize_search_system():
    """검색 시스템을 초기화합니다."""
    global embedding_model, vector_retriever, bm25_retriever, hybrid_retriever, title_to_doc_map, all_titles, qa_samples
    
    logger.info("1. 검색 시스템 및 전체 문서 데이터를 로드합니다...")
    
    try:
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        vector_retriever = Chroma(
            persist_directory=VECTOR_DB_PATH, 
            embedding_function=embedding_model
        ).as_retriever(search_kwargs={"k": 10})
        
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], 
            weights=[0.5, 0.5]
        )
        
        with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
            all_docs_data = json.load(f)
        
        title_to_doc_map = {
            item["title"]: Document(page_content=item["content"], metadata=item)
            for item in all_docs_data
        }
        all_titles = list(title_to_doc_map.keys())
        
        with open(QA_FILE_PATH, "r", encoding="utf-8") as f:
            qa_samples = json.load(f)
        
        logger.info("   -> 로드 완료!")
        return True
        
    except Exception as e:
        logger.error(f"   -> 로드 실패: {e}")
        return False

# Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    sources: List[Dict[str, Any]] = []

# RAG 답변 생성 함수 (temp.py의 개선된 로직 사용)
def get_final_response(query: str):
    """사용자 질문을 받아, 다중 검색, RRF 융합, 최종 답변 생성을 수행합니다."""
    logger.info("\n[단계 1: 초고속 동시 검색 (No-API)]")
    
    # 검색 A: 확정적 Title 검색 ('골든 티켓')
    golden_docs = []
    query_no_space = query.replace(" ", "")
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if title in query or title_no_space in query_no_space:
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        logger.info(f"   -> '골든 티켓' 발견: {[doc.metadata['title'] for doc in golden_docs]}")

    # 검색 B: BM25 키워드 검색
    bm25_docs = bm25_retriever.invoke(query)
    logger.info(f"   -> BM25 검색으로 {len(bm25_docs)}개의 후보를 찾았습니다.")

    # 검색 C: 벡터 의미 기반 검색
    # 이 과정에서 query 임베딩을 위해 API가 1회 호출됩니다. (매우 빠름)
    vector_docs = vector_retriever.invoke(query)
    logger.info(f"   -> 벡터 검색으로 {len(vector_docs)}개의 후보를 찾았습니다.")

    # [단계 2: Reciprocal Rank Fusion (RRF)으로 순위 융합]
    logger.info("\n[단계 2: RRF를 이용한 순위 융합 (No-API)]")
    rrf_scores = {}
    
    # 각 검색 결과를 순회하며 RRF 점수 계산
    # '골든 티켓'은 가장 높은 우선순위를 가짐
    all_search_results = [golden_docs, bm25_docs, vector_docs]
    
    for results in all_search_results:
        for i, doc in enumerate(results):
            doc_id = doc.metadata['id']
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0, 'doc': doc}
            # k=60은 RRF에서 일반적으로 사용하는 하이퍼파라미터
            rrf_scores[doc_id]['score'] += 1.0 / (i + 60)

    # 점수가 높은 순으로 정렬
    sorted_docs_with_scores = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)
    final_retrieved_docs = [item['doc'] for item in sorted_docs_with_scores][:5]

    if not final_retrieved_docs:
        return "관련 정보를 찾을 수 없습니다.", []
        
    logger.info("\n[최종 선별된 문서 (LLM 전달용)]")
    for i, doc in enumerate(final_retrieved_docs):
        logger.info(f"  {i+1}. [출처: {doc.metadata.get('title')}]")
    logger.info("-" * 20)
    
    # 3. 최종 답변 생성
    context_str = "\n\n---\n\n".join([f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}" for doc in final_retrieved_docs])
    source_info = [doc.metadata for doc in final_retrieved_docs]
    
    # Few-shot 예시 생성
    few_shot_examples = random.sample(qa_samples, 2)
    few_shot_prompt_part = "\n\n".join(
        [f"예시 질문: {ex['question']}\n예시 답변: {ex['answer']}" for ex in few_shot_examples]
    )
    
    rag_prompt = f"""[지시]
당신은 여러 문서 조각을 종합하여 하나의 완성된 글로 재구성하는 '글쓰기 전문가'입니다.
주어진 [검색된 문서 내용]을 바탕으로, 아래 [질문]에 대한 답변을 매우 상세하고, 논리적이며, 잘 다듬어진 설명문 형태로 작성하세요.
답변의 스타일과 형식은 아래 [답변 예시]를 참고하되, 내용은 반드시 [검색된 문서 내용]에만 근거해야 합니다.

[답변 예시]
{few_shot_prompt_part}
---
[검색된 문서 내용]
{context_str}
---
[질문]
{query}
[답변]"""
    
    logger.info(f"\n[단계 3: GPT-4o 모델로 최종 답변 생성 (유일한 API 호출)]")
    try:
        llm_final = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=api_key)
        answer = llm_final.invoke(rag_prompt).content.strip()
        return answer, source_info
    except Exception as e:
        return f"답변 생성 중 오류 발생: {e}", []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 이벤트 핸들러"""
    # 시작 시
    logger.info("한양대학교 AI 챗봇 서버를 시작합니다...")
    success = initialize_search_system()
    if not success:
        logger.error("❌ 검색 시스템 초기화 실패")
    
    yield
    
    # 종료 시 (필요한 경우 정리 작업)
    logger.info("🔄 서버를 종료합니다...")

# FastAPI 앱 생성
app = FastAPI(
    title="한양대학교 AI 챗봇 (GitHub 기반)",
    description="GitHub 저장소 기반의 한양대학교 정보 제공 AI 챗봇",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 대화 기록 저장
conversations = {}

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "한양대학교 AI 챗봇 (GitHub 기반)에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지를 처리합니다."""
    try:
        logger.info(f"새로운 채팅 요청: {request.message[:50]}...")
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # RAG 응답 생성
        answer, sources = get_final_response(request.message)
        
        logger.info(f"응답 생성 완료: {len(answer)} 문자, {len(sources)} 개 소스")
        
        return ChatResponse(
            response=answer,
            conversation_id=conversation_id,
            timestamp=datetime.now(),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {str(e)}", exc_info=True)
        
        # 요청이 취소된 경우
        if "cancelled" in str(e).lower() or "abort" in str(e).lower():
            raise HTTPException(status_code=499, detail="Client Closed Request")
        
        # OpenAI API 관련 오류
        if "openai" in str(e).lower() or "api" in str(e).lower():
            raise HTTPException(status_code=503, detail="AI 서비스 일시 불가. 잠시 후 다시 시도해주세요.")
        
        # 일반적인 서버 오류
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy", 
        "message": "서버가 정상적으로 작동 중입니다.",
        "search_system_loaded": hybrid_retriever is not None
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    