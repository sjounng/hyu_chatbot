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
from langchain_chroma import Chroma
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

# RAG 답변 생성 함수
def get_final_response(original_query: str):
    """GitHub 저장소의 RAG 로직을 사용하여 답변을 생성합니다."""
    logger.info("\n[단계 1: AI를 이용한 프롬프트 재구성]")

    REWRITE_PROMPT = """[지시]
당신은 사용자의 질문 의도를 파악하여, 정보 검색에 더 적합한 명확하고 상세한 질문으로 재구성하는 전문가입니다.
아래 [사용자 원본 질문]을 '한양대학교' 관련 정보를 찾는다는 맥락에 맞게, 완전한 문장의 상세한 질문으로 한 문장만 생성해주세요. 다른 설명은 붙이지 마세요.

[사용자 원본 질문]
{user_query}

[재구성된 질문]"""

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
    try:
        formatted_prompt = REWRITE_PROMPT.format(user_query=original_query)
        rewritten_query = llm.invoke(formatted_prompt).content.strip()
        logger.info(f"   -> 원본 질문: '{original_query}'")
        logger.info(f"   -> AI가 재구성한 질문: '{rewritten_query}'")
    except Exception as e:
        logger.warning(f"   -> 프롬프트 재구성 실패: {e}. 원본 질문으로 계속 진행합니다.")
        rewritten_query = original_query

    # --- 재구성된 질문 기반 검색 ---
    logger.info("\n[단계 2: 확정적 Title 검색 시작]")

    def is_title_similar(query: str, title: str, threshold: float = 0.85) -> bool:
        return SequenceMatcher(None, query, title).ratio() >= threshold

    query_no_space = rewritten_query.replace(" ", "")
    golden_docs = []
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if (
            title in rewritten_query
            or title_no_space in query_no_space
            or is_title_similar(query_no_space, title_no_space)
        ):
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        logger.info(
            f"   -> '황금 티켓' 발견! Title 일치 문서: {[doc.metadata['title'] for doc in golden_docs]}"
        )

    logger.info("[단계 3: 보조 하이브리드 검색 시작]")
    hybrid_docs = hybrid_retriever.invoke(rewritten_query)
    if not hybrid_docs:
        logger.info("   -> 하이브리드 검색 결과 없음, BM25 단독 검색 시도")
        try:
            hybrid_docs = bm25_retriever.invoke(rewritten_query)
        except Exception as e:
            logger.error(f"   -> BM25 fallback 실패: {e}")
            hybrid_docs = []

    logger.info("[단계 4: 결과 종합 및 정제]")
    combined_docs_dict = OrderedDict()
    for doc in golden_docs:
        combined_docs_dict[doc.metadata["id"]] = doc
    for doc in hybrid_docs:
        if doc.metadata["id"] not in combined_docs_dict:
            combined_docs_dict[doc.metadata["id"]] = doc
    final_retrieved_docs = list(combined_docs_dict.values())[:7]

    if not final_retrieved_docs:
        return "관련 정보를 찾을 수 없습니다.", []

    logger.info(f"   -> 최종적으로 {len(final_retrieved_docs)}개의 관련 문서를 선별했습니다.")

    # 5. 최종 GPT 답변 생성
    context_str = "\n\n---\n\n".join(
        [
            f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}"
            for doc in final_retrieved_docs
        ]
    )
    source_info = [doc.metadata for doc in final_retrieved_docs]

    # Few-shot 예시 생성
    few_shot_examples = random.sample(qa_samples, 2)
    few_shot_prompt_part = "\n".join(
        [
            f"예시 질문: {ex['question']}\n예시 답변: {ex['answer']}"
            for ex in few_shot_examples
        ]
    )

    rag_prompt = f"""[지시]
당신은 여러 문서 조각을 종합하여 하나의 완성된 글로 재구성하는 '글쓰기 전문가'입니다.
주어진 [검색된 문서 내용]을 바탕으로, 아래 [질문]에 대한 답변을 매우 상세하고, 논리적이며, 잘 다듬어진 설명문 형태로 작성하세요.

[답변 예시]
{few_shot_prompt_part}
---
[검색된 문서 내용]
{context_str}
---
[질문]
{rewritten_query}
[답변]"""

    logger.info("\n[단계 5: 최종 답변 생성 시작]")
    try:
        answer = llm.invoke(rag_prompt).content.strip()
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
    