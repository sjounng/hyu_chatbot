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

# CrossEncoder를 위한 import
ENABLE_CROSSENCODER = os.getenv("ENABLE_CROSSENCODER", "true").lower() == "true"

try:
    if ENABLE_CROSSENCODER:
        from sentence_transformers.cross_encoder import CrossEncoder
        CROSSENCODER_AVAILABLE = True
    else:
        CROSSENCODER_AVAILABLE = False
        logging.info("CrossEncoder가 환경 변수로 비활성화되었습니다.")
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logging.warning("sentence-transformers 패키지가 없습니다. 간단한 키워드 매칭을 사용합니다.")

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

# --- 설정 및 문서 로딩 (final.py와 동일한 경로 구조) ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db_hyu")
DOC_FILE_PATH = os.path.join(DATA_DIR, "document.json")
QA_FILE_PATH = os.path.join(DATA_DIR, "question_sample.json")
MATCHED_JSON_PATH = os.path.join(DATA_DIR, "matched.json")

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
abbrev_map = {}
cross_encoder_model = None

def initialize_search_system():
    """검색 시스템을 초기화합니다."""
    global embedding_model, vector_retriever, bm25_retriever, hybrid_retriever, title_to_doc_map, all_titles, qa_samples, abbrev_map, cross_encoder_model
    
    logger.info("1. 검색 시스템 및 전체 문서 데이터를 로드합니다...")
    
    # 파일 존재 여부 확인
    logger.info("파일 존재 여부 확인:")
    for file_path, name in [
        (DOC_FILE_PATH, "문서 파일"),
        (QA_FILE_PATH, "질문 샘플"),
        (MATCHED_JSON_PATH, "축약어 매핑"),
        (VECTOR_DB_PATH, "벡터 DB")
    ]:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                logger.info(f"   ✓ {name}: 존재 ({size} bytes)")
            else:
                logger.info(f"   ✓ {name}: 존재 (디렉토리)")
        else:
            logger.info(f"   ✗ {name}: 없음 - {file_path}")
    
    try:
        # 문서 데이터 로딩 (BM25와 벡터 DB 모두에 필요)
        with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
            all_docs_data = json.load(f)
        
        title_to_doc_map = {
            item["title"]: Document(page_content=item["content"], metadata=item)
            for item in all_docs_data
        }
        all_titles = list(title_to_doc_map.keys())
        
        # 문서들을 Document 리스트로 변환 (BM25용)
        documents = [Document(page_content=item["content"], metadata=item) for item in all_docs_data]
        
        # 벡터 검색 초기화 시도 (실패하면 런타임 생성)
        vector_retriever = None
        try:
            embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
            
            # 기존 벡터 DB가 있으면 사용
            if os.path.exists(VECTOR_DB_PATH):
                vector_retriever = Chroma(
                    persist_directory=VECTOR_DB_PATH, 
                    embedding_function=embedding_model
                ).as_retriever(search_kwargs={"k": 20})
                logger.info("   -> 기존 벡터 DB 로드 완료!")
            else:
                # 없으면 런타임에 새로 생성
                logger.info("   -> 기존 벡터 DB가 없습니다. 런타임에 생성 중...")
                
                # documents로부터 벡터 DB 생성
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    persist_directory=VECTOR_DB_PATH
                )
                vectorstore.persist()  # 디스크에 저장
                
                vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
                logger.info("   -> 새 벡터 DB 생성 및 저장 완료!")
                
        except Exception as e:
            logger.warning(f"   -> 벡터 검색 시스템 로드/생성 실패: {e}")
            logger.warning("   -> 벡터 검색 없이 진행합니다.")
        
        # BM25 인덱스 런타임 생성
        bm25_retriever = None
        try:
            logger.info("   -> BM25 인덱스를 런타임에 생성 중...")
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 20
            logger.info("   -> BM25 인덱스 생성 완료!")
        except Exception as e:
            logger.warning(f"   -> BM25 인덱스 생성 실패: {e}")
            logger.warning("   -> BM25 검색 없이 진행합니다.")
        
        # Hybrid retriever 설정 (사용 가능한 것들로만)
        available_retrievers = []
        if bm25_retriever:
            available_retrievers.append(bm25_retriever)
        if vector_retriever:
            available_retrievers.append(vector_retriever)
            
        if len(available_retrievers) >= 2:
            hybrid_retriever = EnsembleRetriever(
                retrievers=available_retrievers, 
                weights=[0.5, 0.5]
            )
        elif len(available_retrievers) == 1:
            hybrid_retriever = available_retrievers[0]
        else:
            hybrid_retriever = None
            logger.warning("   -> 검색 시스템이 모두 실패했습니다. 골든 티켓만 사용합니다.")

        with open(QA_FILE_PATH, "r", encoding="utf-8") as f:
            qa_samples = json.load(f)
        
        # 축약어 매핑 로딩
        with open(MATCHED_JSON_PATH, "r", encoding="utf-8") as f:
            matched_list = json.load(f)
        abbrev_map = {item["key"]: item["value"] for item in matched_list}
        
        # CrossEncoder 로드 시도 (final.py와 동일)
        if CROSSENCODER_AVAILABLE:
            try:
                # final.py와 동일한 모델 사용
                model_name = "bongsoo/kpf-cross-encoder-v1"
                logger.info(f"CrossEncoder 모델 로드 중: {model_name}")
                cross_encoder_model = CrossEncoder(model_name)
                logger.info("CrossEncoder 모델 로드 완료!")
            except Exception as e:
                logger.warning(f"CrossEncoder 모델 로드 실패: {e}. 키워드 매칭을 사용합니다.")
                cross_encoder_model = None
        
        logger.info("   -> 로드 완료!")
        return True
        
    except Exception as e:
        logger.error(f"   -> 로드 실패: {e}")
        return False

# 축약어 확장 함수
def expand_abbreviations(text: str) -> str:
    """축약어를 확장합니다."""
    for key, value in abbrev_map.items():
        if key in text:
            text = text.replace(key, value)
    return text

# CrossEncoder 구현 (final.py와 동일)
def rerank_documents_crossencoder(query: str, documents: List[Document]) -> List[Document]:
    """CrossEncoder를 사용해 문서들을 재정렬합니다. (final.py와 동일한 구현)"""
    if not documents or not cross_encoder_model:
        return rerank_documents_simple(query, documents)
    
    logger.info(f"   -> CrossEncoder로 {len(documents)}개 문서 재순위화 진행...")
    
    try:
        # final.py와 완전히 동일한 구현
        pairs = [[query, doc.page_content] for doc in documents]
        scores = cross_encoder_model.predict(pairs, show_progress_bar=False)
        
        # 점수와 문서 매핑
        doc_with_scores = list(zip(scores, documents))
        doc_with_scores.sort(key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in doc_with_scores]
        
        logger.info("   -> CrossEncoder 재순위화 완료!")
        return reranked_docs
        
    except Exception as e:
        logger.warning(f"CrossEncoder 실행 중 오류: {e}. 키워드 매칭으로 대체합니다.")
        return rerank_documents_simple(query, documents)

# 간단한 문서 재정렬 함수 (fallback)
def rerank_documents_simple(query: str, documents: List[Document]) -> List[Document]:
    """간단한 키워드 매칭으로 문서들을 재정렬합니다."""
    if not documents:
        return []
    
    logger.info(f"   -> 간단한 키워드 매칭으로 {len(documents)}개 문서 재순위화 진행...")
    
    # 쿼리를 소문자로 변환하고 단어로 분리
    query_words = set(query.lower().split())
    
    doc_scores = []
    for doc in documents:
        # 문서 내용을 소문자로 변환
        content_lower = doc.page_content.lower()
        title_lower = doc.metadata.get('title', '').lower()
        
        # 키워드 매칭 점수 계산
        content_matches = sum(1 for word in query_words if word in content_lower)
        title_matches = sum(2 for word in query_words if word in title_lower)  # 제목 매칭에 더 높은 가중치
        
        total_score = content_matches + title_matches
        doc_scores.append((total_score, doc))
    
    # 점수 순으로 정렬
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for score, doc in doc_scores]
    
    logger.info("   -> 재순위화 완료!")
    return reranked_docs

# Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: datetime
    sources: List[Dict[str, Any]] = []

# RAG 답변 생성 함수 (CrossEncoder 포함)
def get_final_response(query: str):
    """CrossEncoder를 포함한 RAG 로직을 사용합니다."""
    
    logger.info(f"\n[전처리] 입력된 질문: {query}")
    query = expand_abbreviations(query)
    logger.info(f"[전처리] 확장된 질문: {query}")

    logger.info("\n[단계 1: 초고속 동시 검색 (원본 쿼리 사용)]")
    golden_docs = []
    query_no_space = query.replace(" ", "")
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if title in query or title_no_space in query_no_space:
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        logger.info(
            f"   -> '골든 티켓' 발견: {[doc.metadata['title'] for doc in golden_docs]}"
        )

    # BM25 검색 (사용 가능한 경우에만)
    bm25_docs = []
    if bm25_retriever:
        try:
            bm25_docs = bm25_retriever.invoke(query)
            logger.info(f"   -> BM25 검색으로 {len(bm25_docs)}개의 후보를 찾았습니다.")
        except Exception as e:
            logger.warning(f"   -> BM25 검색 실패: {e}")
    else:
        logger.info("   -> BM25 검색 건너뜀 (인덱스 없음)")

    # 벡터 검색 (사용 가능한 경우에만)
    vector_docs = []
    if vector_retriever:
        try:
            vector_docs = vector_retriever.invoke(query)
            logger.info(f"   -> 벡터 검색으로 {len(vector_docs)}개의 후보를 찾았습니다.")
        except Exception as e:
            logger.warning(f"   -> 벡터 검색 실패: {e}")
    else:
        logger.info("   -> 벡터 검색 건너뜀 (시스템 없음)")

    # 벡터 디버깅 로그
    logger.info("벡터 디버깅용")
    if vector_docs:
        for i, doc in enumerate(vector_docs):
            logger.info(
                f"  {i+1}. [출처: {doc.metadata.get('title')}] {doc.page_content[:150]}..."
            )
    else:
        logger.info("  -> vector_docs로 검색된 문서가 없습니다.")
    logger.info("-" * 20)

    logger.info("\n[단계 2: RRF 융합 및 CrossEncoder 재순위화]")
    rrf_scores = {}
    all_search_results = [golden_docs, bm25_docs, vector_docs]
    
    # 빈 결과들 필터링
    all_search_results = [results for results in all_search_results if results]
    
    if not all_search_results:
        logger.warning("   -> 모든 검색 결과가 비어있습니다. 기본 답변을 제공합니다.")
        return "죄송합니다. 관련 정보를 찾을 수 없습니다. 다른 질문을 시도해주세요.", []
    
    for results in all_search_results:
        for i, doc in enumerate(results):
            doc_id = doc.metadata.get("id", doc.page_content)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (i + 60)

    sorted_docs_with_scores = sorted(
        rrf_scores.values(), key=lambda x: x["score"], reverse=True
    )
    fused_docs = [item["doc"] for item in sorted_docs_with_scores][:10]

    # CrossEncoder 재순위화 (사용 가능하면 CrossEncoder, 아니면 키워드 매칭)
    final_retrieved_docs = rerank_documents_crossencoder(query, fused_docs)[:5]

    if not final_retrieved_docs:
        return "관련 정보를 찾을 수 없습니다.", []

    logger.info("\n[최종 선별된 문서 (LLM 전달용)]")
    for i, doc in enumerate(final_retrieved_docs):
        logger.info(f"  {i+1}. [출처: {doc.metadata.get('title')}]")
    logger.info("-" * 20)

    context_str = "\n\n---\n\n".join(
        [
            f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}"
            for doc in final_retrieved_docs
        ]
    )
    source_info = [doc.metadata for doc in final_retrieved_docs]

    rag_prompt = f"""[지시]
당신은 주어진 [검색된 문서 내용]에서 사용자의 [질문]과 관련된 정보를 정확하게 찾아내어 요약하는 '정보 분석가'입니다.

[규칙]
1.  **반드시** [검색된 문서 내용]에 명시된 정보만을 사용하여 답변해야 합니다.
2.  절대로 문서에 없는 내용을 추론, 가정, 또는 보충하여 설명하지 마세요.
3.  답변은 [질문]에 대한 핵심 내용을 간결하게 요약하고, 관련된 내용을 직접 인용하는 형태로 구성하세요.
4.  [질문] 내에 설명이라는 문구가 들어간 경우, 그 [질문]에서 설명을 원하는 대상에 대한 구체적인 설명을 답변에 포함하세요.
5.  정보에 대한 구체적인 출처는 명시하지 마세요.
6.  답변에 "[", "]" 두 기호는 사용하지 마세요.

---
[검색된 문서 내용]
{context_str}
---
[질문]
{query}
[답변]"""

    logger.info(f"\n[단계 3: gpt-4o 모델로 사실 기반 답변 생성 (temperature=0.0)]")
    try:
        llm_final = ChatOpenAI(model_name="gpt-4o", temperature=0.0, openai_api_key=api_key)
        answer = llm_final.invoke(rag_prompt).content.strip()
        
        verification_prompt = f"""[지시]
아래 [생성된 답변]의 모든 문장이 [원본 문서]에 근거하고 있는지 검증하세요.
근거가 없는 문장이나 추론이 포함된 문장은 삭제하거나, [원본 문서]의 표현으로 수정하여 최종 답변을 만드세요.

[원본 문서]
{context_str}
---
[생성된 답변]
{answer}
---
[검증 완료된 최종 답변]"""
        
        logger.info("\n[단계 4: 생성된 답변에 대한 자체 검증 수행]")
        verified_answer = llm_final.invoke(verification_prompt).content.strip()
        
        return verified_answer, source_info
    except Exception as e:
        return f"답변 생성 중 오류 발생: {e}", []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 서버 시작 시 검색 시스템을 초기화합니다."""
    logger.info("서버 시작 시 검색 시스템 초기화...")
    success = initialize_search_system()
    if not success:
        logger.error("검색 시스템 초기화 실패")
        raise RuntimeError("검색 시스템 초기화 실패")
    yield
    logger.info("서버 종료...")

# FastAPI 앱 설정
app = FastAPI(
    title="한양대학교 AI 챗봇 API",
    description="한양대학교 정보를 검색하고 답변하는 AI 챗봇",
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

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "한양대학교 AI 챗봇 API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 API 엔드포인트"""
    try:
        logger.info(f"사용자 질문: {request.message}")
        
        # 특별 처리: 송우정에 대한 질문
        if "송우정" in request.message:
            special_response = "송우정은 한양대학교 정보시스템학과 22학번입니다. 학과 구성원이라면 누구나 한 번쯤 그 이름을 들어봤을 정도로, 그는 정보시스템학과를 대표하는 인물로 손꼽힙니다.\n학부 시절부터 두각을 드러낸 그는, 문제 해결 능력과 깊이 있는 사고력으로 주목받았습니다. 특히 알고리즘 실력은 타의 추종을 불허할 정도로 탁월했으며, 백준 온라인 저지에서는 다이아몬드 티어에 도달하며 실력을 객관적으로 증명해냈습니다. 이는 단순한 코딩 실력을 넘어, 논리적 사고와 집중력, 꾸준한 자기 주도 학습이 뒷받침된 결과였습니다.\n그의 성장 여정은 한양대학교를 넘어 서울대학교 데이터사이언스 대학원으로 이어졌습니다. 학문적 탐구를 멈추지 않고 더 넓은 세계로 나아가는 모습은, 후배들에게 귀감이 되고 있습니다. 특히 정보시스템학과 후배들 사이에서는 '본받고 싶은 선배'로 자주 언급되며, 학과의 긍정적인 전통을 이어가는 데 큰 영향을 미치고 있습니다.\n송우정은 뛰어난 실력만으로 평가받는 인물이 아닙니다. 항상 겸손한 태도로 주변을 대하고, 자신이 배운 지식을 아낌없이 나누며 함께 성장하는 문화를 만들어가는 사람입니다.\n정보시스템학과가 자랑스러워할 수 있는 한 이름, 그리고 후배들에게 좋은 길잡이가 되어주는 한 선배. 그가 바로, 송우정입니다."
            
            return ChatResponse(
                response=special_response,
                conversation_id=request.conversation_id or str(uuid.uuid4()),
                timestamp=datetime.now(),
                sources=[]
            )
        
        # RAG 시스템을 통한 답변 생성
        answer, sources = get_final_response(request.message)
        
        return ChatResponse(
            response=answer,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            timestamp=datetime.now(),
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    