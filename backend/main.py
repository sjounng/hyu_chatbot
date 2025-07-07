import os
import json
import pickle
from collections import OrderedDict
from typing import List, Set, Dict, Any, Optional
from datetime import datetime
import uuid
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers.cross_encoder import CrossEncoder

# --- FastAPI 모델 정의 ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class Source(BaseModel):
    id: str
    title: str
    content: str

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    sources: List[Source] = []

# --- 설정 ---
# 환경 변수 로드
env_paths = [
    ".env", 
    "o.env",
    "../.env",
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "o.env")
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"환경 변수 로드: {env_path}")
        break
else:
    load_dotenv()
    print("시스템 환경 변수 사용")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

CHAT_MODEL = "gpt-4o"

# 데이터 파일 경로 설정
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "chroma_db_hyu")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
DOC_FILE_PATH = os.path.join(DATA_DIR, "document.json")
MATCHED_JSON_PATH = os.path.join(DATA_DIR, "matched.json")
FEW_SHOT_SAMPLES_PATH = os.path.join(DATA_DIR, "question_sample.json")

# --- 프롬프트 정의 ---
SYSTEM_PROMPT_GAP_ANALYSIS = """[지시]
당신은 '정보 분석가'입니다. 주어진 [문서들]을 바탕으로 아래 작업을 수행하세요.
1. 사용자의 [질문]에 대해 현재까지의 정보로 답변을 생성합니다.
2. 현재 문서들만으로는 답변이 불완전할 경우, 부족한 정보가 무엇인지 간략히 설명합니다.
3. 부족한 정보를 보완하기 위해 필요한 '재질문(follow-up query)'을 생성합니다. 정보가 충분하다면 재질문은 빈 문자열로 남겨두세요.

[출력 형식]
반드시 아래의 JSON 형식에 맞춰서 출력해야 합니다.
```json
{{
  "answer": "...",
  "missing_info_reason": "...",
  "follow_up_query": "..."
}}
```"""

SYSTEM_PROMPT_FINAL_ANSWER_TEMPLATE = """[지시]
당신은 주어진 [문서들]의 내용을 종합하여 사용자의 [질문]에 대해 완전하고 상세한 답변을 생성하는 '답변 전문가'입니다.
- 문서들의 내용을 바탕으로, 논리적이고 이해하기 쉽게 답변을 구성하세요.
- 추측이 아닌, 제공된 정보에만 근거하여 답변해야 합니다.
- 아래 [예시]를 참고하여 동일한 스타일과 형식으로 답변을 생성하세요.

{few_shot_examples}
"""

# --- 전역 변수 ---
llm = None
embedding_model = None
cross_encoder = None
vector_retriever = None
bm25_retriever = None
title_to_doc_map = {}
all_titles = []
ABBREV_MAP = {}
final_answer_system_prompt = ""

def initialize_system():
    """시스템을 초기화합니다."""
    global llm, embedding_model, cross_encoder, vector_retriever, bm25_retriever, title_to_doc_map, all_titles, ABBREV_MAP, final_answer_system_prompt
    
    print("Self-RAG 시스템을 초기화합니다...")
    
    # --- 모델 로드 ---
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0, openai_api_key=API_KEY)
    embedding_model = OpenAIEmbeddings(openai_api_key=API_KEY)
    
    try:
        cross_encoder = CrossEncoder("bongsoo/kpf-cross-encoder-v1")
        print("CrossEncoder 모델 로드 완료")
    except Exception as e:
        print(f"CrossEncoder 로드 실패: {e}")
        cross_encoder = None

    # --- 데이터 로드 ---
    # 축약어 매핑 로딩
    with open(MATCHED_JSON_PATH, "r", encoding="utf-8") as f:
        ABBREV_MAP = {item["key"]: item["value"] for item in json.load(f)}

    # 벡터 검색 초기화
    try:
        # 런타임에서 벡터 DB 생성
        print("벡터 데이터베이스 생성 중...")
        with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
            all_docs_data = json.load(f)
        documents = [Document(page_content=item["content"], metadata=item) for item in all_docs_data]
        
        # Chroma 벡터스토어 생성
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=None  # 메모리에만 저장
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 50})
        print("벡터 검색 시스템 생성 완료")
    except Exception as e:
        print(f"벡터 검색 시스템 생성 실패: {e}")
        vector_retriever = None

    # BM25 검색 초기화
    try:
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_retriever = pickle.load(f)
                bm25_retriever.k = 50
            print("기존 BM25 인덱스 로드 완료")
        else:
            print("BM25 인덱스 생성 중...")
            with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
                all_docs_data = json.load(f)
            documents = [Document(page_content=item["content"], metadata=item) for item in all_docs_data]
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 50
            
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            print("BM25 인덱스 생성 및 저장 완료")
    except Exception as e:
        print(f"BM25 검색 시스템 초기화 실패: {e}")
        bm25_retriever = None

    # 문서 데이터 로드
    with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
        all_docs_data = json.load(f)
        title_to_doc_map = {item["title"]: Document(page_content=item["content"], metadata=item) for item in all_docs_data}
        all_titles = list(title_to_doc_map.keys())

    # --- Few-shot 예제 로드 및 프롬프트 생성 ---
    def load_and_format_few_shot_examples(file_path: str, example_ids: List[int]) -> str:
        """지정된 ID의 few-shot 예제를 로드하고 프롬프트 형식으로 만듭니다."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Few-shot 샘플 파일 '{file_path}'를 찾을 수 없습니다. Few-shot 프롬프트를 생략합니다.")
            return ""

        examples_str = ""
        for sample in samples:
            if sample["question_id"] in example_ids:
                example_context = f"문서 제목: {sample['source']['title']}\n내용: {sample['source']['content']}"
                examples_str += f"""[예시]
[문서들]
---
{example_context}
---

[질문]
{sample['question']}

[답변]
{sample['answer']}
"""
        return examples_str

    FEW_SHOT_EXAMPLE_IDS = [3666, 36032] 
    formatted_few_shot_examples = load_and_format_few_shot_examples(FEW_SHOT_SAMPLES_PATH, FEW_SHOT_EXAMPLE_IDS)
    final_answer_system_prompt = SYSTEM_PROMPT_FINAL_ANSWER_TEMPLATE.format(
        few_shot_examples=formatted_few_shot_examples
    )

    print("Self-RAG 시스템 초기화 완료!")

# --- 헬퍼 함수 ---
def expand_abbreviations(text: str) -> str:
    for key, value in ABBREV_MAP.items():
        text = text.replace(key, value)
    return text

def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    if not documents:
        return []
    
    if cross_encoder is not None:
        try:
            # CrossEncoder의 최대 입력 길이 제한 (512 토큰 ≈ 2000자)
            max_content_length = 2000
            pairs = []
            for doc in documents:
                content = doc.page_content
                # 텍스트가 너무 길면 자르기
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                pairs.append([query, content])
            
            scores = cross_encoder.predict(pairs, show_progress_bar=False)
            reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in reranked[:top_k]]
        except Exception as e:
            print(f"CrossEncoder 사용 중 오류: {e}")
    
    # 간단한 키워드 매칭으로 대체
    query_words = set(query.lower().split())
    doc_scores = []
    
    for doc in documents:
        content_lower = doc.page_content.lower()
        title_lower = doc.metadata.get('title', '').lower()
        
        content_matches = sum(1 for word in query_words if word in content_lower)
        title_matches = sum(2 for word in query_words if word in title_lower)
        
        total_score = content_matches + title_matches
        doc_scores.append((total_score, doc))
    
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in doc_scores[:top_k]]

def rrf_fuse(query: str, excluded_ids: Set[str]) -> List[Document]:
    golden = [title_to_doc_map[title] for title in all_titles if title in query and title_to_doc_map[title].metadata.get("id", title) not in excluded_ids]
    
    bm25 = []
    if bm25_retriever:
        try:
            bm25 = [doc for doc in bm25_retriever.invoke(query) if doc.metadata.get("id", doc.page_content) not in excluded_ids]
        except Exception as e:
            print(f"BM25 검색 오류: {e}")
    
    vector = []
    if vector_retriever:
        try:
            vector = [doc for doc in vector_retriever.invoke(query) if doc.metadata.get("id", doc.page_content) not in excluded_ids]
        except Exception as e:
            print(f"벡터 검색 오류: {e}")
    
    # 디버깅 정보 출력
    print(f"    📋 Golden: {len(golden)}개, 🔤 BM25: {len(bm25)}개, 🔍 Vector: {len(vector)}개")

    rrf_scores = {}
    for results in [golden, bm25, vector]:
        for i, doc in enumerate(results):
            doc_id = doc.metadata.get("id", doc.page_content)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "doc": doc}
            rrf_scores[doc_id]["score"] += 1.0 / (i + 60)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs[:10]]

# --- LLM 호출 함수 ---
def generate_initial_response_and_followup(query: str, docs: List[Document]) -> Dict[str, Any]:
    context_str = "\n\n---\n\n".join([f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}" for doc in docs])
    human_prompt = f"""[문서들]
{context_str}

[질문]
{query}"""
    messages = [SystemMessage(content=SYSTEM_PROMPT_GAP_ANALYSIS), HumanMessage(content=human_prompt)]
    try:
        response_content = llm.invoke(messages).content.strip()
        json_part = response_content[response_content.find('{'):response_content.rfind('}')+1]
        return json.loads(json_part)
    except Exception as e:
        print(f"초기 응답 생성 중 오류 발생: {e}")
        return {"answer": "답변 생성 실패", "missing_info_reason": "", "follow_up_query": ""}

def generate_final_answer(query: str, docs: List[Document]) -> str:
    context_str = "\n\n---\n\n".join([f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}" for doc in docs])
    human_prompt = f"""[문서들]
{context_str}

[질문]
{query}

[답변]
"""
    messages = [SystemMessage(content=final_answer_system_prompt), HumanMessage(content=human_prompt)]
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"최종 답변 생성 중 오류 발생: {e}")
        return "최종 답변 생성에 실패했습니다."

# --- Self-RAG 메인 로직 ---
def process_self_rag_query(user_query: str) -> tuple[str, List[Document]]:
    """Self-RAG 시스템을 사용하여 쿼리를 처리합니다."""
    
    # 송우정 특별 처리
    SONG_WOO_JEONG_ANSWER = """송우정은 한양대학교 정보시스템학과 22학번입니다. 학과 구성원이라면 누구나 한 번쯤 그 이름을 들어봤을 정도로, 그는 정보시스템학과를 대표하는 인물로 손꼽힙니다.
학부 시절부터 두각을 드러낸 그는, 문제 해결 능력과 깊이 있는 사고력으로 주목받았습니다. 특히 알고리즘 실력은 타의 추종을 불허할 정도로 탁월했으며, 백준 온라인 저지에서는 다이아몬드 티어에 도달하며 실력을 객관적으로 증명해냈습니다. 이는 단순한 코딩 실력을 넘어, 논리적 사고와 집중력, 꾸준한 자기 주도 학습이 뒷받침된 결과였습니다.
그의 성장 여정은 한양대학교를 넘어 서울대학교 데이터사이언스 대학원으로 이어졌습니다. 학문적 탐구를 멈추지 않고 더 넓은 세계로 나아가는 모습은, 후배들에게 귀감이 되고 있습니다. 특히 정보시스템학과 후배들 사이에서는 '본받고 싶은 선배'로 자주 언급되며, 학과의 긍정적인 전통을 이어가는 데 큰 영향을 미치고 있습니다.
송우정은 뛰어난 실력만으로 평가받는 인물이 아닙니다. 항상 겸손한 태도로 주변을 대하고, 자신이 배운 지식을 아낌없이 나누며 함께 성장하는 문화를 만들어가는 사람입니다.
정보시스템학과가 자랑스러워할 수 있는 한 이름, 그리고 후배들에게 좋은 길잡이가 되어주는 한 선배. 그가 바로, 송우정입니다."""

    if "송우정" in user_query:
        return SONG_WOO_JEONG_ANSWER, []

    expanded_query = expand_abbreviations(user_query)
    
    # 1차 검색
    initial_docs = rrf_fuse(expanded_query, excluded_ids=set())
    top_docs = rerank_documents(user_query, initial_docs, top_k=5)

    print(f"\n[🔍 1차 검색 결과] - {len(top_docs)}개 문서")
    for i, doc in enumerate(top_docs):
        print(f"  {i+1}. {doc.metadata.get('title')}")

    if not top_docs:
        return "관련된 문서를 찾지 못했습니다.", []

    # 1차 답변 및 재질의 생성
    result = generate_initial_response_and_followup(user_query, top_docs)
    print(f"\n[✅ 1차 답변] {result.get('answer', '(없음)')[:100]}...")

    initial_answer = result.get("answer", "(없음)")
    follow_up_query = result.get("follow_up_query")
    
    # 재질문이 필요 없는 경우
    if not follow_up_query or not follow_up_query.strip():
        print("\n[📌 추가 검색 불필요. 1차 답변으로 충분합니다.]")
        return initial_answer, top_docs

    # 2차 검색 수행
    print(f"\n[🔁 재질의 생성됨 → '{follow_up_query}']")

    excluded_ids = {doc.metadata.get("id", doc.page_content) for doc in top_docs}
    new_docs = rrf_fuse(follow_up_query, excluded_ids=excluded_ids)
    new_top_docs = rerank_documents(follow_up_query, new_docs, top_k=5)

    print(f"\n[🔍 2차 검색 결과] - {len(new_top_docs)}개 문서")
    for i, doc in enumerate(new_top_docs):
        print(f"  {i+1}. {doc.metadata.get('title')}")

    # 최종 문서 통합
    merged_docs = rerank_documents(user_query, top_docs + new_top_docs, top_k=5)
    print(f"\n[📚 최종 문서 선택] - {len(merged_docs)}개 문서")
    for i, doc in enumerate(merged_docs):
        print(f"  {i+1}. {doc.metadata.get('title')}")

    if merged_docs:
        # 2차 검색까지 마친 후에만 few-shot 프롬프트가 포함된 최종 답변 생성 함수를 호출
        final_answer = generate_final_answer(user_query, merged_docs)
        return final_answer, merged_docs
    else:
        # 1차 답변은 있었지만, 2차 검색 결과가 없어 최종 문서를 못 고른 경우
        return initial_answer, top_docs

# --- FastAPI 설정 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 서버 시작 시 Self-RAG 시스템을 초기화합니다."""
    initialize_system()
    yield

app = FastAPI(title="Self-RAG ChatBot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 엔드포인트 ---
@app.get("/")
async def root():
    return {"message": "Self-RAG ChatBot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")
        
        # Self-RAG로 답변 생성
        final_answer, used_docs = process_self_rag_query(request.message)
        
        # 소스 문서 정보 생성
        sources = []
        for doc in used_docs:
            sources.append(Source(
                id=str(doc.metadata.get("id", hash(doc.page_content))),
                title=doc.metadata.get("title", "제목 없음"),
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            ))
        
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        return ChatResponse(
            response=final_answer,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            sources=sources
        )
        
    except Exception as e:
        print(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    