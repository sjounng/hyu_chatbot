import os
import json
import pickle
import random
from collections import OrderedDict
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List
from dotenv import load_dotenv

# --- 0. 설정 및 초기화 ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 및 파일 경로 설정
CHAT_MODEL = 'gpt-4o'
VECTOR_DB_PATH = "./chroma_db_hyu"
BM25_INDEX_PATH = "./bm25_index.pkl"
DOC_FILE_PATH = "hyuwiki_documents_20250621_234549.json"
QA_FILE_PATH = "qa_random_200_samples_20250622_203907.json"

# --- 1. 검색 시스템 로드 ---
print("1. 검색 시스템 및 전체 문서 데이터를 로드합니다...")
# 1-1. 벡터 검색기 로드 (의미 기반)
embedding_model = OpenAIEmbeddings(openai_api_key=API_KEY)
vector_retriever = Chroma(
    persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model
).as_retriever(search_kwargs={"k": 20})

# 1-2. BM25 검색기 로드 (키워드 기반)
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)
    bm25_retriever.k = 20

# 1-3. Title 검색을 위한 데이터 로드
with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
    all_docs_data = json.load(f)
title_to_doc_map = {item["title"]: Document(page_content=item["content"], metadata=item) for item in all_docs_data}
all_titles = list(title_to_doc_map.keys())
print("   -> 로드 완료!")

# --- 2. Few-shot 예시 생성 ---
with open(QA_FILE_PATH, "r", encoding="utf-8") as f:
    qa_samples = json.load(f)
few_shot_examples = random.sample(qa_samples, 2)
few_shot_prompt_part = "\n\n".join(
    [f"예시 질문: {ex['question']}\n예시 답변: {ex['answer']}" for ex in few_shot_examples]
)


# --- 3. 최종 RAG 답변 생성 함수 (RRF 적용) ---
def get_final_response(query: str):
    """사용자 질문을 받아, 다중 검색, RRF 융합, 최종 답변 생성을 수행합니다."""
    print("\n[단계 1: 초고속 동시 검색 (No-API)]")
    
    # 검색 A: 확정적 Title 검색 ('골든 티켓')
    golden_docs = []
    query_no_space = query.replace(" ", "")
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if title in query or title_no_space in query_no_space:
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        print(f"   -> '골든 티켓' 발견: {[doc.metadata['title'] for doc in golden_docs]}")

    # 검색 B: BM25 키워드 검색
    bm25_docs = bm25_retriever.invoke(query)
    print(f"   -> BM25 검색으로 {len(bm25_docs)}개의 후보를 찾았습니다.")

    # 검색 C: 벡터 의미 기반 검색
    # 이 과정에서 query 임베딩을 위해 API가 1회 호출됩니다. (매우 빠름)
    vector_docs = vector_retriever.invoke(query)
    print(f"   -> 벡터 검색으로 {len(vector_docs)}개의 후보를 찾았습니다.")

    # [단계 2: Reciprocal Rank Fusion (RRF)으로 순위 융합]
    print("\n[단계 2: RRF를 이용한 순위 융합 (No-API)]")
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
        
    print("\n[최종 선별된 문서 (LLM 전달용)]")
    for i, doc in enumerate(final_retrieved_docs):
        print(f"  {i+1}. [출처: {doc.metadata.get('title')}]")
    print("-" * 20)
    
    # 3. 최종 답변 생성
    context_str = "\n\n---\n\n".join([f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}" for doc in final_retrieved_docs])
    source_info = [doc.metadata for doc in final_retrieved_docs]
    
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
    
    print(f"\n[단계 3: {CHAT_MODEL} 모델로 최종 답변 생성 (유일한 API 호출)]")
    try:
        llm_final = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.2, openai_api_key=API_KEY)
        answer = llm_final.invoke(rag_prompt).content.strip()
        return answer, source_info
    except Exception as e:
        return f"답변 생성 중 오류 발생: {e}", []

# --- 4. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n챗봇을 시작합니다. (종료하려면 'exit' 또는 '종료' 입력)")
    while True:
        user_query = input("\n🤔 질문을 입력하세요: ")
        if user_query.lower() in ['exit', '종료']:
            print("🤖 챗봇을 종료합니다."); break
        answer, sources = get_final_response(user_query)
        print("\n" + "="*50)
        print(f"🤖 답변:\n{answer}")
        if sources:
            unique_sources = list(OrderedDict.fromkeys((src.get('title'), src.get('url')) for src in sources))
            print("\n📚 참고 자료:")
            for title, url in unique_sources:
                print(f"  - {title} ({url})")
        print("="*50)
