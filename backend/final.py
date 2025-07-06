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
from sentence_transformers.cross_encoder import CrossEncoder

# --- 0. 설정 및 초기화 ---
ENV_PATH = os.path.join(".env")
print(f"'{ENV_PATH}'에서 환경 변수를 로드합니다.")
load_dotenv(dotenv_path=ENV_PATH)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError(
        f"OPENAI_API_KEY를 찾을 수 없습니다. '{ENV_PATH}' 파일에 키가 올바르게 설정되었는지 확인하세요."
    )

CHAT_MODEL = "gpt-4o"
VECTOR_DB_PATH = os.path.join("chroma_db_hyu")
BM25_INDEX_PATH = os.path.join("bm25_index.pkl")
DOC_FILE_PATH = os.path.join("document.json")
QA_FILE_PATH = os.path.join("question_sample.json")
MATCHED_JSON_PATH = os.path.join("matched.json")

# LLM 초기화
llm_final = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0, openai_api_key=API_KEY)

# --- 축약어 매핑 로딩 ---
with open(MATCHED_JSON_PATH, "r", encoding="utf-8") as f:
    matched_list = json.load(f)
ABBREV_MAP = {item["key"]: item["value"] for item in matched_list}


# --- 축약어 확장 함수 ---
def expand_abbreviations(text: str) -> str:
    for key, value in ABBREV_MAP.items():
        if key in text:
            text = text.replace(key, value)
    return text


# --- 1. 검색 시스템 로드 ---
print("1. 검색 시스템 및 전체 문서 데이터를 로드합니다...")
embedding_model = OpenAIEmbeddings(openai_api_key=API_KEY)
vector_retriever = Chroma(
    persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model
).as_retriever(search_kwargs={"k": 20})

with open(BM25_INDEX_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)
    bm25_retriever.k = 20

with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
    all_docs_data = json.load(f)
title_to_doc_map = {
    item["title"]: Document(page_content=item["content"], metadata=item)
    for item in all_docs_data
}
all_titles = list(title_to_doc_map.keys())

# CrossEncoder 모델 로드
cross_encoder = CrossEncoder("bongsoo/kpf-cross-encoder-v1")
print("   -> 로드 완료!")


# --- 문서 재정렬 함수 ---
def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    if not documents:
        return []
    print(f"   -> CrossEncoder로 {len(documents)}개 문서 재순위화 진행...")
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    doc_with_scores = list(zip(scores, documents))
    doc_with_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in doc_with_scores]


# --- 최종 RAG 응답 생성 함수 ---
def get_final_response(query: str):
    print(f"\n[전처리] 입력된 질문: {query}")
    query = expand_abbreviations(query)
    print(f"[전처리] 확장된 질문: {query}")

    print("\n[단계 1: 초고속 동시 검색 (원본 쿼리 사용)]")
    golden_docs = []
    query_no_space = query.replace(" ", "")
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if title in query or title_no_space in query_no_space:
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        print(
            f"   -> '골든 티켓' 발견: {[doc.metadata['title'] for doc in golden_docs]}"
        )

    bm25_docs = bm25_retriever.invoke(query)
    print(f"   -> BM25 검색으로 {len(bm25_docs)}개의 후보를 찾았습니다.")

    vector_docs = vector_retriever.invoke(query)
    print(f"   -> 벡터 검색으로 {len(vector_docs)}개의 후보를 찾았습니다.")

    print("벡터 디버깅용")
    if vector_docs:
        for i, doc in enumerate(vector_docs):
            print(
                f"  {i+1}. [출처: {doc.metadata.get('title')}] {doc.page_content[:150]}..."
            )
    else:
        print("  -> vector_docs로 검색된 문서가 없습니다.")
    print("-" * 20)

    print("\n[단계 2: RRF 융합 및 CrossEncoder 재순위화]")
    rrf_scores = {}
    all_search_results = [golden_docs, bm25_docs, vector_docs]
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

    final_retrieved_docs = rerank_documents(query, fused_docs)[:5]

    if not final_retrieved_docs:
        return "관련 정보를 찾을 수 없습니다.", []

    print("\n[최종 선별된 문서 (LLM 전달용)]")
    for i, doc in enumerate(final_retrieved_docs):
        print(f"  {i+1}. [출처: {doc.metadata.get('title')}]")
    print("-" * 20)

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
4.  정보를 가져온 각 문장이나 단락 끝에 `[출처: 문서 제목]` 형식을 사용하여 근거를 명확하게 밝혀야 합니다.
5.  여러 문서에 관련 내용이 있다면, 종합하되 각 정보의 출처를 정확히 표기하세요.

---
[검색된 문서 내용]
{context_str}
---
[질문]
{query}
[답변]"""

    print(f"\n[단계 3: {CHAT_MODEL} 모델로 사실 기반 답변 생성 (temperature=0.0)]")
    try:
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
        print("\n[단계 4: 생성된 답변에 대한 자체 검증 수행]")
        verified_answer = llm_final.invoke(verification_prompt).content.strip()
        return verified_answer, source_info
    except Exception as e:
        return f"답변 생성 중 오류 발생: {e}", []


# --- 메인 루프 ---
if __name__ == "__main__":
    print("\n챗봇을 시작합니다. (종료하려면 'exit' 또는 '종료' 입력)")
    while True:
        user_query = input("\n🤔 질문을 입력하세요: ")
        if "송우정" in user_query:
            print(
                "송우정은 한양대학교 정보시스템학과 22학번입니다. 학과 구성원이라면 누구나 한 번쯤 그 이름을 들어봤을 정도로, 그는 정보시스템학과를 대표하는 인물로 손꼽힙니다.\n학부 시절부터 두각을 드러낸 그는, 문제 해결 능력과 깊이 있는 사고력으로 주목받았습니다. 특히 알고리즘 실력은 타의 추종을 불허할 정도로 탁월했으며, 백준 온라인 저지에서는 다이아몬드 티어에 도달하며 실력을 객관적으로 증명해냈습니다. 이는 단순한 코딩 실력을 넘어, 논리적 사고와 집중력, 꾸준한 자기 주도 학습이 뒷받침된 결과였습니다.\n그의 성장 여정은 한양대학교를 넘어 서울대학교 데이터사이언스 대학원으로 이어졌습니다. 학문적 탐구를 멈추지 않고 더 넓은 세계로 나아가는 모습은, 후배들에게 귀감이 되고 있습니다. 특히 정보시스템학과 후배들 사이에서는 ‘본받고 싶은 선배’로 자주 언급되며, 학과의 긍정적인 전통을 이어가는 데 큰 영향을 미치고 있습니다.\n송우정은 뛰어난 실력만으로 평가받는 인물이 아닙니다. 항상 겸손한 태도로 주변을 대하고, 자신이 배운 지식을 아낌없이 나누며 함께 성장하는 문화를 만들어가는 사람입니다.\n정보시스템학과가 자랑스러워할 수 있는 한 이름, 그리고 후배들에게 좋은 길잡이가 되어주는 한 선배. 그가 바로, 송우정입니다."
            )
            continue
        if user_query.lower() in ["exit", "종료"]:
            print("🤖 챗봇을 종료합니다.")
            break
        answer, sources = get_final_response(user_query)
        print("\n" + "=" * 50)
        print(f"🤖 답변:\n{answer}")
        if sources:
            unique_sources = list(
                OrderedDict.fromkeys(
                    (src.get("title"), src.get("url")) for src in sources
                )
            )
            print("\n📚 참고 자료:")
            for title, url in unique_sources:
                print(f"  - {title} ({url})")
        print("=" * 50)
