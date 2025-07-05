import re
from typing import List

def clean_text(text: str) -> str:
    """텍스트를 정리합니다."""
    if not text:
        return ""
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 특수 문자 정리
    text = re.sub(r'[^\w\s가-힣.,!?;:()[\]{}"\'-]', '', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """텍스트를 청크로 분할합니다."""
    if not text:
        return []
    
    # 문장 단위로 분할
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때의 길이
        potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # 현재 청크가 있으면 저장
            if current_chunk:
                chunks.append(current_chunk)
            
            # 새로운 청크 시작 (이전 청크의 끝 부분을 포함)
            if overlap > 0 and chunks:
                last_chunk = chunks[-1]
                overlap_text = last_chunk[-overlap:] if len(last_chunk) > overlap else last_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """텍스트에서 키워드를 추출합니다."""
    if not text:
        return []
    
    # 한글, 영문 단어 추출
    words = re.findall(r'[가-힣a-zA-Z]+', text.lower())
    
    # 단어 빈도 계산
    word_freq = {}
    for word in words:
        if len(word) > 1:  # 1글자 단어 제외
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 빈도순으로 정렬
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 키워드 반환
    return [word for word, freq in sorted_words[:max_keywords]] 