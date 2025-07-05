import re
from urllib.parse import urlparse
from typing import Optional

def validate_url(url: str) -> bool:
    """URL이 유효한지 검증합니다."""
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """텍스트가 유효한지 검증합니다."""
    if not text:
        return False
    
    text_length = len(text.strip())
    return min_length <= text_length <= max_length

def validate_email(email: str) -> bool:
    """이메일이 유효한지 검증합니다."""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_input(text: str) -> str:
    """사용자 입력을 안전하게 정리합니다."""
    if not text:
        return ""
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 스크립트 태그 제거
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 위험한 문자들 제거
    text = re.sub(r'[<>"\']', '', text)
    
    return text.strip()

def validate_conversation_id(conversation_id: str) -> bool:
    """대화 ID가 유효한지 검증합니다."""
    if not conversation_id:
        return False
    
    # UUID 형식 검증
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, conversation_id.lower())) 