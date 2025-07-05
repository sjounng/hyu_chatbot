# 한양대학교 AI 챗봇 백엔드

FastAPI 기반의 한양대학교 정보 제공 AI 챗봇 백엔드입니다.

## 🚀 기능

- **RAG 챗봇**: 벡터 DB 기반 검색과 OpenAI GPT 모델을 활용한 정확한 정보 제공
- **벡터 검색**: 사용자 질문과 관련된 문서를 실시간으로 검색
- **대화 관리**: 대화 기록 저장 및 관리
- **컨텍스트 생성**: 검색된 정보를 바탕으로 AI 응답 생성

## 📁 프로젝트 구조

```
backend/
├── main.py                 # FastAPI 애플리케이션 진입점
├── config.py              # 설정 관리
├── requirements.txt       # Python 의존성
├── env.example           # 환경 변수 예시
├── models/               # 데이터 모델
│   ├── __init__.py
│   ├── chat.py          # 채팅 관련 모델
│   └── document.py      # 문서 관련 모델
├── services/            # 비즈니스 로직
│   ├── __init__.py
│   ├── chat_service.py  # RAG 챗봇 서비스
│   ├── document_service.py  # 문서 검색 서비스
│   └── embedding_service.py # 임베딩 서비스
├── api/                 # API 라우터
│   ├── __init__.py
│   ├── chat.py         # 채팅 API
│   └── documents.py    # 문서 검색 API
└── utils/              # 유틸리티 함수
    ├── __init__.py
    ├── text_processing.py  # 텍스트 처리
    └── validation.py      # 입력 검증
```

## 🛠️ 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp env.example .env
# .env 파일을 편집하여 OpenAI API 키 등 설정
```

### 4. 서버 실행

```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 📚 API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 주요 API 엔드포인트

### 채팅 API

- `POST /api/v1/chat/` - 채팅 메시지 전송
- `GET /api/v1/chat/history/{conversation_id}` - 대화 기록 조회
- `DELETE /api/v1/chat/conversation/{conversation_id}` - 대화 삭제

### 문서 API

- `POST /api/v1/documents/crawl` - 위키 페이지 크롤링
- `GET /api/v1/documents/` - 모든 문서 조회
- `GET /api/v1/documents/{document_id}` - 특정 문서 조회
- `GET /api/v1/documents/search/{query}` - 문서 검색
- `GET /api/v1/documents/context/{query}` - 컨텍스트 생성

## 🔑 환경 변수

| 변수명                     | 설명              | 기본값          |
| -------------------------- | ----------------- | --------------- |
| `OPENAI_API_KEY`           | OpenAI API 키     | -               |
| `OPENAI_MODEL`             | 사용할 GPT 모델   | `gpt-3.5-turbo` |
| `HOST`                     | 서버 호스트       | `0.0.0.0`       |
| `PORT`                     | 서버 포트         | `8000`          |
| `CHROMA_PERSIST_DIRECTORY` | 벡터 DB 저장 경로 | `./chroma_db`   |

## 🤝 개발 가이드

### 새로운 API 추가

1. `api/` 폴더에 새로운 라우터 파일 생성
2. `main.py`에 라우터 등록
3. 필요한 서비스 로직을 `services/` 폴더에 구현

### 새로운 모델 추가

1. `models/` 폴더에 Pydantic 모델 정의
2. `models/__init__.py`에 import 추가

## 📝 라이센스

MIT License
