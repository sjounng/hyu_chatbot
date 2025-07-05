# 한양대학교 AI 챗봇

한양대학교 관련 정보를 제공하는 AI 챗봇입니다. FastAPI 백엔드와 Next.js 프론트엔드로 구성되어 있습니다.

## 🚀 주요 기능

- **RAG (Retrieval-Augmented Generation) 시스템**: 한양대학교 관련 문서를 기반으로 한 정확한 정보 제공
- **실시간 채팅**: 사용자와의 자연스러운 대화
- **중지 기능**: AI 응답 중 언제든지 중지 가능
- **타이핑 애니메이션**: AI 응답을 타이핑하는 효과
- **반응형 디자인**: 모바일과 데스크톱에서 모두 사용 가능

## 🛠️ 기술 스택

### 백엔드

- **FastAPI**: Python 기반 웹 프레임워크
- **LangChain**: LLM 통합 및 RAG 시스템
- **OpenAI GPT-4**: AI 모델
- **Chroma**: 벡터 데이터베이스
- **BM25**: 키워드 기반 검색

### 프론트엔드

- **Next.js 14**: React 기반 프레임워크
- **TypeScript**: 타입 안전성
- **Tailwind CSS**: 스타일링
- **Framer Motion**: 애니메이션

## 📁 프로젝트 구조

```
fastapi-llm-chatbot/
├── backend/                 # FastAPI 백엔드
│   ├── main.py             # 메인 서버 파일
│   ├── requirements.txt     # Python 의존성
│   └── github_backend/     # 데이터 파일들
├── frontend/               # Next.js 프론트엔드
│   └── ai-chatbot/        # 채팅 애플리케이션
└── README.md              # 프로젝트 설명
```

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/ShinBanSeok/hyu_chatbot.git
cd hyu_chatbot
```

### 2. 백엔드 설정

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# backend/.env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 백엔드 실행

```bash
python main.py
# 서버가 http://localhost:8000 에서 실행됩니다
```

### 5. 프론트엔드 설정

```bash
cd frontend/ai-chatbot
npm install
```

### 6. 프론트엔드 실행

```bash
npm run dev
# 애플리케이션이 http://localhost:3000 에서 실행됩니다
```

## 🎯 사용 방법

1. 브라우저에서 `http://localhost:3000` 접속
2. 한양대학교 관련 질문 입력
3. AI가 상세한 답변을 제공
4. 응답 중 중지 버튼으로 언제든지 중단 가능

## 🔧 주요 기능 설명

### RAG 시스템

- **프롬프트 재구성**: 사용자 질문을 더 정확한 검색을 위해 AI가 재구성
- **하이브리드 검색**: 벡터 검색과 키워드 검색을 결합
- **문서 선별**: 관련성 높은 문서만 선별하여 답변 생성

### 중지 기능

- **API 요청 중지**: 백엔드 요청을 완전히 취소
- **타이핑 중지**: 현재까지 타이핑된 부분까지만 표시
- **자연스러운 중단**: 사용자가 원하는 시점에 중단 가능

## 📝 API 엔드포인트

- `POST /api/v1/chat/`: 채팅 메시지 처리
- `GET /health`: 서버 상태 확인
- `GET /`: 루트 엔드포인트

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 팀원

- 한양대학교 AI 챗봇 개발팀

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.
