import uuid
from datetime import datetime
from typing import List, Optional
from openai import OpenAI
from ..config import settings
from ..models.chat import ChatRequest, ChatResponse, ChatMessage
from .document_service import DocumentService

class ChatService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.conversations = {}  # 임시 메모리 저장소
        self.document_service = DocumentService()  # RAG를 위한 문서 서비스
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """사용자 메시지를 처리하고 AI 응답을 생성합니다."""
        
        # 대화 ID 생성 또는 기존 ID 사용
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 대화 기록 가져오기
        conversation = self.conversations.get(conversation_id, [])
        
        # 사용자 메시지 추가
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
        conversation.append(user_message)
        
        # RAG를 사용한 AI 응답 생성
        ai_response = await self._generate_rag_response(request.message, conversation)
        
        # AI 메시지 추가
        assistant_message = ChatMessage(
            role="assistant",
            content=ai_response,
            timestamp=datetime.now(),
            conversation_id=conversation_id
        )
        conversation.append(assistant_message)
        
        # 대화 기록 저장
        self.conversations[conversation_id] = conversation
        
        return ChatResponse(
            response=ai_response,
            conversation_id=conversation_id,
            timestamp=datetime.now()
        )
    
    async def _generate_rag_response(self, user_message: str, conversation: List[ChatMessage]) -> str:
        """RAG를 사용하여 AI 응답을 생성합니다."""
        
        try:
            # 1. Retrieval: 사용자 질문과 관련된 문서 검색
            context = await self.document_service.get_context_for_query(user_message)
            
            # 2. Augmentation: 시스템 프롬프트에 컨텍스트 추가
            if context:
                system_prompt = f"""당신은 한양대학교에 대한 정보를 제공하는 AI 어시스턴트입니다.

다음은 한양대학교에 대한 참고 정보입니다:
{context}

위의 정보를 바탕으로 정확하고 친근하게 답변해주세요. 
정보에 없는 내용에 대해서는 솔직히 모른다고 답변하세요."""
            else:
                system_prompt = """당신은 한양대학교에 대한 정보를 제공하는 AI 어시스턴트입니다. 
                한양대학교의 역사, 학과, 캠퍼스, 입학 정보 등에 대해 정확하고 친근하게 답변해주세요.
                모르는 정보에 대해서는 솔직히 모른다고 답변하세요."""
            
            # 3. Generation: 대화 기록을 OpenAI 형식으로 변환
            messages = [{"role": "system", "content": system_prompt}]
            
            # 최근 10개의 메시지만 포함 (토큰 제한 고려)
            recent_messages = conversation[-10:] if len(conversation) > 10 else conversation
            for msg in recent_messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"죄송합니다. 응답을 생성하는 중 오류가 발생했습니다: {str(e)}"
    
    def get_conversation_history(self, conversation_id: str) -> List[ChatMessage]:
        """특정 대화의 기록을 반환합니다."""
        return self.conversations.get(conversation_id, [])
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """특정 대화를 삭제합니다."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False 