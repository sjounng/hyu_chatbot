from fastapi import APIRouter, HTTPException
from typing import List
from ..models.chat import ChatRequest, ChatResponse, ChatMessage
from ..services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# 서비스 인스턴스 생성
chat_service = ChatService()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지를 처리하고 AI 응답을 반환합니다."""
    try:
        response = await chat_service.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/history/{conversation_id}", response_model=List[ChatMessage])
async def get_conversation_history(conversation_id: str):
    """특정 대화의 기록을 반환합니다."""
    try:
        history = chat_service.get_conversation_history(conversation_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 기록 조회 중 오류가 발생했습니다: {str(e)}")

@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """특정 대화를 삭제합니다."""
    try:
        success = chat_service.delete_conversation(conversation_id)
        if success:
            return {"message": "대화가 성공적으로 삭제되었습니다."}
        else:
            raise HTTPException(status_code=404, detail="해당 대화를 찾을 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 삭제 중 오류가 발생했습니다: {str(e)}") 