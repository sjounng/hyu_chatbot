// API 타입 정의
interface ChatRequest {
  message: string;
  conversation_id?: string;
}

interface Source {
  id: string;
  title: string;
  content: string;
  [key: string]: unknown;
}

interface ChatResponse {
  response: string;
  conversation_id: string;
  timestamp: string;
  sources: Source[];
}

const API_BASE_URL = "http://localhost:8000";

export const chatAPI = {
  /**
   * 채팅 메시지를 백엔드로 전송
   */
  async sendMessage(
    userMessage: string,
    conversationId?: string,
    signal?: AbortSignal
  ): Promise<string> {
    try {
      const requestBody: ChatRequest = {
        message: userMessage,
        conversation_id: conversationId,
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/chat/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      return data.response;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        return "요청이 취소되었습니다.";
      }
      console.error("API 호출 오류:", error);
      return "죄송합니다. 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.";
    }
  },

  /**
   * 서버 상태 확인
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error("Health check 실패:", error);
      return false;
    }
  },
};
