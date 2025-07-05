import { useState, useEffect, useRef } from "react";
import { ChatMessage } from "../types/chat";
import { chatAPI } from "../utils/api";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);
  const [typingInterval, setTypingInterval] = useState<NodeJS.Timeout | null>(
    null
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const typeText = (messageId: string, fullText: string) => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex <= fullText.length) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, displayedText: fullText.slice(0, currentIndex) }
              : msg
          )
        );
        currentIndex++;
      } else {
        clearInterval(interval);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, isTypingResponse: false, displayedText: fullText }
              : msg
          )
        );
        setIsLoading(false);
        setTypingInterval(null);
      }
    }, 50);

    setTypingInterval(interval);

    // 중지 시 현재 상태를 유지하기 위한 cleanup 함수 반환
    return () => {
      clearInterval(interval);
      setTypingInterval(null);
    };
  };

  const callBackendAPI = async (userMessage: string): Promise<string> => {
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await chatAPI.sendMessage(
        userMessage,
        conversationId,
        controller.signal
      );

      return response;
    } catch (error) {
      console.error("API 호출 오류:", error);
      return "죄송합니다. 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.";
    } finally {
      setAbortController(null);
    }
  };

  const stopGeneration = () => {
    // API 요청 중지
    if (abortController) {
      abortController.abort();
    }

    // 타이핑 중지
    if (typingInterval) {
      clearInterval(typingInterval);
      setTypingInterval(null);
    }

    // cleanup 함수 호출
    if ((window as any).__typingCleanup) {
      (window as any).__typingCleanup();
      (window as any).__typingCleanup = null;
    }

    setIsLoading(false);

    // 타이핑 중인 메시지의 타이핑 상태를 완료로 변경하되, 현재까지 타이핑된 텍스트는 유지
    setMessages((prev) =>
      prev.map((msg) =>
        msg.isTypingResponse
          ? {
              ...msg,
              isTypingResponse: false,
              text: msg.displayedText || msg.text,
            }
          : msg
      )
    );
  };

  const simulateAIResponse = async (userMessage: string) => {
    const typingMessage: ChatMessage = {
      id: `typing-${Date.now()}`,
      text: "...",
      isUser: false,
      timestamp: new Date(),
      isTyping: true,
    };

    setMessages((prev) => [...prev, typingMessage]);
    setIsLoading(true);

    // 백엔드 API 호출
    const aiResponseText = await callBackendAPI(userMessage);

    const aiResponse: ChatMessage = {
      id: `ai-${Date.now()}`,
      text: aiResponseText,
      isUser: false,
      timestamp: new Date(),
      isTypingResponse: true,
      displayedText: "",
    };

    setMessages((prev) => {
      const filteredMessages = prev.filter((msg) => !msg.isTyping);
      return [...filteredMessages, aiResponse];
    });

    // 로딩 상태는 타이핑이 완료된 후에만 false로 변경됨

    setTimeout(() => {
      const cleanup = typeText(aiResponse.id, aiResponseText);

      // 중지 시 cleanup 함수를 호출할 수 있도록 저장
      if (cleanup) {
        // cleanup 함수를 전역에서 접근할 수 있도록 저장
        (window as any).__typingCleanup = cleanup;
      }
    }, 100);
  };

  const handleSendMessage = (question: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      text: question,
      isUser: true,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);

    simulateAIResponse(question);
  };

  return {
    messages,
    messagesEndRef,
    handleSendMessage,
    isLoading,
    stopGeneration,
  };
}
