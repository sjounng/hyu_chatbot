import { useState, useEffect, useRef } from "react";
import { ChatMessage } from "../types/chat";
import { chatAPI } from "../utils/api";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId] = useState<string | undefined>();
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
    // 안전한 텍스트 처리
    const safeText = fullText && typeof fullText === "string" ? fullText : "";
    if (!safeText) {
      setIsLoading(false);
      return;
    }

    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex <= safeText.length) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, displayedText: safeText.slice(0, currentIndex) }
              : msg
          )
        );
        currentIndex++;
      } else {
        clearInterval(interval);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, isTypingResponse: false, displayedText: safeText }
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

      // 응답 안전성 검사
      if (!response || typeof response !== "string") {
        return "죄송합니다. 서버에서 올바르지 않은 응답을 받았습니다.";
      }

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
    const globalWindow = window as unknown as Record<string, unknown>;
    if (globalWindow.__typingCleanup) {
      const cleanup = globalWindow.__typingCleanup as () => void;
      cleanup();
      globalWindow.__typingCleanup = null;
    }

    setIsLoading(false);

    // 타이핑 중인 메시지의 타이핑 상태를 완료로 변경하되, 현재까지 타이핑된 텍스트는 유지
    setMessages((prev) =>
      prev.map((msg) =>
        msg.isTypingResponse
          ? {
              ...msg,
              isTypingResponse: false,
              text: msg.displayedText || msg.text || "응답이 중단되었습니다.",
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
        const globalWindow = window as unknown as Record<string, unknown>;
        globalWindow.__typingCleanup = cleanup;
      }
    }, 100);
  };

  const handleSendMessage = (question: string) => {
    // 입력값 안전성 검사
    if (!question || typeof question !== "string" || !question.trim()) {
      return;
    }

    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      text: question.trim(),
      isUser: true,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);

    simulateAIResponse(question.trim());
  };

  return {
    messages,
    messagesEndRef,
    handleSendMessage,
    isLoading,
    stopGeneration,
  };
}
