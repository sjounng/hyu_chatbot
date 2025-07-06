"use client";

import { ChatMessage as ChatMessageType } from "../types/chat";

interface ChatMessageProps {
  message: ChatMessageType;
  index: number;
}

export default function ChatMessage({ message, index }: ChatMessageProps) {
  // 안전한 텍스트 렌더링을 위한 헬퍼 함수
  const getSafeText = (text?: string): string => {
    return text && typeof text === "string" ? text : "";
  };

  const getDisplayText = (): string => {
    if (message.isTypingResponse) {
      return getSafeText(message.displayedText);
    }
    return getSafeText(message.displayedText) || getSafeText(message.text);
  };

  return (
    <div
      className={`flex w-full animate-in slide-in-from-bottom-2 duration-300 ease-out ${
        message.isUser ? "justify-end" : "justify-start"
      }`}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="flex flex-col">
        <div
          className={`max-w-2xl lg:max-w-4xl px-6 py-4 rounded-lg transition-all duration-300 ease-out ${
            message.isUser
              ? "bg-[#0E4A84] text-white rounded-br-none hover:bg-[#0a3a6b]"
              : "bg-gray-200 text-gray-800 rounded-bl-none hover:bg-gray-300"
          }`}
        >
          <div className="text-base leading-relaxed">
            {message.isTyping ? (
              <div className="thinking-dots-smooth">
                <div className="thinking-dot-smooth"></div>
                <div className="thinking-dot-smooth"></div>
                <div className="thinking-dot-smooth"></div>
              </div>
            ) : (
              <span>
                {message.isTypingResponse ? (
                  <span>
                    {getDisplayText()}
                    {getDisplayText() && (
                      <span className="typing-cursor">|</span>
                    )}
                  </span>
                ) : (
                  getDisplayText() || "메시지를 불러오는 중..."
                )}
              </span>
            )}
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-1 ml-2 animate-in fade-in duration-500 delay-200">
          {message.timestamp?.toLocaleTimeString?.("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          }) || ""}
        </p>
      </div>
    </div>
  );
}
