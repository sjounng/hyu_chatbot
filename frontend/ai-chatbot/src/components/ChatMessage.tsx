"use client";

import { ChatMessage as ChatMessageType } from "../types/chat";

interface ChatMessageProps {
  message: ChatMessageType;
  index: number;
}

export default function ChatMessage({ message, index }: ChatMessageProps) {
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
                    {message.displayedText}
                    <span className="typing-cursor">|</span>
                  </span>
                ) : (
                  message.displayedText || message.text
                )}
              </span>
            )}
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-1 ml-2 animate-in fade-in duration-500 delay-200">
          {message.timestamp.toLocaleTimeString("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  );
}
