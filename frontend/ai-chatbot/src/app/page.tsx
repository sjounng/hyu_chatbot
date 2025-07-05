"use client";

import { useState } from "react";
import QuestionForm from "./questionForm";
import SideBar from "./sideBar";
import ChatMessage from "../components/ChatMessage";
import { useChat } from "../hooks/useChat";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const {
    messages,
    messagesEndRef,
    handleSendMessage,
    isLoading,
    stopGeneration,
  } = useChat();

  return (
    <SideBar onSidebarToggle={setIsSidebarOpen}>
      <div className="flex flex-col h-full w-full">
        <div className="z-10 w-full items-center justify-between font-mono text-sm">
          <h1 className="text-4xl text-[#0E4A84] font-bold text-center mb-8">
            HANYANG AI Chatbot
          </h1>

          {/* 채팅 메시지 영역 */}
          <div className="flex-1 overflow-y-auto px-4 pb-32">
            <div className="flex flex-col space-y-4 w-full">
              {messages.map((message, index) => (
                <ChatMessage key={message.id} message={message} index={index} />
              ))}
              {/* 자동 스크롤을 위한 빈 div */}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        {/* 채팅 입력 칸 */}
        <div className="fixed bottom-0 left-0 right-0 p-4 mb-12 z-50">
          <div
            className={`transition-all flex justify-center duration-300 ease-in-out ${
              isSidebarOpen ? "ml-80" : "ml-0"
            }`}
          >
            <QuestionForm
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              onStopGeneration={stopGeneration}
            />
          </div>
        </div>
      </div>
    </SideBar>
  );
}
