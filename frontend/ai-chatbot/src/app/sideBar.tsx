"use client";

import { useState } from "react";
import { Button } from "./button";
import { Menu, X, MessageSquare, Clock } from "lucide-react";
import Link from "next/link";

interface ChatHistory {
  id: string;
  title: string;
  timestamp: Date;
}

interface SideBarProps {
  children: React.ReactNode;
  onSidebarToggle?: (isOpen: boolean) => void;
}

export default function SideBar({ children, onSidebarToggle }: SideBarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([
    {
      id: "1",
      title: "파이썬으로 웹 스크래핑하는 방법",
      timestamp: new Date(),
    },
    { id: "2", title: "React 컴포넌트 최적화", timestamp: new Date() },
    { id: "3", title: "데이터베이스 설계 원칙", timestamp: new Date() },
  ]);

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("ko-KR", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const handleToggle = () => {
    const newIsOpen = !isOpen;
    setIsOpen(newIsOpen);
    onSidebarToggle?.(newIsOpen);
  };

  return (
    <div className="flex h-screen">
      {/* 사이드바 */}
      <div
        className={`
          h-full bg-white dark:bg-stone-800 border-r border-[#E5E5E5] shadow-lg transition-all duration-300 ease-in-out
          ${isOpen ? "w-80" : "w-0"}
          overflow-hidden flex-shrink-0
        `}
      >
        {/* 메뉴 항목 */}
        <nav className="flex flex-col items-start px-6 py-4 space-y-4 border-b border-[#E5E5E5] min-w-80">
          {["Home", "Info"].map((label) => {
            const href = label === "Home" ? "/" : `/${label.toLowerCase()}`;
            return (
              <Link
                key={label}
                href={href}
                className="w-full py-2 text-lg font-medium hover:underline"
                onClick={() => setIsOpen(false)}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* 채팅 히스토리 */}
        <div className="px-6 py-4 min-w-80">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <MessageSquare size={20} />
            대화 기록
          </h2>
          <div className="space-y-3">
            {chatHistory.map((chat) => (
              <button
                key={chat.id}
                className="w-full text-left p-3 rounded-lg hover:bg-stone-100 transition-colors"
                onClick={() => {
                  // TODO: 채팅 기록 불러오기
                  setIsOpen(false);
                }}
              >
                <div className="font-medium text-sm mb-1 line-clamp-2">
                  {chat.title}
                </div>
                <div className="text-xs text-stone-500 flex items-center gap-1">
                  <Clock size={12} />
                  {formatDate(chat.timestamp)}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 영역 */}
      <div className="flex-1 flex flex-col min-w-0 h-full">
        {/* 토글 버튼 */}
        <div className="p-4 flex-shrink-0">
          <Button
            className="p-2 rounded-lg focus:outline-none bg-white hover:bg-stone-100 text-black"
            onClick={handleToggle}
            aria-label={isOpen ? "메뉴 닫기" : "메뉴 열기"}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </Button>
        </div>

        {/* 메인 컨텐츠 */}
        <div className="flex-1 p-4 min-w-0 overflow-auto">{children}</div>
      </div>
    </div>
  );
}
