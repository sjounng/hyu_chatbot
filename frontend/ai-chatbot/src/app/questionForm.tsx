"use client";

import { useState, KeyboardEvent, useRef, useEffect } from "react";

interface QuestionFormProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  onStopGeneration?: () => void;
}

export default function QuestionForm({
  onSendMessage,
  isLoading = false,
  onStopGeneration,
}: QuestionFormProps) {
  const [question, setQuestion] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [question]);

  const sendMessage = () => {
    if (!question.trim() || isComposing) return;

    const messageToSend = question.trim();
    setQuestion("");
    onSendMessage(messageToSend);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !isComposing) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleCompositionStart = () => {
    setIsComposing(true);
  };

  const handleCompositionEnd = () => {
    setIsComposing(false);
  };

  const handleButtonClick = (e: React.MouseEvent) => {
    e.preventDefault();
    if (isLoading && onStopGeneration) {
      onStopGeneration();
    } else {
      sendMessage();
    }
  };

  return (
    <div className="w-full max-w-5xl">
      <div className="flex w-full cursor-text flex-col items-center justify-center rounded-[28px] border border-[#E5E5E5] bg-white shadow-xl">
        <div className="relative flex w-full items-end px-2.5 py-2.5">
          <div className="relative flex w-full flex-auto flex-col">
            <div className="relative mx-2.5">
              <div className="relative flex-auto bg-transparent pt-1.5">
                <div className="flex flex-col-reverse justify-end">
                  <div className="flex min-h-12 items-end">
                    <div className="max-w-full min-w-0 flex-1">
                      <div className="text-[#0E4A84] font- max-h-52 overflow-auto min-h-12">
                        <textarea
                          ref={textareaRef}
                          value={question}
                          onChange={(e) => setQuestion(e.target.value)}
                          onKeyDown={handleKeyDown}
                          onCompositionStart={handleCompositionStart}
                          onCompositionEnd={handleCompositionEnd}
                          className="block text-lg w-full resize-none border-0 bg-transparent px-0 py-2 focus:outline-none focus:ring-0 placeholder:text-[#898C8E] placeholder:text-lg"
                          placeholder="한양대학교에 관련된 궁금한 점을 물어봐주세요!"
                          rows={1}
                          style={{ minHeight: "48px", maxHeight: "200px" }}
                          disabled={isLoading}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={handleButtonClick}
                className={`flex h-9 items-center justify-center rounded-full transition-colors w-9 absolute right-2.5 bottom-2.5 ${
                  isLoading
                    ? "bg-black text-white hover:bg-black"
                    : "bg-black text-white hover:opacity-70 disabled:bg-[#D7D7D7] disabled:text-[#f4f4f4] disabled:hover:opacity-100"
                }`}
                disabled={(!question.trim() || isComposing) && !isLoading}
              >
                {isLoading ? (
                  // 중지 버튼 (네모 모양)
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="icon-md"
                  >
                    <rect
                      x="6"
                      y="6"
                      width="12"
                      height="12"
                      fill="currentColor"
                    />
                  </svg>
                ) : (
                  // 전송 버튼 (화살표 모양)
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 18 18"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="icon-md"
                  >
                    <path
                      d="M7.99992 14.9993V5.41334L4.70696 8.70631C4.31643 9.09683 3.68342 9.09683 3.29289 8.70631C2.90237 8.31578 2.90237 7.68277 3.29289 7.29225L8.29289 2.29225L8.36906 2.22389C8.76184 1.90354 9.34084 1.92613 9.70696 2.29225L14.707 7.29225L14.7753 7.36842C15.0957 7.76119 15.0731 8.34019 14.707 8.70631C14.3408 9.07242 13.7618 9.09502 13.3691 8.77467L13.2929 8.70631L9.99992 5.41334V14.9993C9.99992 15.5516 9.55221 15.9993 8.99992 15.9993C8.44764 15.9993 7.99993 15.5516 7.99992 14.9993Z"
                      fill="currentColor"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
        <div className="absolute end-2.5 bottom-2.5 flex items-center gap-2"></div>
      </div>
    </div>
  );
}
