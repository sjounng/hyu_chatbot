export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  isTyping?: boolean;
  isTypingResponse?: boolean;
  displayedText?: string;
}
