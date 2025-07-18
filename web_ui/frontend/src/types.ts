// Define types for the Finance Analyst AI Agent web UI

// Message role types
export type MessageRole = 'user' | 'assistant' | 'system';

// Message status types
export type MessageStatus = 'thinking' | 'typing' | 'complete' | 'error';

// Tool execution data
export interface ToolExecution {
  tool_name: string;
  tool_input?: any;
  tool_output?: any;
  status: 'started' | 'completed' | 'error';
}

// Chat message structure
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  status?: MessageStatus;
  tools?: ToolExecution[];
}

// WebSocket message types
export type WebSocketMessageType = 'thinking' | 'typing' | 'tool_execution' | 'complete' | 'error';

// WebSocket message structure
export interface WebSocketMessage {
  type: WebSocketMessageType;
  message_id: string;
  content?: string;
  tool_execution?: ToolExecution;
}

// Conversation history
export interface Conversation {
  id: string;
  title: string;
  timestamp: Date;
  messages: ChatMessage[];
}
