// Message types
export type MessageRole = 'user' | 'assistant' | 'system';
export type MessageStatus = 'complete' | 'thinking' | 'typing' | 'error';

export interface ToolExecution {
  tool: string;
  input: any;
  output?: any;
  status: 'started' | 'completed' | 'error';
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  status: MessageStatus;
  toolExecutions?: ToolExecution[];
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'thinking' | 'typing' | 'complete' | 'error' | 'tool_execution';
  message_id: string;
  content?: string;
  tool_execution?: ToolExecution;
}

// API response types
export interface QueryResponse {
  message_id: string;
  response: string;
  status: 'complete' | 'error';
}

// Theme types
export type ThemeMode = 'dark' | 'light';

// Conversation types
export interface Conversation {
  id: string;
  title: string;
  timestamp: number;
  messages: ChatMessage[];
}

// Example query types
export interface ExampleQuery {
  text: string;
  category: string;
}
