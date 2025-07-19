import { useState, useCallback } from 'react';
import { ChatMessage, MessageRole, MessageStatus, ToolExecution } from '../types';
import { generateId } from '../utils';

/**
 * Custom hook for managing chat messages
 */
const useChatMessages = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  /**
   * Add a new message to the chat
   */
  const addMessage = useCallback((
    role: MessageRole,
    content: string,
    status: MessageStatus = 'complete',
    toolExecutions?: ToolExecution[]
  ) => {
    const newMessage: ChatMessage = {
      id: generateId(),
      role,
      content,
      timestamp: Date.now(),
      status,
      toolExecutions
    };

    setMessages(prevMessages => [...prevMessages, newMessage]);
    return newMessage.id;
  }, []);

  /**
   * Update an existing message by ID
   */
  const updateMessage = useCallback((
    messageId: string,
    updates: Partial<ChatMessage>
  ) => {
    setMessages(prevMessages => 
      prevMessages.map(msg => 
        msg.id === messageId ? { ...msg, ...updates } : msg
      )
    );
  }, []);

  /**
   * Update the status of a message
   */
  const updateMessageStatus = useCallback((
    messageId: string,
    status: MessageStatus
  ) => {
    updateMessage(messageId, { status });
  }, [updateMessage]);

  /**
   * Add or update tool execution for a message
   */
  const updateToolExecution = useCallback((
    messageId: string,
    toolExecution: ToolExecution
  ) => {
    setMessages(prevMessages => 
      prevMessages.map(msg => {
        if (msg.id !== messageId) return msg;
        
        // Find if this tool already exists in the message
        const existingToolExecutions = msg.toolExecutions || [];
        const toolIndex = existingToolExecutions.findIndex((t: ToolExecution) => 
          t.tool_name === toolExecution.tool_name && 
          JSON.stringify(t.tool_input) === JSON.stringify(toolExecution.tool_input)
        );
        
        let updatedToolExecutions;
        if (toolIndex >= 0) {
          // Update existing tool execution
          updatedToolExecutions = [...existingToolExecutions];
          updatedToolExecutions[toolIndex] = toolExecution;
        } else {
          // Add new tool execution
          updatedToolExecutions = [...existingToolExecutions, toolExecution];
        }
        
        return {
          ...msg,
          toolExecutions: updatedToolExecutions
        };
      })
    );
  }, []);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    addMessage,
    updateMessage,
    updateMessageStatus,
    updateToolExecution,
    clearMessages
  };
};

export default useChatMessages;
