import React, { createContext, useContext, ReactNode } from 'react';
import { ThemeMode, ChatMessage } from '../types';
import useThemeMode from '../hooks/useThemeMode';
import useChatMessages from '../hooks/useChatMessages';

// Define the context type
interface AppContextType {
  // Theme
  theme: ThemeMode;
  toggleTheme: () => void;
  
  // Chat messages
  messages: ChatMessage[];
  addMessage: (role: 'user' | 'assistant' | 'system', content: string, status?: 'complete' | 'thinking' | 'typing' | 'error') => string;
  updateMessage: (messageId: string, updates: Partial<ChatMessage>) => void;
  updateMessageStatus: (messageId: string, status: 'complete' | 'thinking' | 'typing' | 'error') => void;
  updateToolExecution: (messageId: string, toolExecution: any) => void;
  clearMessages: () => void;
  sendMessage: (content: string) => void;
  isProcessing: boolean;
  
  // UI state
  isSidebarOpen: boolean;
  setIsSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>;
}

// Create the context with a default value
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, toggleTheme] = useThemeMode();
  const { 
    messages, 
    addMessage, 
    updateMessage, 
    updateMessageStatus,
    updateToolExecution,
    clearMessages 
  } = useChatMessages();
  
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(false);
  const [isProcessing, setIsProcessing] = React.useState(false);
  
  // Send message function that handles the chat flow
  const sendMessage = (content: string) => {
    if (!content.trim() || isProcessing) return;
    
    // Add user message
    const userMessageId = addMessage('user', content);
    
    // Set processing state
    setIsProcessing(true);
    
    // Add assistant thinking message
    const assistantMessageId = addMessage('assistant', '', 'thinking');
    
    // Simulate API call - in a real app, this would be an actual API call
    setTimeout(() => {
      // Update assistant message with response
      updateMessage(assistantMessageId, {
        content: `This is a simulated response to: "${content}". In a real implementation, this would call the Finance Analyst AI backend.`,
        status: 'complete'
      });
      
      setIsProcessing(false);
    }, 1500);
  };
  
  // Create the value object
  const value: AppContextType = {
    theme,
    toggleTheme,
    messages,
    addMessage,
    updateMessage,
    updateMessageStatus,
    updateToolExecution,
    clearMessages,
    sendMessage,
    isProcessing,
    isSidebarOpen,
    setIsSidebarOpen
  };
  
  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// Custom hook to use the context
export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

export default AppContext;
