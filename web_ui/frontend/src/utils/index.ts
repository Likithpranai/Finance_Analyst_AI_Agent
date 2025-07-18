import { ChatMessage, ThemeMode } from '../types';

// Generate a unique ID
export const generateId = (): string => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

// Format a timestamp
export const formatTimestamp = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// Format a date for conversation history
export const formatDate = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
};

// Toggle theme mode
export const toggleTheme = (currentTheme: ThemeMode): ThemeMode => {
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  // Update HTML class
  document.documentElement.classList.remove(currentTheme);
  document.documentElement.classList.add(newTheme);
  
  // Store preference
  localStorage.setItem('theme', newTheme);
  
  return newTheme;
};

// Get stored theme or default to dark
export const getStoredTheme = (): ThemeMode => {
  const storedTheme = localStorage.getItem('theme') as ThemeMode | null;
  return storedTheme || 'dark';
};

// Initialize theme based on stored preference
export const initializeTheme = (): ThemeMode => {
  const theme = getStoredTheme();
  document.documentElement.classList.add(theme);
  return theme;
};

// Auto resize textarea
export const autoResizeTextarea = (element: HTMLTextAreaElement): void => {
  element.style.height = 'auto';
  element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
};

// Get example queries for finance analysis
export const getExampleQueries = () => [
  {
    text: "Analyze AAPL's technical indicators",
    category: "Technical Analysis"
  },
  {
    text: "Compare MSFT and GOOGL fundamentals",
    category: "Fundamental Analysis"
  },
  {
    text: "Show me Bitcoin's price trend",
    category: "Cryptocurrency"
  },
  {
    text: "Optimize a portfolio of AAPL, TSLA, and NVDA",
    category: "Portfolio Management"
  }
];

// Store conversations in local storage
export const storeConversation = (messages: ChatMessage[]): void => {
  if (messages.length === 0) return;
  
  const existingConversations = JSON.parse(localStorage.getItem('conversations') || '[]');
  
  // Create a new conversation object
  const conversation = {
    id: generateId(),
    title: extractConversationTitle(messages),
    timestamp: Date.now(),
    messages
  };
  
  // Add to existing conversations and store
  localStorage.setItem('conversations', JSON.stringify([conversation, ...existingConversations]));
};

// Extract a title from conversation messages
export const extractConversationTitle = (messages: ChatMessage[]): string => {
  const firstUserMessage = messages.find(msg => msg.role === 'user');
  if (firstUserMessage) {
    // Extract first 3-5 words or 30 characters
    const text = firstUserMessage.content;
    const words = text.split(' ');
    if (words.length <= 5) return text;
    return words.slice(0, 4).join(' ') + '...';
  }
  return 'New conversation';
};

// Get stored conversations
export const getStoredConversations = () => {
  return JSON.parse(localStorage.getItem('conversations') || '[]');
};

// Format financial data for display
export const formatFinancialValue = (value: number): string => {
  if (value >= 1000000000) {
    return `$${(value / 1000000000).toFixed(2)}B`;
  } else if (value >= 1000000) {
    return `$${(value / 1000000).toFixed(2)}M`;
  } else if (value >= 1000) {
    return `$${(value / 1000).toFixed(2)}K`;
  } else {
    return `$${value.toFixed(2)}`;
  }
};
