import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ChatMessageComponent from './ChatMessage';
import ChatInput from './ChatInput';
import { useAppContext } from '../context/AppContext';
import useWebSocket from '../hooks/useWebSocket';
import { ChatMessage, WebSocketMessage } from '../types';
import '../styles/ChatInterface.css';

interface ChatInterfaceProps {
  isSidebarOpen: boolean;
  darkMode: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ isSidebarOpen, darkMode }) => {
  const { 
    messages, 
    addMessage, 
    updateMessage, 
    updateMessageStatus, 
    updateToolExecution,
    clearMessages
  } = useAppContext();
  
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showExamples, setShowExamples] = useState(messages.length === 0);
  
  // Generate a unique client ID for WebSocket connection
  const clientId = useRef(`client-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`);
  
  // Setup WebSocket connection
  const { 
    sendMessage, 
    connectionStatus, 
    lastMessage 
  } = useWebSocket({
    url: `ws://localhost:8000/ws/${clientId.current}`,
    onOpen: () => console.log('WebSocket connected'),
    onClose: () => console.log('WebSocket disconnected'),
    onError: (error) => console.error('WebSocket error:', error)
  });
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);
  
  // Handle incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;
    
    try {
      const data = JSON.parse(lastMessage.data) as WebSocketMessage;
      console.log('WebSocket message:', data);
      
      if (!data.message_id) return;
      
      switch (data.type) {
        case 'thinking':
          updateMessageStatus(data.message_id, 'thinking');
          // If tools are provided, update them
          if (data.tools && data.tools.length > 0) {
            data.tools.forEach(tool => {
              updateToolExecution(data.message_id, tool);
            });
          }
          break;
          
        case 'typing':
          if (data.content) {
            updateMessage(data.message_id, {
              content: data.content,
              status: 'typing'
            });
          }
          break;
          
        case 'partial':
          if (data.content) {
            updateMessage(data.message_id, {
              content: data.content,
              status: 'typing'
            });
          }
          break;
          
        case 'tool_execution':
          if (data.tool_execution) {
            updateToolExecution(data.message_id, data.tool_execution);
          }
          break;
          
        case 'complete':
          if (data.content) {
            updateMessage(data.message_id, {
              content: data.content,
              status: 'complete'
            });
            setIsProcessing(false);
          }
          break;
          
        case 'error':
          updateMessage(data.message_id, {
            content: data.content || 'An error occurred',
            status: 'error'
          });
          setIsProcessing(false);
          break;
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, [lastMessage, updateMessage, updateMessageStatus, updateToolExecution]);
  
  // Handle sending a message
  const handleSendMessage = (content: string) => {
    if (!content.trim() || isProcessing) return;
    
    // Add user message
    const userMessageId = addMessage('user', content);
    
    // Add assistant message with loading status
    const assistantMessageId = addMessage('assistant', '', 'thinking');
    
    // Set processing state
    setIsProcessing(true);
    setShowExamples(false);
    
    // Send message to WebSocket
    if (connectionStatus === 'connected') {
      // Add timestamp to message ID to ensure uniqueness
      const messageId = `${assistantMessageId}-${Date.now()}`;
      
      sendMessage(JSON.stringify({
        query: content,
        message_id: messageId
      }));
      
      // Update the message ID in the state to match what we sent
      updateMessage(assistantMessageId, { id: messageId });
    } else {
      // Fallback to REST API if WebSocket is not connected
      fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: content,
          message_id: assistantMessageId
        })
      })
      .then(response => response.json())
      .then(data => {
        // Format the response as markdown if it's a complex object
        let formattedResponse = data.response;
        if (typeof formattedResponse === 'object') {
          formattedResponse = '```json\n' + JSON.stringify(formattedResponse, null, 2) + '\n```';
        }
        
        updateMessage(assistantMessageId, {
          content: formattedResponse,
          status: 'complete'
        });
        setIsProcessing(false);
      })
      .catch(error => {
        console.error('Error sending message:', error);
        updateMessage(assistantMessageId, {
          content: 'Error: Failed to get response from server.',
          status: 'error'
        });
        setIsProcessing(false);
      });
    }
  };
  
  // Handle example query selection
  const handleExampleClick = (query: string) => {
    handleSendMessage(query);
  };

  // Example financial queries based on the Finance Analyst AI Agent capabilities
  const exampleQueries = [
    "What's the current price of AAPL?",
    "Calculate RSI and MACD for Tesla",
    "Compare Microsoft and Google fundamentals",
    "Show me technical indicators for Bitcoin",
    "Analyze P/E ratio for Amazon"
  ];

  return (
    <div 
      className={`chat-interface flex flex-col h-full transition-all duration-300 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}
      style={{ 
        marginLeft: isSidebarOpen ? '280px' : '0',
        width: isSidebarOpen ? 'calc(100% - 280px)' : '100%',
        backgroundImage: darkMode ? 
          'radial-gradient(circle at 10% 20%, rgba(21, 25, 40, 0.8) 0%, rgba(10, 16, 27, 0.6) 90%)' : 
          'radial-gradient(circle at 10% 20%, rgba(240, 245, 255, 0.8) 0%, rgba(250, 252, 255, 0.6) 90%)'
      }}
    >
      <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 scroll-smooth">
        <AnimatePresence>
          {messages.length === 0 && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className={`welcome-container flex flex-col items-center justify-center h-full ${darkMode ? 'text-white' : 'text-gray-800'}`}
            >
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="mb-6"
              >
                <div className={`finance-logo-container p-4 rounded-full ${darkMode ? 'bg-gray-800' : 'bg-blue-50'}`}>
                  <svg width="80" height="80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 9H7V20H3V9Z" fill={darkMode ? '#60A5FA' : '#3B82F6'} />
                    <path d="M10 4H14V20H10V4Z" fill={darkMode ? '#93C5FD' : '#2563EB'} />
                    <path d="M17 13H21V20H17V13Z" fill={darkMode ? '#BFDBFE' : '#1D4ED8'} />
                    <path d="M21 7L16 2L11 7L6 2L1 7" stroke={darkMode ? '#F0F9FF' : '#1E3A8A'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              </motion.div>
              
              <motion.h1 
                className="text-4xl font-bold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-500"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.5 }}
              >
                Finance Analyst AI
              </motion.h1>
              
              <motion.p 
                className="text-lg mb-10 text-center max-w-2xl leading-relaxed"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6, duration: 0.5 }}
              >
                Your AI-powered financial analysis assistant. Ask me about stocks, technical indicators, 
                market trends, portfolio optimization, or financial data analysis.
              </motion.p>
              
              {showExamples && (
                <motion.div 
                  className="example-queries grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8, duration: 0.5 }}
                >
                  {exampleQueries.map((query, index) => (
                    <motion.button
                      key={index}
                      className={`p-4 rounded-xl text-left shadow-sm border ${darkMode ? 
                        'bg-gray-800 hover:bg-gray-700 border-gray-700' : 
                        'bg-white hover:bg-gray-50 border-gray-100'}`}
                      whileHover={{ scale: 1.02, boxShadow: darkMode ? 
                        '0 4px 12px rgba(0, 0, 0, 0.3)' : 
                        '0 4px 12px rgba(0, 0, 0, 0.1)' 
                      }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleExampleClick(query)}
                      transition={{ duration: 0.2 }}
                    >
                      {query}
                    </motion.button>
                  ))}
                </motion.div>
              )}
            </motion.div>
          )}
          
          <div className="messages-container space-y-6 pb-2">
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index === messages.length - 1 ? 0 : 0 }}
              >
                <ChatMessageComponent 
                  message={message} 
                  darkMode={darkMode} 
                />
              </motion.div>
            ))}
          </div>
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>
      
      <div className={`chat-input-container p-4 md:p-6 border-t ${darkMode ? 'border-gray-700 bg-gray-900/90 backdrop-blur-sm' : 'border-gray-200 bg-white/90 backdrop-blur-sm'}`}>
        <div className="max-w-5xl mx-auto w-full">
          <ChatInput 
            onSendMessage={handleSendMessage} 
            isProcessing={isProcessing} 
            darkMode={darkMode} 
          />
          <div className={`text-xs mt-2 text-center ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            <div className="flex items-center justify-center gap-2">
              {connectionStatus === 'connected' ? (
                <>
                  <span className="inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                  <span className="text-green-500">Connected to Finance Analyst AI</span>
                </>
              ) : connectionStatus === 'connecting' ? (
                <>
                  <span className="inline-block w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></span>
                  <span className="text-yellow-500">Connecting to server...</span>
                </>
              ) : (
                <>
                  <span className="inline-block w-2 h-2 rounded-full bg-red-500"></span>
                  <span className="text-red-500">Disconnected (using REST fallback)</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
