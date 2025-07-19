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
        width: isSidebarOpen ? 'calc(100% - 280px)' : '100%'
      }}
    >
      <div className="flex-1 overflow-y-auto p-4">
        <AnimatePresence>
          {messages.length === 0 && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={`welcome-container flex flex-col items-center justify-center h-full ${darkMode ? 'text-white' : 'text-gray-800'}`}
            >
              <h1 className="text-3xl font-bold mb-2">Finance Analyst AI</h1>
              <p className="text-lg mb-8 text-center max-w-2xl">
                Your AI-powered financial analysis assistant. Ask me about stocks, technical indicators, market trends, or financial data.
              </p>
              
              {showExamples && (
                <div className="example-queries grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-3xl">
                  {exampleQueries.map((query, index) => (
                    <motion.button
                      key={index}
                      className={`p-3 rounded-lg text-left ${darkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-100 hover:bg-gray-200'}`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleExampleClick(query)}
                    >
                      {query}
                    </motion.button>
                  ))}
                </div>
              )}
            </motion.div>
          )}
          
          {messages.map((message) => (
            <ChatMessageComponent 
              key={message.id} 
              message={message} 
              darkMode={darkMode} 
            />
          ))}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </div>
      
      <div className={`chat-input-container p-4 border-t ${darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'}`}>
        <ChatInput 
          onSendMessage={handleSendMessage} 
          isProcessing={isProcessing} 
          darkMode={darkMode} 
        />
        <div className={`text-xs mt-2 text-center ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {connectionStatus === 'connected' ? (
            <span className="text-green-500">● Connected</span>
          ) : connectionStatus === 'connecting' ? (
            <span className="text-yellow-500">● Connecting...</span>
          ) : (
            <span className="text-red-500">● Disconnected (using REST fallback)</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
