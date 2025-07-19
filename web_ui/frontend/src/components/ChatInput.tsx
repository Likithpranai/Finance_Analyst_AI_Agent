import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiSend, FiMic, FiRefreshCw, FiChevronDown } from 'react-icons/fi';
import '../styles/ChatInput.css';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
  darkMode: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isProcessing, darkMode }) => {
  const [message, setMessage] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Financial query suggestions
  const suggestions = [
    "What's the current price of AAPL?",
    "Calculate RSI and MACD for Tesla",
    "Compare Microsoft and Google fundamentals",
    "Show me technical indicators for Bitcoin",
    "Analyze P/E ratio for Amazon",
    "What's the market trend for S&P 500?",
    "Show me the latest economic indicators"
  ];

  // Auto-resize textarea as content grows
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isProcessing) {
      onSendMessage(message);
      setMessage('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={`chat-input-container ${darkMode ? 'dark' : 'light'}`}>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="textarea-wrapper">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isProcessing ? "Processing your request..." : "Ask about stocks, technical analysis, market trends, or financial data..."}
            disabled={isProcessing}
            rows={1}
            className={`chat-input-textarea ${isProcessing ? 'disabled' : ''}`}
          />
          
          <motion.button
            type="button"
            className="suggestions-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowSuggestions(!showSuggestions)}
            disabled={isProcessing}
            aria-label="Show suggestions"
          >
            <FiChevronDown />
          </motion.button>
        </div>
        
        {showSuggestions && (
          <motion.div 
            className="suggestions-dropdown"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {suggestions.map((suggestion, index) => (
              <motion.button
                key={index}
                type="button"
                className="suggestion-item"
                whileHover={{ backgroundColor: darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' }}
                onClick={() => {
                  setMessage(suggestion);
                  setShowSuggestions(false);
                  if (textareaRef.current) {
                    textareaRef.current.focus();
                  }
                }}
              >
                {suggestion}
              </motion.button>
            ))}
          </motion.div>
        )}
        
        <div className="chat-input-buttons">
          {isProcessing ? (
            <motion.button
              type="button"
              className="processing-indicator"
              initial={{ rotate: 0 }}
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              disabled
            >
              <FiRefreshCw />
            </motion.button>
          ) : (
            <>
              <motion.button
                type="button"
                className="mic-button"
                whileHover={{ scale: 1.1, backgroundColor: darkMode ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)' }}
                whileTap={{ scale: 0.95 }}
                disabled={isProcessing}
                aria-label="Voice input"
              >
                <FiMic />
              </motion.button>
              
              <motion.button
                type="submit"
                className={`send-button ${message.trim() ? 'active' : ''}`}
                whileHover={{ scale: 1.05, backgroundColor: message.trim() ? (darkMode ? 'var(--dark-primary-hover)' : 'var(--light-primary-hover)') : '' }}
                whileTap={{ scale: 0.95 }}
                disabled={!message.trim() || isProcessing}
                aria-label="Send message"
              >
                <FiSend />
              </motion.button>
            </>
          )}
        </div>
      </form>
      
      <div className="chat-input-footer">
        <p className="disclaimer">
          Finance Analyst AI provides analysis based on historical data and may not reflect current market conditions. Not financial advice.
        </p>
      </div>
    </div>
  );
};

export default ChatInput;
