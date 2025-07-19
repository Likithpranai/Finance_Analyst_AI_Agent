import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppContext } from '../context/AppContext';
import { ChatMessage as ChatMessageType } from '../types';
import '../styles/GrokChatUI.css';

interface GrokChatUIProps {
  darkMode: boolean;
  toggleDarkMode: () => void;
}

const GrokChatUI: React.FC<GrokChatUIProps> = ({ darkMode, toggleDarkMode }) => {
  const { messages, sendMessage, clearMessages, isProcessing } = useAppContext();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (inputValue.trim() && !isProcessing) {
      sendMessage(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const thinkHarder = () => {
    if (messages.length > 0) {
      // Get the last user message and resend it with a "think harder" flag
      const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
      if (lastUserMessage && lastUserMessage.content) {
        sendMessage(`Think harder about this: ${lastUserMessage.content}`);
      }
    }
  };

  return (
    <div className="grok-chat-container">
      <div className="grok-top-bar">
        <div className="grok-top-bar-left">
          <div className="grok-icon grok-logo">📊</div>
          <div className="grok-icon" title="Technical Analysis">📈</div>
          <div className="grok-icon" title="Fundamental Analysis">📊</div>
          <div className="grok-icon" title="Portfolio Management">💼</div>
          <div className="grok-icon" title="Market Data">🌐</div>
          <div className="grok-icon" title="Predictions">🔮</div>
        </div>
        <div className="grok-top-bar-right">
          <div className="grok-icon" onClick={toggleDarkMode} title={darkMode ? 'Light Mode' : 'Dark Mode'}>{darkMode ? '☀️' : '🌙'}</div>
          <div className="grok-icon" title="Settings">⚙️</div>
          <div className="grok-icon" title="Help">❓</div>
          <div className="grok-icon grok-share" title="Share Analysis">↑ Share</div>
        </div>
      </div>
      <div className="grok-chat-messages">
        <AnimatePresence>
          {messages.length === 0 ? (
            <motion.div 
              className="grok-welcome-message"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h1>Finance Analyst AI</h1>
              <p>Your advanced ReAct-powered financial analysis assistant</p>
              <div className="grok-categories">
                <motion.div className="grok-category" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <div className="grok-category-icon">📈</div>
                  <div className="grok-category-title">Technical Analysis</div>
                </motion.div>
                <motion.div className="grok-category" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <div className="grok-category-icon">📊</div>
                  <div className="grok-category-title">Fundamental Analysis</div>
                </motion.div>
                <motion.div className="grok-category" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <div className="grok-category-icon">💼</div>
                  <div className="grok-category-title">Portfolio Management</div>
                </motion.div>
                <motion.div className="grok-category" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <div className="grok-category-icon">🌐</div>
                  <div className="grok-category-title">Market Data</div>
                </motion.div>
              </div>
              <div className="grok-example-queries">
                <motion.div 
                  className="grok-example" 
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setInputValue("Calculate RSI, MACD and Bollinger Bands for AAPL")}
                >
                  Calculate RSI, MACD and Bollinger Bands for AAPL
                </motion.div>
                <motion.div 
                  className="grok-example" 
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setInputValue("Compare P/E ratios for TSLA, NVDA, and AMD")}
                >
                  Compare P/E ratios for TSLA, NVDA, and AMD
                </motion.div>
                <motion.div 
                  className="grok-example" 
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setInputValue("Analyze Bitcoin price trends for the last month")}
                >
                  Analyze Bitcoin price trends for the last month
                </motion.div>
                <motion.div 
                  className="grok-example" 
                  whileHover={{ scale: 1.02, backgroundColor: 'rgba(59, 130, 246, 0.1)' }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setInputValue("Optimize a portfolio with AAPL, MSFT, AMZN, and GOOGL")}
                >
                  Optimize a portfolio with AAPL, MSFT, AMZN, and GOOGL
                </motion.div>
              </div>
            </motion.div>
          ) : (
            messages.map((message, index) => (
              <motion.div 
                key={index}
                className={`grok-message ${message.role === 'user' ? 'grok-user-message' : 'grok-ai-message'}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                {message.content}
                {message.role === 'assistant' && (
                  <div className="grok-reactions">↩️ 🔄 ❤️ 👍 👎 ⋯ {(Math.random() * 2).toFixed(1)}s</div>
                )}
              </motion.div>
            ))
          )}
          <div ref={messagesEndRef} />
        </AnimatePresence>
      </div>
      <div className="grok-input-bar">
        <motion.button 
          className="grok-think-harder"
          onClick={thinkHarder}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          disabled={isProcessing || messages.length === 0}
        >
          Deep Analysis
        </motion.button>
        <div className="grok-input-container">
          <div className="grok-icon">🔍</div>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about stocks, technical analysis, or financial data..."
            disabled={isProcessing}
            className="grok-input"
          />
          {inputValue.trim() ? (
            <motion.div 
              className="grok-send-icon"
              onClick={handleSendMessage}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              ↑
            </motion.div>
          ) : (
            <div className="grok-mic-icon">📊</div>
          )}
        </div>
        <div className="grok-indicator">
          <span className="grok-model-name">Finance Analyst</span> 
          <div className="grok-version-badge">ReAct</div>
        </div>
      </div>
    </div>
  );
};

export default GrokChatUI;
