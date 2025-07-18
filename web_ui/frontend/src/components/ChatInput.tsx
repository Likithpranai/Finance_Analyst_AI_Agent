import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiSend, FiMic } from 'react-icons/fi';
import '../styles/ChatInput.css';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
  darkMode: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isProcessing, darkMode }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about stocks, technical analysis, or financial markets..."
          disabled={isProcessing}
          rows={1}
          className={`chat-input-textarea ${isProcessing ? 'disabled' : ''}`}
        />
        
        <div className="chat-input-buttons">
          <motion.button
            type="button"
            className="mic-button"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            disabled={isProcessing}
          >
            <FiMic />
          </motion.button>
          
          <motion.button
            type="submit"
            className={`send-button ${message.trim() ? 'active' : ''}`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            disabled={!message.trim() || isProcessing}
          >
            <FiSend />
          </motion.button>
        </div>
      </form>
      
      <div className="chat-input-footer">
        <p className="disclaimer">
          Finance Analyst AI may produce inaccurate information about stocks, markets, or financial data.
        </p>
      </div>
    </div>
  );
};

export default ChatInput;
