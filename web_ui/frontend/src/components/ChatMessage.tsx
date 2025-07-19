import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion } from 'framer-motion';
import { ChatMessage as ChatMessageType, ToolExecution } from '../types';
import '../styles/ChatMessage.css';

interface MessageProps {
  message: ChatMessageType;
  darkMode: boolean;
}

const ChatMessage: React.FC<MessageProps> = ({ message, darkMode }) => {
  const { role, content, status, tools } = message;
  
  // Custom renderer for code blocks
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={darkMode ? vscDarkPlus : vs}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  };

  // Animation variants for message appearance
  const messageVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }
  };
  
  // Animation variants for thinking dots
  const dotVariants = {
    initial: { opacity: 0.3, y: 0 },
    animate: { opacity: 1, y: -5, transition: { duration: 0.5, repeat: Infinity, repeatType: "reverse" as const } }
  };
  
  // Animation variants for tool execution
  const toolVariants = {
    hidden: { opacity: 0, height: 0 },
    visible: { opacity: 1, height: 'auto', transition: { duration: 0.3 } }
  };

  return (
    <motion.div 
      className={`chat-message ${role} ${darkMode ? 'dark' : 'light'}`}
      initial="hidden"
      animate="visible"
      variants={messageVariants}
    >
      <div className="message-avatar">
        {role === 'user' ? (
          <div className="user-avatar">{darkMode ? '👤' : 'U'}</div>
        ) : (
          <div className="assistant-avatar">{darkMode ? '🤖' : 'FA'}</div>
        )}
      </div>
      
      <div className="message-content">
        {role === 'assistant' && (status === 'thinking' || status === 'typing') ? (
          <div className="loading-content">
            <div className="thinking-indicator">
              <motion.span className="dot" variants={dotVariants} initial="initial" animate="animate" transition={{ delay: 0 }}></motion.span>
              <motion.span className="dot" variants={dotVariants} initial="initial" animate="animate" transition={{ delay: 0.15 }}></motion.span>
              <motion.span className="dot" variants={dotVariants} initial="initial" animate="animate" transition={{ delay: 0.3 }}></motion.span>
            </div>
            <div className="thinking-text">
              {status === 'thinking' ? 'Analyzing financial data...' : 'Generating response...'}
            </div>
            
            {tools && tools.length > 0 && (
              <motion.div 
                className="tools-list"
                initial="hidden"
                animate="visible"
                variants={toolVariants}
              >
                <div className="tools-header">Using tools:</div>
                <ul>
                  {tools.map((tool, index) => (
                    <motion.li 
                      key={index} 
                      className={`tool-item ${tool.status || 'running'}`}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="tool-icon">
                        {tool.status === 'completed' ? '✅' : 
                         tool.status === 'error' ? '❌' : '🔍'}
                      </div>
                      <div className="tool-name">{tool.tool_name}</div>
                      {tool.status === 'started' && (
                        <div className="tool-status-indicator">
                          <div className="tool-status-spinner"></div>
                        </div>
                      )}
                    </motion.li>
                  ))}
                </ul>
              </motion.div>
            )}
          </div>
        ) : (
          <div className="markdown-content">
            <ReactMarkdown components={components}>
              {content}
            </ReactMarkdown>
          </div>
        )}
        
        {role === 'assistant' && status === 'error' && (
          <motion.div 
            className="error-indicator"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            ⚠️ Error occurred while processing your request
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default ChatMessage;
