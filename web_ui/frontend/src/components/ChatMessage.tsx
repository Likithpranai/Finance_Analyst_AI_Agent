import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
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

  return (
    <div className={`chat-message ${role} ${darkMode ? 'dark' : 'light'}`}>
      <div className="message-avatar">
        {role === 'user' ? (
          <div className="user-avatar">U</div>
        ) : (
          <div className="assistant-avatar">FA</div>
        )}
      </div>
      
      <div className="message-content">
        {role === 'assistant' && (status === 'thinking' || status === 'typing') ? (
          <div className="loading-content">
            <div className="thinking-indicator">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
            <div className="thinking-text">Analyzing financial data...</div>
            
            {tools && tools.length > 0 && (
              <div className="tools-list">
                <div className="tools-header">Using tools:</div>
                <ul>
                  {tools.map((tool, index) => (
                    <li key={index} className="tool-item">
                      <div className="tool-icon">🔍</div>
                      <div className="tool-name">{tool.tool_name}</div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <ReactMarkdown components={components}>
            {content}
          </ReactMarkdown>
        )}
        
        {role === 'assistant' && status === 'error' && (
          <div className="error-indicator">
            ⚠️ Error occurred while processing your request
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
