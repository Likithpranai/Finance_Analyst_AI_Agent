import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Helper function to format financial analysis responses
const formatResponse = (text) => {
  if (!text) return '';
  
  // Format numbers and percentages
  let formatted = text
    .replace(/(\$\d+(\.\d+)?)/g, '<span class="highlight-number">$1</span>')
    .replace(/(\d+\.\d+\s*%)/g, '<span class="highlight-percent">$1</span>')
    .replace(/(\+\d+\.\d+\s*%)/g, '<span class="highlight-positive">$1</span>')
    .replace(/(-\d+\.\d+\s*%)/g, '<span class="highlight-negative">$1</span>');
    
  // Format specific financial metrics
  formatted = formatted
    .replace(/(RSI\(\d+\)[^:]+:\s*)(\d+\.\d+)/g, '$1<span class="highlight-number">$2</span>')
    .replace(/(MACD[^:]+:\s*)([-]?\d+\.\d+)/g, '$1<span class="highlight-number">$2</span>')
    .replace(/(P\/E[^:]+:\s*)(\d+\.\d+)/g, '$1<span class="highlight-number">$2</span>')
    .replace(/(EPS[^:]+:\s*)([\$]?[-]?\d+\.\d+)/g, '$1<span class="highlight-number">$2</span>')
    .replace(/(ROE[^:]+:\s*)(\d+\.\d+%)/g, '$1<span class="highlight-percent">$2</span>');
    
  // Format trend indicators
  formatted = formatted
    .replace(/(Trend:\s*)(Bullish)/gi, '$1<span class="highlight-positive">$2</span>')
    .replace(/(Trend:\s*)(Bearish)/gi, '$1<span class="highlight-negative">$2</span>')
    .replace(/(Rating:\s*)(Strong Buy|Buy)/gi, '$1<span class="highlight-positive">$2</span>')
    .replace(/(Rating:\s*)(Sell|Strong Sell)/gi, '$1<span class="highlight-negative">$2</span>')
    .replace(/(Rating:\s*)(Hold|Neutral)/gi, '$1<span class="highlight-neutral">$2</span>');
  
  // Format sections
  formatted = formatted
    .replace(/^([A-Z][A-Z\s]+):\s*$/gm, '<h3 class="analysis-section">$1</h3>')
    .replace(/^([A-Z][A-Z\s]+)$/gm, '<h3 class="analysis-section">$1</h3>')
    .replace(/^-{3,}$/gm, '<hr class="section-divider">')
    .replace(/^#{3}\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/^#{2}\s+(.+)$/gm, '<h2>$1</h2>')
    .replace(/^#{1}\s+(.+)$/gm, '<h1>$1</h1>');
  
  // Format stock symbols
  formatted = formatted
    .replace(/\b([A-Z]{2,5})\b(?!\<\/span|\<\/h|[a-z])/g, '<span class="stock-symbol">$1</span>');
  
  // Format lists
  formatted = formatted
    .replace(/^\*\s+(.+)$/gm, '<li>$1</li>')
    .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  
  // Format bold text
  formatted = formatted
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Convert consecutive line breaks to paragraphs
  formatted = '<p>' + formatted.replace(/\n{2,}/g, '</p><p>') + '</p>';
  
  // Fix any broken HTML from our replacements
  formatted = formatted
    .replace(/<\/p><h/g, '</p><h')
    .replace(/<\/h\d><p>/g, '</h$1>')
    .replace(/<p><hr/g, '<hr')
    .replace(/<\/li><p>/g, '</li>')
    .replace(/<p><li>/g, '<ul><li>')
    .replace(/<\/li><\/p>/g, '</li></ul>');
  
  return formatted;
};

function App() {
  const [messages, setMessages] = useState([
    { id: 1, text: 'Welcome to the Finance Analyst AI Agent. How can I help you with financial analysis today?', sender: 'agent' }
  ]);
  
  // Example queries to help users get started
  const exampleQueries = [
    // Technical Analysis
    "What's the current price of AAPL and its technical indicators?",
    "Calculate RSI, MACD, and Bollinger Bands for TSLA",
    "Show me a technical analysis of Bitcoin over the last 3 months",
    
    // Fundamental Analysis
    "Analyze Tesla's financial ratios and compare to industry averages",
    "What are the key financial metrics for Amazon?",
    "Show me the income statement and balance sheet for Microsoft",
    
    // Combined Analysis
    "Give me a comprehensive analysis of NVDA with buy/sell recommendation",
    "Compare the performance of MSFT and GOOGL with technical and fundamental factors",
    
    // Market Research
    "What's the latest news affecting Apple stock?",
    "Analyze the recent earnings report for Netflix",
    
    // Portfolio Management
    "How should I optimize a portfolio containing AAPL, MSFT, and GOOGL?",
    "What's the risk assessment for a tech-heavy portfolio?"
  ];
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input })
      });
      
      const data = await response.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { 
          id: Date.now(), 
          text: `Error: ${data.error}`, 
          sender: 'agent',
          error: true
        }]);
      } else {
        // Handle visualizations if present
        const visualizations = data.visualizations || [];
        
        setMessages(prev => [...prev, { 
          id: Date.now(), 
          text: data.response, 
          sender: 'agent',
          processingTime: data.processingTime,
          visualizations: visualizations
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        text: `Network error: ${error.message}`, 
        sender: 'agent',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-icon">📊</div>
        <div className="sidebar-icon">📈</div>
        <div className="sidebar-icon">💹</div>
        <div className="sidebar-icon">📉</div>
        <div className="sidebar-icon">💰</div>
        <div className="sidebar-icon">🏦</div>
        <div className="sidebar-icon sidebar-divider"></div>
        <div className="sidebar-icon">⚙️</div>
        <div className="sidebar-icon">❓</div>
      </div>
      <div className="main">
        <header className="header">
          <h1 className="title">Finance Analyst AI</h1>
          <div className="header-icons">
            <span className="header-button">New Chat</span>
            <span className="header-button">Share</span>
            <span className="header-button">Settings</span>
          </div>
        </header>
        <div className="chat" ref={chatContainerRef}>
          {messages.length === 1 && (
            <div className="example-queries">
              <h3>Try asking about:</h3>
              <div className="query-suggestions">
                {exampleQueries.map((query, index) => (
                  <div 
                    key={index} 
                    className="query-suggestion" 
                    onClick={() => setInput(query)}
                  >
                    {query}
                  </div>
                ))}
              </div>
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              <div className="message-content">
                {msg.sender === 'agent' && <div className="agent-icon">📊</div>}
                <div className="message-text">
                  {msg.sender === 'agent' ? (
                    <>
                      <div dangerouslySetInnerHTML={{ __html: formatResponse(msg.text) }} />
                      {msg.visualizations && msg.visualizations.length > 0 && (
                        <div className="visualizations">
                          {msg.visualizations.map((viz, index) => (
                            <div key={index} className="visualization-container">
                              <img 
                                src={viz} 
                                alt="Financial Visualization" 
                                className="visualization-image" 
                                onClick={() => window.open(viz, '_blank')}
                              />
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    msg.text
                  )}
                </div>
              </div>
              {msg.sender === 'agent' && !msg.error && (
                <div className="message-footer">
                  <div className="message-actions">
                    <span className="action-button" title="Like">👍</span>
                    <span className="action-button" title="Dislike">👎</span>
                    <span className="action-button" title="Regenerate">🔄</span>
                    <span className="action-button" title="Copy to clipboard">📋</span>
                    <span className="action-button" title="More options">⋮</span>
                  </div>
                  {msg.processingTime && (
                    <span className="processing-time">{msg.processingTime.toFixed(2)}s</span>
                  )}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message agent loading">
              <div className="message-content">
                <div className="agent-icon">📊</div>
                <div className="loading-indicator">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        <div className="input-bar">
          <div className="think-harder" onClick={() => setInput(input + " (Think deeper about this)")}>Think Harder</div>
          <div className="input-container">
            <span className="lock-icon" title="Your queries are secure">🔒</span>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask about stocks, crypto, forex, or financial analysis..."
              onKeyPress={e => e.key === 'Enter' && handleSend()}
            />
            <button className="send-button" onClick={handleSend} disabled={isLoading || !input.trim()}>
              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
