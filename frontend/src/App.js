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

  const handleThinkDeeper = () => {
    setInput(input + " (Provide deeper analysis)");
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-icon active">
          <i className="fas fa-comments-dollar"></i>
        </div>
        <div className="sidebar-icon">
          <i className="fas fa-chart-bar"></i>
        </div>
        <div className="sidebar-icon">
          <i className="fas fa-chart-line"></i>
        </div>
        <div className="sidebar-icon">
          <i className="fas fa-chart-area"></i>
        </div>
        <div className="sidebar-icon">
          <i className="fas fa-cog"></i>
        </div>
      </div>
      <div className="main">
        <header className="header">
          <div className="header-main">
            <h1 className="title">
              <div className="title-logo">
                <i className="fas fa-chart-pie"></i>
              </div>
              <span className="title-text">Finance <span className="title-highlight">Analyst</span></span>
              <span className="title-version">v2.0</span>
            </h1>
            <div>
              <button className="header-button">
                <span className="header-button-icon"><i className="fas fa-file-alt"></i></span>
                Reports
              </button>
              <button className="header-button">
                <span className="header-button-icon"><i className="fas fa-download"></i></span>
                Export
              </button>
              <button className="header-button primary">
                <span className="header-button-icon"><i className="fas fa-plus"></i></span>
                New Analysis
              </button>
            </div>
          </div>
          <div className="market-ticker">
            <div className="ticker-item">
              <span className="ticker-symbol">S&P 500</span>
              <span className="ticker-price">4,927.11</span>
              <span className="ticker-change positive">
                <i className="fas fa-caret-up"></i> 0.74%
              </span>
            </div>
            <div className="ticker-item">
              <span className="ticker-symbol">NASDAQ</span>
              <span className="ticker-price">15,628.95</span>
              <span className="ticker-change positive">
                <i className="fas fa-caret-up"></i> 1.02%
              </span>
            </div>
            <div className="ticker-item">
              <span className="ticker-symbol">DOW</span>
              <span className="ticker-price">38,239.66</span>
              <span className="ticker-change positive">
                <i className="fas fa-caret-up"></i> 0.56%
              </span>
            </div>
            <div className="ticker-item">
              <span className="ticker-symbol">BTC</span>
              <span className="ticker-price">63,421.85</span>
              <span className="ticker-change negative">
                <i className="fas fa-caret-down"></i> 1.24%
              </span>
            </div>
            <div className="ticker-item">
              <span className="ticker-symbol">ETH</span>
              <span className="ticker-price">3,089.45</span>
              <span className="ticker-change negative">
                <i className="fas fa-caret-down"></i> 0.87%
              </span>
            </div>
            <div className="ticker-item">
              <span className="ticker-symbol">USD/JPY</span>
              <span className="ticker-price">156.72</span>
              <span className="ticker-change positive">
                <i className="fas fa-caret-up"></i> 0.12%
              </span>
            </div>
          </div>
        </header>
        <div className="chat" ref={chatContainerRef}>
          {messages.length === 1 && (
            <div className="welcome">
              <h2>Welcome to Finance Analyst</h2>
              <p className="welcome-subtitle">Professional-grade financial analysis and market intelligence</p>
              
              <div className="dashboard-preview">
                <div className="dashboard-header">
                  <h3><i className="fas fa-chart-line"></i> Market Overview</h3>
                  <div className="dashboard-controls">
                    <span className="time-filter active">1D</span>
                    <span className="time-filter">1W</span>
                    <span className="time-filter">1M</span>
                    <span className="time-filter">3M</span>
                    <span className="time-filter">1Y</span>
                    <span className="time-filter">5Y</span>
                  </div>
                </div>
                <div className="chart-container">
                  <div className="chart-placeholder">
                    <div className="chart-line"></div>
                    <div className="chart-line"></div>
                    <div className="chart-line"></div>
                    <div className="chart-indicators">
                      <div className="indicator">
                        <span className="indicator-name">S&P 500</span>
                        <span className="indicator-value positive">+0.74%</span>
                      </div>
                      <div className="indicator">
                        <span className="indicator-name">NASDAQ</span>
                        <span className="indicator-value positive">+1.02%</span>
                      </div>
                      <div className="indicator">
                        <span className="indicator-name">VIX</span>
                        <span className="indicator-value negative">-2.31%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="query-categories">
                <div className="query-category">
                  <div className="category-header">
                    <i className="fas fa-chart-line"></i>
                    <h3>Technical Analysis</h3>
                  </div>
                  <div className="example-queries">
                    <div className="example-query" onClick={() => handleExampleQuery("Analyze AAPL with RSI, MACD, and Bollinger Bands. Is it a good time to buy?")}>
                      <i className="fas fa-chart-area query-icon"></i>
                      Analyze AAPL with RSI, MACD, and Bollinger Bands
                    </div>
                    <div className="example-query" onClick={() => handleExampleQuery("Calculate support and resistance levels for TSLA based on the last 3 months")}>
                      <i className="fas fa-arrows-alt-v query-icon"></i>
                      Calculate support and resistance levels for TSLA
                    </div>
                  </div>
                </div>
                
                <div className="query-category">
                  <div className="category-header">
                    <i className="fas fa-balance-scale"></i>
                    <h3>Fundamental Analysis</h3>
                  </div>
                  <div className="example-queries">
                    <div className="example-query" onClick={() => handleExampleQuery("Compare P/E, P/B, and profit margins for MSFT, AAPL, and GOOGL")}>
                      <i className="fas fa-table query-icon"></i>
                      Compare P/E, P/B, and profit margins for tech giants
                    </div>
                    <div className="example-query" onClick={() => handleExampleQuery("Analyze NVDA's latest income statement and balance sheet. Is it financially healthy?")}>
                      <i className="fas fa-file-invoice-dollar query-icon"></i>
                      Analyze NVDA's financial statements
                    </div>
                  </div>
                </div>
                
                <div className="query-category">
                  <div className="category-header">
                    <i className="fas fa-chart-pie"></i>
                    <h3>Portfolio Analysis</h3>
                  </div>
                  <div className="example-queries">
                    <div className="example-query" onClick={() => handleExampleQuery("Optimize a portfolio with AAPL, MSFT, AMZN, and BRK.B for maximum Sharpe ratio")}>
                      <i className="fas fa-sliders-h query-icon"></i>
                      Optimize portfolio for maximum Sharpe ratio
                    </div>
                    <div className="example-query" onClick={() => handleExampleQuery("What would be the 5-year return of a portfolio with 40% VOO, 30% QQQ, 20% VGT, and 10% BND?")}>
                      <i className="fas fa-calculator query-icon"></i>
                      Calculate 5-year ETF portfolio return
                    </div>
                  </div>
                </div>
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
          <div className="think-harder" onClick={handleThinkDeeper}>
            Advanced Analysis
          </div>
          <div className="input-container">
            <span className="lock-icon" title="Your queries are secure">🔒</span>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Analyze stocks, financial metrics, market trends, investment strategies..."
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
