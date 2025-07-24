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
  
  // State for sidebar navigation
  const [activeView, setActiveView] = useState('chat');
  const [likedMessages, setLikedMessages] = useState(new Set());
  const [dislikedMessages, setDislikedMessages] = useState(new Set());
  
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

  const handleExampleQuery = (query) => {
    setInput(query);
    setTimeout(() => handleSend(), 100);
  };

  // Sidebar navigation handlers
  const handleSidebarClick = (view) => {
    setActiveView(view);
  };

  // Message action handlers
  const handleLike = (messageId) => {
    setLikedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
        // Remove from disliked if it was disliked
        setDislikedMessages(prevDisliked => {
          const newDislikedSet = new Set(prevDisliked);
          newDislikedSet.delete(messageId);
          return newDislikedSet;
        });
      }
      return newSet;
    });
  };

  const handleDislike = (messageId) => {
    setDislikedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
        // Remove from liked if it was liked
        setLikedMessages(prevLiked => {
          const newLikedSet = new Set(prevLiked);
          newLikedSet.delete(messageId);
          return newLikedSet;
        });
      }
      return newSet;
    });
  };

  const handleRegenerate = async (messageId) => {
    // Find the user message that preceded this agent message
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex > 0) {
      const userMessage = messages[messageIndex - 1];
      if (userMessage.sender === 'user') {
        // Remove the current agent response
        setMessages(prev => prev.filter(msg => msg.id !== messageId));
        
        // Regenerate response
        setIsLoading(true);
        try {
          const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: userMessage.text })
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
      }
    }
  };

  const handleCopy = (messageText) => {
    // Remove HTML tags for clean text copying
    const cleanText = messageText.replace(/<[^>]*>/g, '');
    navigator.clipboard.writeText(cleanText).then(() => {
      // You could add a toast notification here
      console.log('Message copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy message:', err);
    });
  };

  const handleMoreOptions = (messageId) => {
    // This could open a dropdown menu with additional options
    // For now, we'll just log it
    console.log('More options for message:', messageId);
    // You could implement features like:
    // - Export message
    // - Share message
    // - Report issue
    // - Save to favorites
  };

  // Render different views based on activeView
  const renderMainContent = () => {
    switch (activeView) {
      case 'dashboard':
        return (
          <div className="dashboard-view">
            <h2>Market Dashboard</h2>
            <p>Real-time market data and analytics dashboard coming soon...</p>
            <div className="dashboard-placeholder">
              <i className="fas fa-chart-bar" style={{fontSize: '4rem', color: '#666'}}></i>
              <p>Interactive charts and market overview will be displayed here.</p>
            </div>
          </div>
        );
      case 'technical':
        return (
          <div className="technical-view">
            <h2>Technical Analysis Tools</h2>
            <p>Advanced technical analysis and charting tools coming soon...</p>
            <div className="technical-placeholder">
              <i className="fas fa-chart-line" style={{fontSize: '4rem', color: '#666'}}></i>
              <p>TradingView-like charts and technical indicators will be available here.</p>
            </div>
          </div>
        );
      case 'portfolio':
        return (
          <div className="portfolio-view">
            <h2>Portfolio Management</h2>
            <p>Portfolio tracking and optimization tools coming soon...</p>
            <div className="portfolio-placeholder">
              <i className="fas fa-chart-area" style={{fontSize: '4rem', color: '#666'}}></i>
              <p>Portfolio performance tracking and optimization tools will be displayed here.</p>
            </div>
          </div>
        );
      case 'settings':
        return (
          <div className="settings-view">
            <h2>Settings</h2>
            <div className="settings-content">
              <div className="setting-group">
                <h3>API Configuration</h3>
                <p>Configure your API keys and data sources</p>
              </div>
              <div className="setting-group">
                <h3>Display Preferences</h3>
                <p>Customize the interface and chart settings</p>
              </div>
              <div className="setting-group">
                <h3>Notifications</h3>
                <p>Set up alerts and notification preferences</p>
              </div>
            </div>
          </div>
        );
      default:
        return renderChatView();
    }
  };

  const renderChatView = () => {
    return (
      <>
        {messages.length === 1 && (
          <div className="welcome">
            <h2>Welcome to Finance Analyst</h2>
            <p className="welcome-subtitle">Professional-grade financial analysis and market intelligence</p>
            
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
              {msg.sender === 'agent' && <div className="agent-icon"><i className="fas fa-chart-line"></i></div>}
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
                  <span 
                    className={`action-button ${likedMessages.has(msg.id) ? 'active liked' : ''}`} 
                    title="Like" 
                    onClick={() => handleLike(msg.id)}
                  >
                    <i className="fas fa-thumbs-up"></i>
                  </span>
                  <span 
                    className={`action-button ${dislikedMessages.has(msg.id) ? 'active disliked' : ''}`} 
                    title="Dislike" 
                    onClick={() => handleDislike(msg.id)}
                  >
                    <i className="fas fa-thumbs-down"></i>
                  </span>
                  <span 
                    className="action-button" 
                    title="Regenerate" 
                    onClick={() => handleRegenerate(msg.id)}
                  >
                    <i className="fas fa-sync-alt"></i>
                  </span>
                  <span 
                    className="action-button" 
                    title="Copy to clipboard" 
                    onClick={() => handleCopy(msg.text)}
                  >
                    <i className="fas fa-clipboard"></i>
                  </span>
                  <span 
                    className="action-button" 
                    title="More options" 
                    onClick={() => handleMoreOptions(msg.id)}
                  >
                    <i className="fas fa-ellipsis-v"></i>
                  </span>
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
              <div className="agent-icon"><i className="fas fa-chart-line"></i></div>
              <div className="loading-indicator">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          </div>
        )}
      </>
    );
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div 
          className={`sidebar-icon ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => handleSidebarClick('chat')}
          title="Chat"
        >
          <i className="fas fa-comments-dollar"></i>
        </div>
        <div 
          className={`sidebar-icon ${activeView === 'dashboard' ? 'active' : ''}`}
          onClick={() => handleSidebarClick('dashboard')}
          title="Market Dashboard"
        >
          <i className="fas fa-chart-bar"></i>
        </div>
        <div 
          className={`sidebar-icon ${activeView === 'technical' ? 'active' : ''}`}
          onClick={() => handleSidebarClick('technical')}
          title="Technical Analysis"
        >
          <i className="fas fa-chart-line"></i>
        </div>
        <div 
          className={`sidebar-icon ${activeView === 'portfolio' ? 'active' : ''}`}
          onClick={() => handleSidebarClick('portfolio')}
          title="Portfolio Management"
        >
          <i className="fas fa-chart-area"></i>
        </div>
        <div 
          className={`sidebar-icon ${activeView === 'settings' ? 'active' : ''}`}
          onClick={() => handleSidebarClick('settings')}
          title="Settings"
        >
          <i className="fas fa-cog"></i>
        </div>
      </div>
      <div className="main">
        <header className="header">
          <div className="header-main">
            <h1 className="title">
              <div className="title-logo">
                <i className="fas fa-chart-line"></i>
              </div>
              <span className="title-text">Finance <span className="title-highlight">Analyst</span></span>
              <span className="title-version">v2.0</span>
            </h1>
          </div>
        </header>
        <div className="chat" ref={chatContainerRef}>
          {renderMainContent()}
          {activeView === 'chat' && <div ref={chatEndRef} />}
        </div>
        {activeView === 'chat' && (
          <div className="input-bar">
            <div className="input-container">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Analyze stocks, financial metrics, market trends, investment strategies..."
                onKeyPress={e => e.key === 'Enter' && handleSend()}
              />
              <button className="send-button" onClick={handleSend} disabled={isLoading || !input.trim()}>
                {isLoading ? <i className="fas fa-spinner fa-spin"></i> : <i className="fas fa-paper-plane"></i>}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
