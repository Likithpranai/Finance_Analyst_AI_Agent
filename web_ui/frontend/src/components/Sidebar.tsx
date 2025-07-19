import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiMenu, FiX, FiPlus, FiMoon, FiSun, FiHelpCircle, 
  FiSettings, FiTrash2, FiDollarSign, FiBarChart2, 
  FiTrendingUp, FiGrid, FiClock, FiStar
} from 'react-icons/fi';
import { useAppContext } from '../context/AppContext';
import { getStoredConversations, formatDate } from '../utils';
import '../styles/Sidebar.css';

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
  darkMode: boolean;
  toggleDarkMode: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar, darkMode, toggleDarkMode }) => {
  const { clearMessages } = useAppContext();
  const [conversations, setConversations] = useState<Array<{id: string, title: string, timestamp: number}>>([]);
  
  // Load conversations from local storage
  useEffect(() => {
    const storedConversations = getStoredConversations();
    setConversations(storedConversations);
  }, []);
  
  // Start a new conversation
  const handleNewConversation = () => {
    clearMessages();
    // Close sidebar on mobile after starting new conversation
    if (window.innerWidth <= 768) {
      toggleSidebar();
    }
  };
  
  // Load a conversation
  const handleLoadConversation = (conversationId: string) => {
    // This would load the conversation from storage
    console.log(`Loading conversation: ${conversationId}`);
    // Close sidebar on mobile after selecting conversation
    if (window.innerWidth <= 768) {
      toggleSidebar();
    }
  };
  
  // Delete a conversation
  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation(); // Prevent triggering the parent click event
    const updatedConversations = conversations.filter(conv => conv.id !== conversationId);
    setConversations(updatedConversations);
    localStorage.setItem('conversations', JSON.stringify(updatedConversations));
  };

  // Categories for suggested queries
  const categories = [
    { name: "Stocks", icon: <FiTrendingUp />, color: "#3B82F6" },
    { name: "Technical Analysis", icon: <FiBarChart2 />, color: "#10B981" },
    { name: "Fundamentals", icon: <FiDollarSign />, color: "#F59E0B" },
    { name: "Market Data", icon: <FiGrid />, color: "#8B5CF6" },
    { name: "Portfolio", icon: <FiStar />, color: "#EC4899" },
    { name: "Historical", icon: <FiClock />, color: "#6366F1" }
  ];
  
  return (
    <>
      {/* Mobile sidebar toggle button */}
      <motion.button 
        className={`sidebar-toggle-button ${darkMode ? 'dark' : 'light'}`} 
        onClick={toggleSidebar}
        aria-label="Toggle sidebar"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        {isOpen ? <FiX /> : <FiMenu />}
      </motion.button>
      
      {/* Sidebar */}
      <motion.div 
        className={`sidebar ${isOpen ? 'open' : 'closed'} ${darkMode ? 'dark' : 'light'}`}
        initial={false}
        animate={{ width: isOpen ? '280px' : '0px' }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
      >
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 9H7V20H3V9Z" fill={darkMode ? '#60A5FA' : '#3B82F6'} />
                <path d="M10 4H14V20H10V4Z" fill={darkMode ? '#93C5FD' : '#2563EB'} />
                <path d="M17 13H21V20H17V13Z" fill={darkMode ? '#BFDBFE' : '#1D4ED8'} />
                <path d="M21 7L16 2L11 7L6 2L1 7" stroke={darkMode ? '#F0F9FF' : '#1E3A8A'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h2>Finance Analyst AI</h2>
          </div>
        </div>
        
        <motion.button 
          className="new-chat-button" 
          onClick={handleNewConversation}
          whileHover={{ scale: 1.02, backgroundColor: darkMode ? 'rgba(59, 130, 246, 0.2)' : 'rgba(59, 130, 246, 0.1)' }}
          whileTap={{ scale: 0.98 }}
          transition={{ duration: 0.2 }}
        >
          <FiPlus />
          <span>New Analysis</span>
        </motion.button>
        
        <div className="sidebar-conversations">
          <h3>Recent Analyses</h3>
          <AnimatePresence>
            {conversations.length > 0 ? (
              <motion.ul
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                {conversations.map((conversation, index) => (
                  <motion.li 
                    key={conversation.id} 
                    className="conversation-item"
                    onClick={() => handleLoadConversation(conversation.id)}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05, duration: 0.2 }}
                    whileHover={{ 
                      backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
                      x: 4
                    }}
                  >
                    <div className="conversation-icon">
                      {index % 6 === 0 && <FiTrendingUp color={categories[0].color} />}
                      {index % 6 === 1 && <FiBarChart2 color={categories[1].color} />}
                      {index % 6 === 2 && <FiDollarSign color={categories[2].color} />}
                      {index % 6 === 3 && <FiGrid color={categories[3].color} />}
                      {index % 6 === 4 && <FiStar color={categories[4].color} />}
                      {index % 6 === 5 && <FiClock color={categories[5].color} />}
                    </div>
                    <div className="conversation-content">
                      <div className="conversation-title">{conversation.title}</div>
                      <div className="conversation-date">{formatDate(conversation.timestamp)}</div>
                    </div>
                    <motion.button 
                      className="delete-conversation-button"
                      onClick={(e) => handleDeleteConversation(e, conversation.id)}
                      aria-label="Delete conversation"
                      whileHover={{ scale: 1.2, color: '#EF4444' }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <FiTrash2 />
                    </motion.button>
                  </motion.li>
                ))}
              </motion.ul>
            ) : (
              <motion.div 
                className="no-conversations"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="empty-state-icon">
                  <FiBarChart2 size={24} />
                </div>
                <p>No financial analyses yet</p>
                <p className="empty-state-hint">Start a new analysis to begin</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        <div className="categories-section">
          <h3>Analysis Categories</h3>
          <div className="categories-grid">
            {categories.map((category, index) => (
              <motion.div 
                key={index} 
                className="category-item"
                whileHover={{ scale: 1.05, backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="category-icon" style={{ backgroundColor: `${category.color}20`, color: category.color }}>
                  {category.icon}
                </div>
                <div className="category-name">{category.name}</div>
              </motion.div>
            ))}
          </div>
        </div>
        
        <div className="sidebar-footer">
          <motion.button 
            className="sidebar-button" 
            onClick={toggleDarkMode}
            whileHover={{ backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}
            whileTap={{ scale: 0.98 }}
          >
            {darkMode ? <FiSun className="icon-sun" /> : <FiMoon className="icon-moon" />}
            <span>{darkMode ? 'Light mode' : 'Dark mode'}</span>
          </motion.button>
          
          <motion.button 
            className="sidebar-button"
            whileHover={{ backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}
            whileTap={{ scale: 0.98 }}
          >
            <FiHelpCircle />
            <span>Help & Documentation</span>
          </motion.button>
          
          <motion.button 
            className="sidebar-button"
            whileHover={{ backgroundColor: darkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)' }}
            whileTap={{ scale: 0.98 }}
          >
            <FiSettings />
            <span>Preferences</span>
          </motion.button>
          
          <div className="sidebar-version">
            <div className="version-badge">
              <span>Finance Analyst AI</span>
              <span className="version-number">v1.2.0</span>
            </div>
            <div className="build-info">Last updated: July 2025</div>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default Sidebar;
