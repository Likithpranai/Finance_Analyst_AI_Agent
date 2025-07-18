import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiMenu, FiX, FiPlus, FiMoon, FiSun, FiHelpCircle, FiSettings, FiTrash2 } from 'react-icons/fi';
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

  return (
    <>
      {/* Mobile sidebar toggle button */}
      <button 
        className={`sidebar-toggle-button ${darkMode ? 'dark' : 'light'}`} 
        onClick={toggleSidebar}
        aria-label="Toggle sidebar"
      >
        {isOpen ? <FiX /> : <FiMenu />}
      </button>
      
      {/* Sidebar */}
      <motion.div 
        className={`sidebar ${isOpen ? 'open' : 'closed'} ${darkMode ? 'dark' : 'light'}`}
        initial={false}
        animate={{ width: isOpen ? '280px' : '0px' }}
        transition={{ duration: 0.3 }}
      >
        <div className="sidebar-header">
          <h2>Finance Analyst AI</h2>
        </div>
        
        <button className="new-chat-button" onClick={handleNewConversation}>
          <FiPlus />
          <span>New conversation</span>
        </button>
        
        <div className="sidebar-conversations">
          <h3>Recent conversations</h3>
          {conversations.length > 0 ? (
            <ul>
              {conversations.map(conversation => (
                <li 
                  key={conversation.id} 
                  className="conversation-item"
                  onClick={() => handleLoadConversation(conversation.id)}
                >
                  <div className="conversation-title">{conversation.title}</div>
                  <div className="conversation-date">{formatDate(conversation.timestamp)}</div>
                  <button 
                    className="delete-conversation-button"
                    onClick={(e) => handleDeleteConversation(e, conversation.id)}
                    aria-label="Delete conversation"
                  >
                    <FiTrash2 />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="no-conversations">No conversations yet</div>
          )}
        </div>
        
        <div className="sidebar-footer">
          <button className="sidebar-button" onClick={toggleDarkMode}>
            {darkMode ? <FiSun /> : <FiMoon />}
            <span>{darkMode ? 'Light mode' : 'Dark mode'}</span>
          </button>
          
          <button className="sidebar-button">
            <FiHelpCircle />
            <span>Help & FAQ</span>
          </button>
          
          <button className="sidebar-button">
            <FiSettings />
            <span>Settings</span>
          </button>
          
          <div className="sidebar-version">
            <span>Finance Analyst AI v1.0.0</span>
          </div>
        </div>
      </motion.div>
    </>
  );
};

export default Sidebar;
