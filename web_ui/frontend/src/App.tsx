import React from 'react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import { AppProvider, useAppContext } from './context/AppContext';
import './styles/App.css';
import './styles/index.css';

const AppContent: React.FC = () => {
  const { theme, isSidebarOpen, setIsSidebarOpen } = useAppContext();
  
  return (
    <div className={`app-container ${theme}`}>
      <Sidebar 
        isOpen={isSidebarOpen} 
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} 
        darkMode={theme === 'dark'}
        toggleDarkMode={() => {}}
      />
      <main className={`main-content ${isSidebarOpen ? 'sidebar-open' : ''}`}>
        <ChatInterface 
          isSidebarOpen={isSidebarOpen}
          darkMode={theme === 'dark'}
        />
      </main>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
};

export default App;
