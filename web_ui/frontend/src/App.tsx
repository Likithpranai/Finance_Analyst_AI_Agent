import React from 'react';
import GrokChatUI from './components/GrokChatUI';
import { AppProvider, useAppContext } from './context/AppContext';
import './styles/App.css';
import './styles/index.css';

const AppContent: React.FC = () => {
  const { theme, toggleTheme } = useAppContext();
  
  return (
    <div className="app-container grok-mode">
      <GrokChatUI 
        darkMode={theme === 'dark'}
        toggleDarkMode={toggleTheme}
      />
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
