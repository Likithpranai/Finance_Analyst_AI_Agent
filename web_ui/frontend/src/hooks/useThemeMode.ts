import { useState, useEffect } from 'react';
import { ThemeMode } from '../types';

/**
 * Custom hook for managing theme mode (dark/light)
 * Handles theme persistence in localStorage and updates HTML class
 */
const useThemeMode = (): [ThemeMode, () => void] => {
  // Get initial theme from localStorage or default to dark
  const getInitialTheme = (): ThemeMode => {
    const savedTheme = localStorage.getItem('theme') as ThemeMode | null;
    return savedTheme || 'dark';
  };

  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  // Apply theme to HTML element when it changes
  useEffect(() => {
    const html = document.documentElement;
    
    // Remove both classes first
    html.classList.remove('dark', 'light');
    
    // Add the current theme class
    html.classList.add(theme);
    
    // Save to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Toggle between dark and light modes
  const toggleTheme = () => {
    setTheme((prevTheme: ThemeMode) => prevTheme === 'dark' ? 'light' : 'dark');
  };

  return [theme, toggleTheme];
};

export default useThemeMode;
