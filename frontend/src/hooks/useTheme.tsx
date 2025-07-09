// useTheme.tsx - Theme Context und Hook
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Theme, themes, ThemeName } from '../lib/theme';

interface ThemeContextType {
  theme: Theme;
  themeName: ThemeName;
  toggleTheme: () => void;
  setTheme: (themeName: ThemeName) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [themeName, setThemeName] = useState<ThemeName>(() => {
    // Versuche gespeicherte Theme-Präferenz zu laden
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('hakgal-theme') as ThemeName;
      if (saved && themes[saved]) {
        return saved;
      }
      // Fallback auf System-Präferenz
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
      }
    }
    return 'light';
  });

  const theme = themes[themeName];

  const setTheme = (newThemeName: ThemeName) => {
    setThemeName(newThemeName);
    if (typeof window !== 'undefined') {
      localStorage.setItem('hakgal-theme', newThemeName);
    }
  };

  const toggleTheme = () => {
    setTheme(themeName === 'light' ? 'dark' : 'light');
  };

  // CSS Custom Properties für Theme setzen
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const root = document.documentElement;
      Object.entries(theme.colors).forEach(([key, value]) => {
        root.style.setProperty(`--color-${key}`, value);
      });
    }
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, themeName, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
