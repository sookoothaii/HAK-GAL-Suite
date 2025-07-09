// theme.ts - Theme-System für HAK-GAL Suite
export interface Theme {
  name: string;
  colors: {
    background: string;
    surface: string;
    surfaceSecondary: string;
    text: string;
    textSecondary: string;
    textMuted: string;
    border: string;
    borderSecondary: string;
    primary: string;
    primaryHover: string;
    success: string;
    warning: string;
    error: string;
    accent: string;
  };
}

export const lightTheme: Theme = {
  name: "light",
  colors: {
    background: "#ffffff",
    surface: "#f8fafc",
    surfaceSecondary: "#f1f5f9", 
    text: "#0f172a",
    textSecondary: "#334155",
    textMuted: "#64748b",
    border: "#e2e8f0",
    borderSecondary: "#cbd5e1",
    primary: "#3b82f6",
    primaryHover: "#2563eb",
    success: "#10b981",
    warning: "#f59e0b",
    error: "#ef4444",
    accent: "#8b5cf6"
  }
};

export const darkTheme: Theme = {
  name: "dark", 
  colors: {
    background: "#0f172a",
    surface: "#1e293b",
    surfaceSecondary: "#334155",
    text: "#f8fafc",
    textSecondary: "#e2e8f0",
    textMuted: "#94a3b8",
    border: "#475569",
    borderSecondary: "#64748b",
    primary: "#60a5fa",
    primaryHover: "#3b82f6",
    success: "#34d399",
    warning: "#fbbf24",
    error: "#f87171",
    accent: "#a78bfa"
  }
};

export const themes = {
  light: lightTheme,
  dark: darkTheme
};

export type ThemeName = keyof typeof themes;
