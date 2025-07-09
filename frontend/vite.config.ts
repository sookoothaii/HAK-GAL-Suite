import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 3000,  // Geändert von 8080 auf 3000 wegen Jenkins-Konflikt
  },
  plugins: [
    react(),
    // componentTagger nur in development, falls verfügbar
    // mode === 'development' && componentTagger(), // Entfernt da wahrscheinlich nicht verfügbar
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
