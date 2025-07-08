import { create } from 'zustand';

export interface KnowledgeItem {
  id: string;
  content: string;
}

export interface Message {
  id: string;
  type: "user" | "success" | "unknown" | "warning";
  content: string;
  reasoning?: string;
}

export interface DataSource {
  id: string;
  filename: string;
}

interface Store {
  // State
  inputMessage: string;
  permanentKnowledge: KnowledgeItem[];
  learningSuggestions: KnowledgeItem[];
  conversationHistory: Message[];
  dataSources: DataSource[];
  rightPanelView: "rag" | "profile";
  profileEntity: string;
  ragContext: string;
  isLoading: boolean;

  // Actions
  setInputMessage: (message: string) => void;
  addToKnowledge: (suggestion: KnowledgeItem) => void;
  discardSuggestion: (id: string) => void;
  deleteKnowledge: (id: string) => void;
  addMessage: (message: Omit<Message, 'id'>) => void;
  setRightPanelView: (view: "rag" | "profile") => void;
  setProfileEntity: (entity: string) => void;
  setIsLoading: (loading: boolean) => void;
  sendMessage: () => void;
}

export const useStore = create<Store>((set, get) => ({
  // Initial state
  inputMessage: "",
  permanentKnowledge: [
    { id: 'pk-1', content: "Server capacity is 100GB" },
    { id: 'pk-2', content: "Network latency is <10ms" },
    { id: 'pk-3', content: "User authentication required" }
  ],
  learningSuggestions: [
    { id: 'ls-1', content: "Database backup scheduled daily" },
    { id: 'ls-2', content: "SSL certificates expire in 30 days" },
    { id: 'ls-3', content: "Memory usage at 75% capacity" }
  ],
  conversationHistory: [
    { 
      id: 'msg-1', 
      type: "user", 
      content: "What is the server status?" 
    },
    { 
      id: 'msg-2', 
      type: "success", 
      content: "Ja.", 
      reasoning: "Server is operational based on monitoring data from last health check." 
    },
    { 
      id: 'msg-3', 
      type: "unknown", 
      content: "Unbekannt.", 
      reasoning: "Insufficient data to determine network topology." 
    }
  ],
  dataSources: [
    { id: 'ds-1', filename: "server_logs.json" },
    { id: 'ds-2', filename: "network_config.xml" },
    { id: 'ds-3', filename: "user_data.csv" }
  ],
  rightPanelView: "rag",
  profileEntity: "",
  ragContext: `Relevant context chunks:
1. Server monitoring indicates normal operation
2. Recent backup completed successfully
3. Network latency within acceptable ranges`,
  isLoading: false,

  // Actions
  setInputMessage: (message) => set({ inputMessage: message }),

  addToKnowledge: (suggestion) => set((state) => ({
    permanentKnowledge: [...state.permanentKnowledge, suggestion],
    learningSuggestions: state.learningSuggestions.filter(item => item.id !== suggestion.id)
  })),

  discardSuggestion: (id) => set((state) => ({
    learningSuggestions: state.learningSuggestions.filter(item => item.id !== id)
  })),

  deleteKnowledge: (id) => set((state) => ({
    permanentKnowledge: state.permanentKnowledge.filter(item => item.id !== id)
  })),

  addMessage: (message) => set((state) => ({
    conversationHistory: [...state.conversationHistory, { 
      ...message, 
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` 
    }]
  })),

  setRightPanelView: (view) => set({ rightPanelView: view }),

  setProfileEntity: (entity) => set({ profileEntity: entity }),

  setIsLoading: (loading) => set({ isLoading: loading }),

  sendMessage: () => {
    const { inputMessage, addMessage, setInputMessage, setProfileEntity, setRightPanelView } = get();
    
    if (inputMessage.trim()) {
      // Check for special commands
      if (inputMessage.toLowerCase().startsWith("what_is ")) {
        const entity = inputMessage.slice(8).trim();
        setProfileEntity(entity);
        setRightPanelView("profile");
      }
      
      addMessage({ type: "user", content: inputMessage });
      setInputMessage("");
    }
  }
}));