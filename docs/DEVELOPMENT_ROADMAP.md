# HAK-GAL Suite Development Configuration

## Project Structure Improvements

### Current Structure:
```
HAK_GAL_SUITE/
├── api.py                    # Backend API (Flask)
├── backend/                  # Core Logic
├── frontend/                 # React Frontend
├── docs/                     # Documentation
└── .env.example              # Environment Template
```

### Recommended Structure (Next Phase):
```
HAK_GAL_SUITE/
├── backend/
│   ├── core/                 # Core Logic (K-Assistant, Grammar, etc.)
│   ├── api/                  # API Layer (Flask/FastAPI)
│   ├── services/             # Business Logic Layer
│   └── tests/                # Backend Tests
├── frontend/
│   ├── src/
│   ├── tests/                # Frontend Tests
│   └── Dockerfile
├── docs/                     # Centralized Documentation
├── docker-compose.yml        # Development & Production
├── config/                   # Configuration Files
└── scripts/                  # Development Scripts
```

## Immediate Improvements Implemented:

### ✅ 1. Knowledge Base Persistence
- Auto-save after add_raw, retract, learn commands
- Persistent storage across server restarts

### ✅ 2. RAG Context Display  
- Backend sends RAG context to frontend
- Data sources shown in UI
- Document chunks displayed in RAG panel

### ✅ 3. Light/Dark Mode Theme System
- Dynamic theme switching
- Improved contrast and readability
- Persistent theme preference

## Next Phase Recommendations:

### 🔄 1. Unified Development Setup
```bash
# Single command startup
npm run dev:all          # Starts both backend and frontend
npm run dev:backend      # Backend only  
npm run dev:frontend     # Frontend only
```

### 🔄 2. Docker Compose Development
```yaml
# docker-compose.dev.yml
services:
  backend:
    build: ./backend
    volumes: [./backend:/app]
    ports: ["5001:5001"]
  
  frontend:
    build: ./frontend  
    volumes: [./frontend:/app]
    ports: ["3000:3000"]
    depends_on: [backend]
```

### 🔄 3. Configuration Management
```python
# config/settings.py
class Settings:
    API_HOST = "localhost"
    API_PORT = 5001
    FRONTEND_PORT = 3000
    CORS_ORIGINS = ["http://localhost:3000"]
    
    # LLM Provider Settings
    DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
    MISTRAL_API_KEY = env("MISTRAL_API_KEY")
    GEMINI_API_KEY = env("GEMINI_API_KEY")
```

### 🔄 4. Automated Testing
- Unit tests for core logic
- Integration tests for API
- E2E tests for frontend workflows
- Automated test pipeline

### 🔄 5. Deployment Pipeline
- Production Docker images
- Environment-specific configurations
- CI/CD with GitHub Actions
- Automated deployment

## Development Commands:

### Current (Manual):
1. Terminal 1: `cd backend && python api.py`
2. Terminal 2: `cd frontend && npm run dev`

### Planned (Automated):
1. `npm run dev` (starts everything)
2. `docker-compose up` (containerized)

## Technical Debt Items:

1. **Parser Robustness**: Handle numeric arguments better
2. **Negation Logic**: Improve contradiction detection  
3. **Deductive Reasoning**: Add simple inference rules
4. **Error Handling**: More detailed error messages
5. **Performance**: Caching and optimization
