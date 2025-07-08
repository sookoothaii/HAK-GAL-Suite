# HAK-GAL Suite Docker Setup

This directory contains Docker configuration files to containerize the HAK-GAL Suite application.

## Components

- **Backend**: Flask API server running on port 5001
- **Frontend**: React/Vite application served by Nginx on port 5173

## Files

- `backend/Dockerfile` - Backend container configuration
- `frontend/Dockerfile` - Frontend container configuration  
- `docker-compose.yml` - Orchestration configuration
- `backend/.dockerignore` - Backend build exclusions
- `frontend/.dockerignore` - Frontend build exclusions
- `frontend/nginx.conf` - Nginx configuration for SPA routing
- `backend/.env` - Backend environment variables

## Prerequisites

- Docker
- Docker Compose

## Quick Start

1. **Build and run all services:**
   ```bash
   docker compose up --build
   ```

2. **Run in detached mode:**
   ```bash
   docker compose up -d --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:5001/api/test

## Individual Services

### Backend Only
```bash
# Build backend
docker compose build backend

# Run backend
docker compose up backend

# Test backend
curl http://localhost:5001/api/test
```

### Frontend Only
```bash
# Build frontend
docker compose build frontend

# Run frontend
docker compose up frontend
```

## Configuration

### Backend Environment Variables

Copy `backend/.env` and update the API keys:

```env
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=false

# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Port Configuration

- Backend: `5001` (internal and external)
- Frontend: `80` (internal) → `5173` (external)

## Docker Architecture

### Backend Container
- Base: `python:3.10-slim`
- Dependencies: Installed from `docs/requirements.txt`
- Entry point: `python api.py`
- Network: `hak-gal-network`

### Frontend Container
- Multi-stage build:
  - Build stage: `node:18-alpine` for building React app
  - Serve stage: `nginx:alpine` for serving static files
- Network: `hak-gal-network`

## Troubleshooting

### Backend Issues
```bash
# Check backend logs
docker compose logs backend

# Rebuild backend
docker compose build --no-cache backend
```

### Frontend Issues
```bash
# Check frontend logs
docker compose logs frontend

# Rebuild frontend
docker compose build --no-cache frontend
```

### Network Issues
```bash
# Check network connectivity
docker compose exec backend ping frontend
docker compose exec frontend ping backend
```

## Development

For development, you can mount volumes to enable hot reload:

```yaml
# Add to docker-compose.yml services
backend:
  volumes:
    - ./backend:/app/backend
    - ./api.py:/app/api.py
    
frontend:
  volumes:
    - ./frontend/src:/app/src
```

## Production Considerations

1. **Environment Variables**: Use proper secrets management
2. **SSL/TLS**: Add reverse proxy with SSL certificates
3. **Scaling**: Consider using Docker Swarm or Kubernetes
4. **Monitoring**: Add health checks and logging
5. **Security**: Review exposed ports and network configuration