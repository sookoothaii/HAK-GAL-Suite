# Backend Dockerfile for HAK-GAL Suite
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file from docs directory
COPY docs/requirements.txt .

# Install Python dependencies with trusted hosts
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy API file from root
COPY api.py .

# Copy backend code
COPY backend/ ./backend/

# Set environment variables
ENV FLASK_APP=api.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5001

# Run the Flask application
CMD ["python", "api.py"]