# Use slim Python image to reduce size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY pytest.ini .
COPY api_usage.json .

# Create necessary directories
RUN mkdir -p logs config

# Expose port
EXPOSE 8000

# Set environment variables for Flask
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_DEBUG=false

# Run the Flask application
CMD ["python", "main.py"] 