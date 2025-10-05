FROM python:3.11-slim AS base

WORKDIR /app

# Only copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Allow overriding via env; do not bake secrets
ENV PYTHONUNBUFFERED=1 \
    UVICORN_PORT=8080 \
    UVICORN_HOST=0.0.0.0

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
