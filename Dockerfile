# Multi-stage build for minimal image size
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# =============================================================================
# Final stage (minimal production image)
# =============================================================================

FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to use our pip packages
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1

# Copy application code
COPY project/ .
COPY data/ ./data/
COPY models/ ./models/

# Expose ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: run Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port=8501"]
