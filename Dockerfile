# ─── Stage 1: Builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY results/ ./results/
COPY requirements.txt .

# Create data directory for runtime dataset generation
RUN mkdir -p /app/data

# Expose ports: 8000 (API), 8501 (Streamlit)
EXPOSE 8000 8501

# Default command: run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
