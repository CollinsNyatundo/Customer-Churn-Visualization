# ── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="BCG Churn Project"
LABEL description="Customer churn Dash dashboard + FastAPI prediction service"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/      ./src/
COPY api/      ./api/
COPY app.py    ./app.py

# Create necessary runtime directories
RUN mkdir -p data/raw data/processed reports models \
 && chown -R appuser:appuser /app

USER appuser

# Environment defaults (override via docker-compose or k8s env)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050

EXPOSE 8050
EXPOSE 8000

# Default: run Dash dashboard via gunicorn
# Override CMD to run the API: uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050", "-w", "2", "--timeout", "120"]
