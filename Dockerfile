# Stage 1: Build React frontend
# Vite outputs directly to backend/static/ (see frontend/vite.config.ts)
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# Stage 2: Python backend
FROM python:3.13-slim

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# System deps needed by some Python packages (e.g. build tools for umap-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject.toml
# torch is large — install separately first so Docker can cache the layer
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# Copy source, backend, and checked-in data assets
COPY src/ ./src/
COPY backend/ ./backend/
COPY data/ ./data/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/backend/static ./backend/static

RUN chown -R 1000:1000 /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
