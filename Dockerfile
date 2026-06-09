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

# System deps:
#   build-essential - compiles any packages without pre-built wheels
#   libgomp1        - OpenMP runtime; needed by PyTorch CPU for multi-threaded ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install all deps from the lock file into a .venv.
# torch uses the pytorch-cpu index (configured in pyproject.toml) on Linux,
# so no CUDA packages are pulled in.
# --frozen             - fail if lock file is out of date
# --no-dev             - skip dev/notebook dependencies
# --no-install-project - only install deps, not the local package itself
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Put the uv-managed venv on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source, backend, and checked-in data assets
# (includes data/models/analysis_model.pt and the pre-built default cluster cache)
COPY src/ ./src/
COPY backend/ ./backend/
COPY data/ ./data/

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/backend/static ./backend/static

RUN chown -R 1000:1000 /app
USER 1000

EXPOSE 7860

# Single worker: cluster jobs run in BackgroundTasks (in-process _jobs dict),
# so multiple workers would break job-status polling.
# PyTorch already uses both vCPUs internally via OpenMP/MKL.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
