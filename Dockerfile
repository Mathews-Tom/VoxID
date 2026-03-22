# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY src/ src/

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# VoxID data directory
VOLUME /data/voxid
ENV VOXID_STORE_PATH=/data/voxid

# API config
ENV VOXID_API_KEY=""
ENV VOXID_RATE_LIMIT=60
ENV VOXID_RATE_WINDOW=60

EXPOSE 8765

ENTRYPOINT ["python", "-m", "uvicorn", "voxid.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
