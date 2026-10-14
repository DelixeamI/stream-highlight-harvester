# ==============================================================================
# Production Dockerfile for Stream Highlight Harvester (narezka)
# Multimodal gaming VOD highlight extraction service with CUDA & NVENC support
# ==============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies: Python 3.12, FFmpeg, audio/image codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install modern fast package manager: uv
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Create virtualenv and install production dependencies
RUN uv venv --python /usr/bin/python3.12 /app/.venv && \
    uv sync --frozen --no-dev

# Copy application source code and configuration assets
COPY src/ /app/src/
COPY config/ /app/config/
COPY plugins/ /app/plugins/
COPY prompts/ /app/prompts/

# Create runtime directory for project storage
RUN mkdir -p /app/projects

EXPOSE 8000

# Default command: launch production ASGI FastAPI server
CMD ["uvicorn", "narezka.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
