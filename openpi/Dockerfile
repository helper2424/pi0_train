# Multi-stage Dockerfile for OpenPi project - optimized for cloud deployment
# This Dockerfile is designed for production use in cloud environments

# ================================
# Stage 1: Dependencies and Build
# ================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0 AS base

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    linux-headers-generic \
    build-essential \
    clang \
    ca-certificates \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure UV for containerized environment
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_CACHE_DIR=/tmp/uv-cache

# Create Python virtual environment
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# ================================
# Stage 2: Dependencies Installation
# ================================
FROM base AS deps

# Copy dependency files for better Docker layer caching
COPY uv.lock pyproject.toml ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src

# Install dependencies (excluding dev dependencies for production)
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# ================================
# Stage 3: Application Build
# ================================
FROM deps AS app-build

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY examples/ examples/
COPY docs/ docs/
COPY .gitmodules .gitmodules

# Copy other necessary files
COPY README.md LICENSE CONTRIBUTING.md ./

# Initialize git submodules (required for some dependencies)
RUN git init && \
    git submodule update --init --recursive || true

# Install the project itself
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# ================================
# Stage 4: Production Runtime
# ================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS production

# Create non-root user for security
RUN groupadd -r openpi && useradd -r -g openpi -u 1001 openpi

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy UV binary
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Set working directory and copy virtual environment
WORKDIR /app
COPY --from=app-build /.venv /.venv
COPY --from=app-build /app .

# Set environment variables for production
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV OPENPI_DATA_HOME=/openpi_assets
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create cache directory with proper permissions
RUN mkdir -p /openpi_assets && chown -R openpi:openpi /openpi_assets /app

# Expose default port for policy server
EXPOSE 8000

# Switch to non-root user
USER openpi

# Health check for the service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to serve the policy
CMD ["uv", "run", "scripts/serve_policy.py", "policy:checkpoint", "--host", "0.0.0.0", "--port", "8000"]

# ================================
# Stage 5: Development Runtime (optional)
# ================================
FROM app-build AS development

# Install development dependencies
RUN --mount=type=cache,target=/tmp/uv-cache \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project

# Set environment variables for development
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV OPENPI_DATA_HOME=/openpi_assets

# Default command for development
CMD ["/bin/bash"] 