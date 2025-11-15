# Development container for AskDocs MCP Server
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

WORKDIR /workspace
