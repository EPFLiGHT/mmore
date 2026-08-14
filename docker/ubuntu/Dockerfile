ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM nvidia/cuda:12.6.3-base-ubuntu22.04 AS gpu
ENV SYNC_ARGS="--extra all --extra cu126"

FROM ubuntu:22.04 AS cpu
ENV SYNC_ARGS="--extra all --extra cpu"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV SYNC_ARGS=${UV_OVERRIDE:-$SYNC_ARGS}

ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tzdata nano curl git ffmpeg libsm6 libxext6 libnss3 \
      libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxrender1 \
      libasound2 libatk1.0-0 libgtk-3-0 libreoffice libjpeg-dev libpango-1.0-0 \
      libpangoft2-1.0-0 weasyprint && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV UV_PYTHON=3.11 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

# Create non-root user
RUN groupadd --gid ${USER_GID} mmoreuser \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m mmoreuser

WORKDIR /app
RUN chown mmoreuser:mmoreuser /app
USER mmoreuser

# Install dependencies (cached unless uv.lock changes)
COPY --chown=mmoreuser:mmoreuser pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/home/mmoreuser/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --frozen --no-install-project ${SYNC_ARGS}

# Install mmore from local source code
COPY --chown=mmoreuser:mmoreuser . /app
RUN uv pip install --no-cache --no-deps -e .

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV HF_HOME="/home/mmoreuser/.cache/huggingface"
ENV TORCH_HOME="/home/mmoreuser/.cache/torch"
ENV XDG_CACHE_HOME="/home/mmoreuser/.cache"

ENTRYPOINT ["/bin/bash"]