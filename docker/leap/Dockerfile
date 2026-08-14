ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM opensuse/leap:15.6 AS gpu
ENV SYNC_ARGS="--extra all --extra cu126"

FROM opensuse/leap:15.6 AS cpu
ENV SYNC_ARGS="--extra all --extra cpu"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV SYNC_ARGS=${UV_OVERRIDE:-$SYNC_ARGS}

# Install system dependencies
RUN zypper --non-interactive refresh && \
    zypper --non-interactive install -y \
      nano curl git ffmpeg \
      libSM6 libXext6 mozilla-nss \
      libXi6 libXrandr2 libXcomposite1 libXcursor1 libXdamage1 libXfixes3 libXrender1 \
      libasound2 libatk-1_0-0 gtk3 libreoffice libjpeg8-devel libpango-1_0-0 && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    echo "Europe/Zurich" > /etc/timezone && \
    zypper clean --all

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV UV_PYTHON=3.11 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies (cached unless uv.lock changes)
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project ${SYNC_ARGS}

# weasyprint needs to be installed using pip
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install weasyprint

# Install mmore from local source code
COPY . /app
RUN uv pip install --no-cache --no-deps -e .

# --- Runtime ---
ENV PATH="/app/.venv/bin:$PATH"
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV HF_HOME="/root/.cache/huggingface"
ENV TORCH_HOME="/root/.cache/torch"
ENV XDG_CACHE_HOME="/root/.cache"

ENTRYPOINT ["/bin/bash"]
