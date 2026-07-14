ARG DEVICE
ARG UV_OVERRIDE

# --- Base images ---
FROM archlinux:latest AS gpu
ENV SYNC_ARGS="--extra all --extra cu126"

FROM archlinux:latest AS cpu
ENV SYNC_ARGS="--extra all --extra cpu"

# --- Build ---
FROM ${DEVICE:-gpu} AS build
ARG UV_OVERRIDE
ENV SYNC_ARGS=${UV_OVERRIDE:-$SYNC_ARGS}
RUN echo "DisableSandbox" >> /etc/pacman.conf

ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
      base-devel \
      tzdata curl git ffmpeg \
      libsm libxext nss \
      libxi libxrandr libxcomposite libxcursor libxdamage libxfixes libxrender \
      alsa-lib atk gtk3 libreoffice-fresh libjpeg-turbo pango && \
    ln -fs /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
    echo "Europe/Zurich" > /etc/timezone && \
    pacman -Sc --noconfirm

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

# weasyprint needs to be installed using pip
RUN --mount=type=cache,target=/home/mmoreuser/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv pip install weasyprint

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
