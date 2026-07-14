# Ubuntu (default)

Based on `ubuntu:22.04` (CPU) or `nvidia/cuda:12.6.3-base-ubuntu22.04` (GPU).

> **Other base OS variants:** Dockerfiles for [Arch Linux](../arch/README.md) and [openSUSE Leap](../leap/README.md) are also available if you need a different base distribution.

> **Pre-built images:** CPU and GPU images are automatically built and published to GHCR on every push to `master` via the CI workflow. Each image is a multi-platform manifest covering `linux/amd64` and `linux/arm64`. Pull them directly with:
> ```bash
> docker pull ghcr.io/EPFLiGHT/mmore:edge-gpu
> docker pull ghcr.io/EPFLiGHT/mmore:edge-cpu
> ```

## Build

> **Note:** The default target architecture matches the build host. Pass `--platform=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)

GPU (default):
```bash
sudo docker build -f docker/ubuntu/Dockerfile -t mmore .
```

CPU-only:
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg DEVICE=cpu -t mmore:cpu .
```

Custom extras (overrides the default `--extra all --extra cu126` or `--extra all --extra cpu`):
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg UV_OVERRIDE="--extra process --extra rag --extra cpu" -t mmore .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/ubuntu/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/home/mmoreuser/.cache mmore

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/home/mmoreuser/.cache mmore:cpu
```
