# Arch Linux

Based on `archlinux:latest`.

## Build

> **Note:** The default target architecture matches the build host. Pass `--platform=<value>` to override:
> - `linux/amd64` — x86_64 servers (e.g. RCP)
> - `linux/arm64` — ARM64 machines (e.g. Apple Silicon)

GPU (default):

```bash
sudo docker build -f docker/arch/Dockerfile -t mmore:arch .
```

CPU-only:
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg DEVICE=cpu -t mmore:arch-cpu .
```

Custom extras (overrides the default `--extra all --extra cu126` or `--extra all --extra cpu`):
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg UV_OVERRIDE="--extra process --extra rag --extra cpu" -t mmore:arch .
```

Custom user UID/GID (e.g. for RCP):
```bash
sudo docker build -f docker/arch/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mmore:arch .
```

## Run

```bash
# GPU
sudo docker run --gpus all -it -v ./examples:/app/examples -v ./.cache:/home/mmoreuser/.cache mmore:arch

# CPU-only
sudo docker run -it -v ./examples:/app/examples -v ./.cache:/home/mmoreuser/.cache mmore:arch-cpu
```
