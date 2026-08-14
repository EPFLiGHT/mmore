# 📦 Installation

## Overview

This page explains how to install mmore.

Four installation paths are currently documented:

- installation standard from PyPI with `pip`
- installation from the source repository with `pip install -e .`
- installation with `uv`
- Docker-based setup

Choose the one that best matches your workflow and environment.

--- 

## API keys and environment variables

Some mmore features require API keys or access tokens.

### Hugging Face

If you use Hugging Face hosted services or models, set your token in the `HF_TOKEN` environment variable before running mmore:

```bash
export HF_TOKEN="your_huggingface_token"
```

### Hosted LLM providers
If you use a hosted LLM provider, make sure the corresponding API key is available in your environment before running mmore.

For example, with OpenAI:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
### Notes
- These environment variables must be set in your shell before launching mmore commands.
-	If you run mmore in Docker, make sure the variables are passed to the container as well.
- If you run mmore on a cluster or in production, make sure these variables are available in the job environment.


--- 

## Standard installation

Use this method if you want to install mmore directly from PyPI.

### Install mmore

```bash
pip install mmore
```

## Installation from the source repository

Use this method if you want to work from the GitHub repository, modify the code, or contribute to mmore.

### Step 1: Clone the repository

```bash
git clone https://github.com/EPFLiGHT/mmore
cd mmore
```

### Step 2: Install the package

```bash
pip install -e .
```

---

## Installation with `uv`

Use this method if you want to install mmore from source while managing the environment and dependencies with `uv`.

### Step 1: Install system dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 \
  libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice \
  libpango-1.0-0 libpangoft2-1.0-0 weasyprint
```

### Step 2: Install `uv`

Refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Clone the repository

```bash
git clone https://github.com/EPFLiGHT/mmore
cd mmore
```

### Step 4: Install the project and dependencies

Pick the pipeline you intend to use. The standard pipeline (`[process]`, with text/document extraction) and the ColVision pipeline (`[colvision]`, with vision-based RAG) are **mutually exclusive** — set up a separate venv if you need both.

```bash
# Standard pipeline (document processing + text RAG)
uv sync --extra process --extra rag

# OR ColVision pipeline (vision-based RAG: ColPali, ColQwen2/2.5, ColGemma3, ColSmol)
uv sync --extra colvision
```

For a CPU-only installation, use:

```bash
uv sync --extra cpu
```

### Step 5: Activate the virtual environment

Before running commands, activate the environment:

```bash
source .venv/bin/activate
```

---

## Installation with Docker

Use this method if you want to run mmore in a containerized environment.

### Step 1: Install Docker

Follow the official [Docker installation guide](https://docs.docker.com/get-started/get-docker/).

### Step 2: Build the Docker image

```bash
sudo docker build -f Dockerfile --build-arg PLATFORM=gpu -t mmore .
```

For CPU-only platforms, use:

```bash
sudo docker build -f Dockerfile --build-arg PLATFORM=cpu -t mmore .
```

If you are running on a shared cluster environment, you can specify `USER_UID` and `USER_GID` variables and set them to your user and group IDs.

### Step 3: Start an interactive session

For GPU-enabled platforms:

```bash
sudo docker run --gpus all -it \
  -v ./examples:/app/examples \
  -v ./.cache:/home/mmoreuser/.cache \
  -e XDG_CACHE_HOME=/home/mmoreuser/.cache \
  -e HF_HOME=/home/mmoreuser/.cache/huggingface \
  -e TORCH_HOME=/home/mmoreuser/.cache/torch \
  mmore
```

For CPU-only platforms:

```bash
sudo docker run -it \
  -v ./examples:/app/examples \
  -v ./.cache:/home/mmoreuser/.cache \
  -e XDG_CACHE_HOME=/home/mmoreuser/.cache \
  -e HF_HOME=/home/mmoreuser/.cache/huggingface \
  -e TORCH_HOME=/home/mmoreuser/.cache/torch \
  mmore
```

```{warning}
You may need the Nvidia container toolkit for Docker containers to access your GPU.
If GPU execution does not work, refer to the official Nvidia container toolkit installation guide:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

To configure the production repository:

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```sh
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Then configure Docker to use the Nvidia runtime:

```sh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

You can then use:

```sh
docker run --gpus all
```

---

## Notes

The `examples` folder is mapped to `/app/examples` inside the container. This corresponds to the default path used in `examples/process/config.yaml`.

For a manual non-Docker setup, use either the standard installation or the `uv` workflow described above.

## See also

- [Quickstart](quickstart.md)
- [Process](process.md)
- [uv workflow](../advanced_usage/uv.md)
- [Running on Windows](windows.md) — what differs on Windows and how to fix it
