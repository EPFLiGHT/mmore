# 🚀 Cluster and Production Deployment

This page provides guidelines for deploying mmore in a shared cluster environment and, if needed, adapting it for production deployment.

The examples below assume a Run:ai-based cluster setup with a shared persistent volume. You may need to adapt paths, scheduler options, and environment variables to match your infrastructure.

## 🐳 Docker Image Requirements

```{important}
Build your own Docker image with your specific user ID and group ID to avoid permission issues in the production environment.
```

### 1. Check your user and group IDs on the cluster
```bash
id -u  # Your user ID
id -g  # Your group ID
```

### 2. Build Docker image with custom IDs

Choose one of the two options below:  

**Option A — CI build (recommended):** Trigger the [Build Student Image](https://github.com/EPFLiGHT/mmore/actions/workflows/push-to-registry.yml) workflow manually from the GitHub Actions tab (*Run workflow*) and input your user UID and group GID. This builds a custom student GPU image published to GHCR, tagged as:
```
ghcr.io/EPFLiGHT/mmore:student-uid<user-id>-gid<group-id>-gpu
```
You can then use this image reference directly in the Run:ai commands below. 

**Option B — local build** (replace `<user-id>` and `<group-id>` with your actual IDs):
```bash
sudo docker build -f Dockerfile --build-arg USER_UID=<user-id> --build-arg USER_GID=<group-id> -t mmore .
```

**Login to DockerHub** *(option B only)*:  
```bash
docker login docker.io
```

**Push the image to the registry** *(option B only)*:  
Replace `username` with your DockerHub username.
```bash
docker tag mmore docker.io/<username>/mmore:latest
docker push docker.io/<username>/mmore:latest
```

### 3. Identify your image reference 

All `runai` commands below use `<image>` as a placeholder. Replace it with:
- Option A: `ghcr.io/EPFLiGHT/mmore:student-uid<user-id>-gid<group-id>-gpu`
- Option B: `docker.io/<username>/mmore:latest`

For detailed installation instructions, see [Installation](../getting_started/installation.md).

## 🖥️ Running on a cluster

### Environment setup

First, define the main environment variables used for input, output, and cache data.

The examples below use the RCP path convention based on `$GASPAR`. Adapt these paths if your cluster uses a different shared storage layout.

```bash
export ROOT_OUT_DIR=/lightscratch/users/$GASPAR/mmore-data/out
export ROOT_IN_DIR=/lightscratch/users/$GASPAR/mmore-data/in
export XDG_CACHE_HOME=/lightscratch/users/$GASPAR/.cache
export HF_HOME=/lightscratch/users/$GASPAR/.cache/huggingface
export TORCH_HOME=/lightscratch/users/$GASPAR/.cache/torch
```

### Directory structure initialization

Create the required directory structure on the shared storage volume:

```bash
mkdir -p $ROOT_IN_DIR
mkdir -p $ROOT_OUT_DIR
mkdir -p $ROOT_OUT_DIR/db
mkdir -p $ROOT_OUT_DIR/process/outputs/images
mkdir -p $ROOT_IN_DIR/sample_data
mkdir -p $XDG_CACHE_HOME
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
```

## 💻 Interactive Development Session

For development, debugging, or manual operations, you can start an interactive session on the cluster.

The example below assumes a Run:ai-based environment. Replace `<group-id>` with your actual group ID. 

```bash
runai submit mmore \
  --image <image> \
  --node-pool h100 \
  --pvc light-scratch:/lightscratch \
  --gpu 1 \
  --run-as-gid <group-id> \
  --preemptible \
  --attach \
  --interactive \
  --tty \
  --command /bin/bash
```

This provides a direct terminal access to the container.

## ⚙️ Production Pipeline Execution

The following examples show how to submit mmore pipeline stages as cluster jobs in a Run:ai-based environment.  
Adapt resource settings, storage mounts, paths, and scheduler flags to your infrastructure.

### 1. Document Processing

Process raw documents and extract multimodal content.  
Replace `<group-id>` with your actual group ID. 

```bash
runai submit \
  --name mmore-process \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=$ROOT_IN_DIR \
  -e ROOT_OUT_DIR=$ROOT_OUT_DIR \
  -e XDG_CACHE_HOME=$XDG_CACHE_HOME \
  -e HF_HOME=$HF_HOME \
  -e TORCH_HOME=$TORCH_HOME \
  --command "python3 -m mmore process --config-file production-config/process/config.yaml"
```

### 2. Post-processing

Clean and structure the extracted data.

```bash
runai submit \
  --name mmore-postprocess \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=$ROOT_IN_DIR \
  -e ROOT_OUT_DIR=$ROOT_OUT_DIR \
  -e XDG_CACHE_HOME=$XDG_CACHE_HOME \
  -e HF_HOME=$HF_HOME \
  -e TORCH_HOME=$TORCH_HOME \
  --command "python3 -m mmore postprocess --config-file production-config/postprocessor/config.yaml --input-data $ROOT_OUT_DIR/process/outputs/merged/merged_results.jsonl"
```

### 3. Vector indexing

Create searchable vector indexes.

```bash
runai submit \
  --name mmore-index \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=$ROOT_IN_DIR \
  -e ROOT_OUT_DIR=$ROOT_OUT_DIR \
  -e XDG_CACHE_HOME=$XDG_CACHE_HOME \
  -e HF_HOME=$HF_HOME \
  -e TORCH_HOME=$TORCH_HOME \
  --command "python3 -m mmore index --config-file production-config/index/config.yaml --documents-path $ROOT_OUT_DIR/postprocessor/outputs/merged/final_pp.jsonl"
```

### 4. RAG Service Deployment

Deploy the retrieval API service.

```bash
runai submit \
  --name mmore-rag \
  --image <image> \
  --backoff-limit 0 \
  --pvc light-scratch:/lightscratch \
  --run-as-gid <group-id> \
  --node-pool h100 \
  --gpu 1 \
  -e ROOT_IN_DIR=$ROOT_IN_DIR \
  -e ROOT_OUT_DIR=$ROOT_OUT_DIR \
  -e HF_TOKEN=$HF_TOKEN \
  -e XDG_CACHE_HOME=$XDG_CACHE_HOME \
  -e HF_HOME=$HF_HOME \
  -e TORCH_HOME=$TORCH_HOME \
  --command "python3 -m mmore live-retrieval --config-file production-config/retriever_api/config.yaml"
```

## Port-forwarding 

To access the service locally, use:

```bash
runai port-forward mmore-rag 8080:8080
```


## See also

- [Installation](../getting_started/installation.md)
- [Indexing](../getting_started/indexing.md)
- [RAG](../getting_started/rag.md)