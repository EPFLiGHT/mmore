# 🌐 Distributed Document Processing

## Overview

This guide explains how to set up and run distributed document processing for the RAG system across multiple nodes.

Distributed processing allows you to scale document indexing across multiple machines, significantly improving processing speed for large document collections. 

The system uses Dask for distributed task scheduling and execution.

## ✅ Prerequisites

Before starting, make sure you have:

- multiple machines/nodes with network connectivity
- a Python environment on each node
- access to a shared filesystem or the ability to copy files between nodes

## 🛠️ Setup Process

### 1. Prepare your configuration file

Check your processing configuration file, for example [`examples/process/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/main/examples/process/config.yaml), and make sure it includes the distributed settings such as:

```yaml
dispatcher_config:
  distributed: true
  scheduler_file: "/path/to/scheduler.json" 
```
The `scheduler_file` should point to a shared location accessible by all nodes.


Other important configuration options include:
- `input_folder`: path to your documents
- `output_folder`: xhere processed results will be stored
- `use_fast_processors`: set to `true` for faster processing (may reduce accuracy)

### 2. Install dependencies on all nodes

On each node, run:

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd mmore

# Make a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 3. Launch the distributed processing

#### Step 1: Start the master node (rank 0)

```bash
bash scripts/process_distributed.sh --config-file /path/to/config.yaml --rank 0
```

The master node will:
- start the Dask scheduler
- launch a worker process
- prompt you to start the processing when ready

#### Step 2: Start worker nodes (rank > 0)

On each additional node, run:

```bash
bash scripts/process_distributed.sh --config-path /path/to/config.yaml --rank 1
```

Replace `rank 1` with a unique rank number for each node (1, 2, 3, etc.). 

The node should be ready within a few seconds.

#### Step 3: Begin processing

Once all nodes are running, return to the master node and type `go`.  
The master node proceeds to crawl the input folder, split the workload among connected nodes and make them start their work.

At the end of processing, the dask server will be automatically shut down by the master node. This also stops the Dask workers on all the connected nodes.


## 📂 Output Structure

After processing completes, the output will be organized as follows:

```
output_folder/
├── processors/
│   ├── Processor_type_1/
│   │   └── results.jsonl
│   ├── Processor_type_2/
│   │   └── results.jsonl
│   └── ...
├── merged/
│   └── merged_results.jsonl
└── images/
```

## 🔧 Troubleshooting

- **Workers not connecting**: ensure all nodes can access the scheduler file location
- **Processing errors**: check logs ona the master node
- **Performance issues**: adjust batch sizes and worker counts in the configuration

## ⚙️ Advanced Configuration

For optimal performance, consider adjusting:
- processor batch sizes
- number of threads per worker
- memory limits for workers

Refer to [Process](../getting_started/process.md) for more details on configuration options.

## See also

- [Process](../getting_started/process.md)
