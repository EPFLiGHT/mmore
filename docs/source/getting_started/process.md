# ⚙️ Process

## Overview

The process module enables the extraction and standardization of text and images from diverse file formats (listed below), making it ideal for creating datasets for applications such as RAG, multimodal content generation, and preprocessing data for multimodal LLMs and LLMs.

## 🔨Quick Start
### 👩‍💻 Global installation
Set up the project on each device you want to use by following [Installation](installation.md).

### 💻 Running locally
To run the process locally, first specify the input folders in the [config file `examples/process/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/process/config.yaml). You can also adjust the parameters to your needs.  
Once ready, run:

```bash
python3 -m mmore process --config-file examples/process/config.yaml
```

### 📌 Google Drive support
mmore also supports processing documents directly from **Google Drive**.

To enable this feature, the user must create a [Google service account](https://cloud.google.com/iam/docs/service-accounts-create) and download the corresponding secrets as a JSON file. Name that file `client_secrets.json` and put it in `googledrive/` (this folder may need to be created at the root of the mmore repository).

Make sure your **Google service account** has permission to view the drives you want to process.

#### Referencing Google Drive sources

Google Drive folders are referenced in the process config file through the `google_drive_ids` field.

For example:

```yaml
data_path: examples/sample_data/ # Put absolute path ! Possible to pass a list of folders 
google_drive_ids: [] # Put ids of Google Drive folders
```
- `data_path` is used for local input folder
- `google_drive_ids` is used to provide one or more Google Drive folder IDs to process

To process documents from Google Drive, add the folder IDs to the list:
```yaml
data_path: examples/sample_data/
google_drive_ids:
  - your_google_drive_folder_id
  - another_google_drive_folder_id
```

You can use local folders, Google Drive folders, or both in the same configuration.

Make sure each referenced Google Drive folder is shared with the service account used by mmore.

You can find an example config file in [`examples/process/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/process/config.yaml).


### 📂 Output structure

The output of the pipeline has the following structure:
```
output_path
├── processors
│   ├── Processor_type_1
│   │   └── results.jsonl
│   ├── Processor_type_2
│   │   └── results.jsonl
│   ├── ...
│   
└── merged
│    └── merged_results.jsonl
|
└── images
```
### 🚀 Running on distributed nodes

A simple bash script is provided to run the process in distributed mode.

```bash
bash scripts/process_distributed.sh -f /path/to/my/input/folder
```

See also [Distributed processing](../advanced_usage/distributed_processing.md).


### 📜 Examples
You can find additional example scripts in the [`/examples`](https://github.com/EPFLiGHT/mmore/blob/master/examples) directory.

## ⚡ Optimization

### 🏎️ Fast mode

For some file types, we provide a fast mode that will allow you to process the files faster, using a different method. To use it, set the `use_fast_processors` to `true` in the config file.

Be aware that the fast mode might not be as accurate as the default mode, especially for scanned non-native PDFs, which may require Optical Character Recognition (OCR) for more accurate extraction.


### 🔧 File type parameters tuning

Many parameters are hardware-dependent and can be customized to suit your needs.  

For example, you can tune: 
- processor batch size
- dispatcher batch size
- number of threads per worker

You can configure parameters by providing a custom config file. You can find an example of a config file in [`examples/process/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/process/config.yaml).


⚠️ Not all parameters are configurable yet.

For distributed execution options, see the [Quick Start](quickstart.md) and [Distributed processing](../advanced_usage/distributed_processing.md).

### ♻️ Incremental reprocessing

The optional top-level `previous_results` parameter lets you reuse results from a prior run to avoid reprocessing unchanged files so as to save time and compute costs.

```yaml
previous_results: examples/process/outputs/merged/merged_results.jsonl
```

Point it to a `merged_results.jsonl` produced by an earlier run. On the next run, each local input file is compared against that JSONL (meanwhile URL inputs are always reprocessed):

- Unchanged files: their previous samples are reused as-is.
- New or modified files: they are processed normally.
- Removed files: their samples are dropped from the output.



## 📜 More information on what's under the hood

### 🚧 Pipeline architecture

Our pipeline is a 3 steps process:

1. **Crawling**  
   Files and folders are scanned to identify the files to process, while skipping those already processed.

2. **Dispatching**  
   Files are dispatched to workers in batches. In distributed setups, this stage is also responsible for load balancing across nodes.

3. **Processing**  
   Workers process files with the appropriate tools for each file type. They extract text, images, audio, and video frames, then pass the results to the next stage.

mmore uses a common data structure for document samples: [MultimodalSample](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/type.py#L38).

The goal is to make it easy to add new processors for new file types, or alternative processing methods for existing ones.


## 🛠️ Supported file types and tools

The project supports multiple file types and utilizes various AI-based tools for processing. Below is a table summarizing the supported file types and corresponding tools (N/A means no choice):

| **File Type**                         | **Default Mode Tool(s)**                                                                                                          | **Fast Mode Tool(s)**                                                                                                         |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **DOCX**                              | [python-docx](https://python-docx.readthedocs.io/en/latest/) to extract the text and images.                                      | N/A                                                                                                                         |
| **MD**                                | [markdown](https://python-markdown.github.io/) for text extraction, [markdownify](https://pypi.org/project/markdownify/) for HTML conversion | N/A                                                                                                                         |
| **PPTX**                              | [python-pptx](https://python-pptx.readthedocs.io/en/latest/) to extract the text and images.                                      | N/A                                                                                                                         |
| **XLSX**                              | [openpyxl](https://openpyxl.readthedocs.io/en/stable/) to extract the text and images.                                           | N/A                                                                                                                         |
| **TXT**                               | [python built-in library](https://docs.python.org/3/library/functions.html#open)                                                 | N/A                                                                                                                         |
| **EML**                               | [python built-in library](https://docs.python.org/3/library/email.html) | N/A                                                                                                                         |
| **MP4, MOV, AVI, MKV, MP3, WAV, AAC** | [moviepy](https://pypi.org/project/moviepy/) for video frame extraction; [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for transcription | [whisper-tiny](https://huggingface.co/openai/whisper-tiny)                                                                  |
| **PDF**                               | [marker-pdf](https://github.com/VikParuchuri/marker) for OCR and structured data extraction                                      | [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for text and image extraction                                                 |
| **HTML**                         | [markdownify](https://pypi.org/project/markdownify/) to convert HTML to MD; [requests](https://docs.python-requests.org/en/master/) for images | N/A
---

mmore also uses [Dask Distributed](https://distributed.dask.org/en/latest/) to manage distributed execution.

## 🔧 Customization
The system is designed to be extensible, allowing you to register custom processors for handling new file types or specialized processing. To implement a new processor you need to inherit the `Processor` class and implement only two methods:
- `accepts`: defines which file types your processor supports (e.g. docx)
- `process`: how to process a single file (input:file type, output: Multimodal sample, see other processors for reference)

For a minimal example, see [`TextProcessor`](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/process/processors/txt_processor.py).

## 🧹 Post-processing

Post-processing refines the extracted text data to improve quality for downstream tasks. The infrastructure is modular and extensible: mmore natively supports the following post-processors: 

- [`Chunker`](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/process/post_processor/chunker)
- [`Filter`](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/process/post_processor/filter)
- [`Named Entity Recognition`](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/process/post_processor/ner)
- [`Tagger`](https://github.com/EPFLiGHT/mmore/blob/master/src/mmore/process/post_processor/tagger)

Applying the **Chunker** is heavily recommended, as it cuts documents into reasonably sized chunks that are more specific to feed to an LLM.  

The chunker supports a `table_handling` option to control how markdown tables are split:

| Mode | Description |
|---|---|
| `single_row` (default) | Each table row has its own chunk, with the header repeated for context |
| `multi_rows` | Rows are grouped to fill the chunk size, header repeated per chunk |
| `keep_whole` | Tables are never split and kept as one chunk regardless of size |
| `none` | No special table handling, tables are chunked like regular text |



You can configure parameters by providing a custom config file. This field is shown in the example config file at [`examples/process/config.yaml`](https://github.com/EPFLiGHT/mmore/blob/master/examples/process/config.yaml).


Once ready, you can run the process using the following command:
```bash
python3 -m mmore postprocess --config-file examples/postprocessor/config.yaml --input-data examples/process/outputs/merged/merged_results.jsonl
```

Specify with `--input-data` the path (absolute or relative to the root of the repository) to the JSONL recording of the output of the initial processing phase.


### ♻️ Incremental post-processing

Like the processing pipeline, the post-processor accepts an optional `previous_results` parameter to reuse results from a prior post-processing run and skip unchanged documents.

```yaml
previous_results: examples/postprocessor/outputs/merged/results.jsonl
```


New post-processors can easily be implemented, and pipelines can be configured through lightweight YAML files. The post-processing stage produces a new JSONL file containing cleaned and optionally enhanced document samples.


## See also

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Indexing](indexing.md)
- [Distributed processing](../advanced_usage/distributed_processing.md)
