# 💻 Developer Documentation

Welcome to the mmore developer documentation!  
This guide will help you set up your development environment and contribute to the project.

## Table of Contents

- [💻 Developer Documentation](#-developer-documentation)
  - [Table of Contents](#table-of-contents)
  - [🛠️ Development setup](#️-development-setup)
    - [System dependencies](#system-dependencies)
      - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
      - [macOS](#macos)
    - [Installing mmore for development](#installing-mmore-for-development)
    - [🧹 Code quality tools](#-code-quality-tools)
      - [Pre-commit hooks](#pre-commit-hooks)
        - [Setup](#setup)
          - [1. Install pre-commit](#1-install-pre-commit)
          - [2. Set up the git hook scripts](#2-set-up-the-git-hook-scripts)
          - [3. Run the checks manually](#3-run-the-checks-manually)
      - [Configured Hooks](#configured-hooks)
      - [Type Checking](#type-checking)
  - [🤝 Contributing Guidelines](#-contributing-guidelines)
    - [Reporting Issues](#reporting-issues)
    - [Code Contributions](#code-contributions)
  - [🗂️ Project Structure](#️-project-structure)
    - [Key Modules](#key-modules)
  - [🧪 Testing](#-testing)
    - [Running tests in the terminal](#running-tests-in-the-terminal)
    - [GPU tests](#gpu-tests)
    - [Writing tests](#writing-tests)
  - [🔀 Pull Request Process](#-pull-request-process)
    - [PR checklist](#pr-checklist)
  - [🖥️ Interactive TUI](#️-interactive-tui)
  - [💡 Development tips](#-development-tips)
    - [Working with `uv`](#working-with-uv)
    - [Verbose / debug output (`MMORE_VERBOSE`)](#verbose--debug-output-mmore_verbose)
  - [❓ Questions](#-questions)

---

## 🛠️ Development setup

### System dependencies

Before installing mmore for development, ensure you have the required system dependencies installed.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 chromium-browser libnss3 \
  libgconf-2-4 libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice \
  libpango-1.0-0 libpangoft2-1.0-0 weasyprint
```

```{note}
On Ubuntu 24.04, replace `libasound2` with `libasound2t64`.

You may also need to add the Ubuntu 20.04 focal repository to access some packages, for example by creating `/etc/apt/sources.list.d/mmore.list` with:

`deb http://cz.archive.ubuntu.com/ubuntu focal main universe`
```

#### macOS

```bash
brew update
brew install ffmpeg chromium gtk+3 pango cairo \
  gobject-introspection libffi pkg-config libx11 libxi \
  libxrandr libxcomposite libxcursor libxdamage libxext \
  libxrender libasound2 atk libreoffice weasyprint
```

If `weasyprint` fails to find GTK or Cairo, also run:

```bash
brew install cairo pango gdk-pixbuf libffi
uv pip install weasyprint
```

### Installing mmore for development

#### 1. Clone the repository

```bash
git clone https://github.com/EPFLiGHT/mmore.git
cd mmore
```

#### 2. Create a virtual environment and install dependencies

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[all,cpu,dev]"
```

```{note}
For **GPU (CUDA 12.6)**, replace `cpu` with `cu126`, for example:

`uv pip install -e ".[all,cu126,dev]"`
```

```{note}
For a **partial install**, replace `all` with only the stages you need, for example:

`uv pip install -e ".[rag,cpu,dev]"`

Available stages are: `process`, `index`, `rag`, and `api`.
```

```{important}
This package requires many large dependencies and a dependency override, so it should be installed with `uv` rather than plain `pip`.

See the [uv guide](../advanced_usage/uv.md) for more information.
```

### 🧹 Code quality tools

mmore uses several tools to maintain code quality and consistency.

#### Pre-commit hooks

We use `pre-commit` to automatically run code formatters and linters before each commit.

#### Setup

##### 1. Install pre-commit 

```bash
uv pip install pre-commit
```

##### 2. Set up the git hook scripts

```bash
pre-commit install
```

##### 3. Run the checks manually
Optional but recommended before your first commit. 

```bash
pre-commit run --all-files
```

#### Configured Hooks

The pre-commit configuration runs `ruff`, a code formatter for consistent style.

### Type Checking

We use `pyright` for static type checking.  
Please ensure your pull requests are type-checked before submission.

To run type checking manually:

```bash
pyright
```

## 🤝 Contributing Guidelines

We welcome contributions! Here's how you can help:

### Reporting Issues

- **Bug reports**: open an issue with a clear description, steps to reproduce, and expected vs. actual behavior
- **Feature requests**: open an issue describing the feature, its use case, and potential implementation approach
- Check the [Issues](https://github.com/EPFLiGHT/mmore/issues) page for ongoing work

### Code Contributions

1. **Fork the repository** and create a new branch for your feature/fix
2. **Write clear, documented code** following the existing style
3. **Add tests** if applicable
4. **Ensure all pre-commit hooks pass**
5. **Run type checking** with `pyright`
6. **Submit a Pull Request** with a clear description

## 🗂️ Project Structure

mmore/
├── mmore/
│   ├── process/          # Document processing pipeline
│   │   ├── processors/   # Individual file type processors
│   │   └── ...
│   ├── postprocess/      # Post-processing utilities
│   ├── index/            # Indexing and vector DB
│   ├── rag/              # RAG implementation
│   └── type/             # Type definitions and data models
├── docs/                 # Documentation
├── examples/             # Example configurations and data
├── tests/                # Test suite
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md

### Key Modules
- **`mmore.process`**: Handles extraction from various file formats
- **`mmore.index`**: Manages hybrid dense+sparse indexing with Milvus
- **`mmore.rag`**: RAG system with LangChain integration
- **`mmore.type`**: Core data structures like `MultimodalSample`

## 🧪 Testing

### Running tests in the terminal

```bash
pytest tests/
```

### GPU tests

Tests requiring a CUDA GPU are marked `@pytest.mark.gpu` and **skipped by
default**. Pass `--gpu` to run them:

```bash
pytest --gpu          # full suite, including GPU tests
pytest --gpu -m gpu   # only the GPU-marked tests
```

To mark a new GPU-only test:

```python
import pytest

@pytest.mark.gpu
def test_something_on_gpu():
    ...
```

### Writing tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Cover edge cases and error conditions
- Mock external dependencies when appropriate
- Mark GPU-only tests with `@pytest.mark.gpu` (see above)

## 🔀 Pull Request Process

1. **Update documentation** if you're adding new features
2. **Add examples** for new functionality
3. **Ensure all tests pass** and pre-commit hooks succeed
4. **Update the changelog** if applicable
5. **Request review** from maintainers

### PR checklist

- [ ] Code follows project style guidelines
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Type checking passes (`pyright`)
- [ ] Tests are added or updated as needed
- [ ] Documentation is updated
- [ ] Examples are provided for new features
- [ ] Commit messages are clear and descriptive

## 🖥️ Interactive TUI

mmore ships with a Terminal UI that wraps the CLI commands behind guided menus and config wizards. Useful for trying the pipeline without writing YAML by hand.

Launch it from a project working directory:

```bash
mmore tui
```

From the main menu you can:

- **Run a single command** — pick any stage (`process`, `postprocess`, `index`, `retrieve`, `rag`, `ragcli`, `websearch`), then either select an existing YAML, generate one through a guided wizard, or type a path manually. Generated configs are written to `./tui-configs/` and validated against the stage's dataclass before running.
- **Run full pipeline** — chains `process → postprocess → index` using existing configs.
- **Build a full pipeline config (guided wizard)** — walks through the three stages in order, wiring the postprocess output JSONL into the index config automatically.
- **Chat with indexed documents** — shortcut to `ragcli`.

Stages whose extras are missing are disabled in the menu with an install hint (e.g. `uv sync --extra rag --extra cpu`). Press `Ctrl-C` inside any sub-flow to cancel back to the main menu; press it again at the main menu to quit.

## 💡 Development tips

### Working with `uv`

- Use `uv pip` instead of `pip` for all package installations
- The project uses dependency overrides that are handled automatically by `uv`
- See the `uv` tutorial for more details

### Verbose / debug output (`MMORE_VERBOSE`)

By default, pipelines show a clean, single progress line per step and hide noisy
third-party output and debug logs.

Debugging a stuck or failing document? Set `MMORE_VERBOSE=1` to see everything (
`DEBUG` logs, all third-party output, etc...):

```bash
MMORE_VERBOSE=1 mmore process --config_file examples/process/config.yaml
```

## ❓ Questions

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Reach out to the maintainers
- Check existing issues for similar questions

Thank you for contributing to mmore! 🎉