# ⏱️ Profiler

## Overview

mmore includes built-in profiling utilities to help you analyze performance bottlenecks and optimize your pipelines. 

The profiling system uses Python's standard `cProfile` module and provides easy-to-use wrappers for function and block-level profiling.

## 🚀 Quick Start

Profiling is disabled by default to avoid overhead. To enable it, simply set the `MMORE_PROFILING_ENABLED` environment variable to `true` before running any mmore command.

### Linux / macOS / WSL

```bash
export MMORE_PROFILING_ENABLED=true
python -m mmore process --config-file examples/process/config.yaml
```

### Windows PowerShell

```powershell
$env:MMORE_PROFILING_ENABLED="true"
python -m mmore process --config-file examples/process/config.yaml
```

### Windows Command Prompt
```cmd
set MMORE_PROFILING_ENABLED=true
python -m mmore process --config-file examples/process/config.yaml
```

When profiling is enabled, a summary table will be printed to the console after execution, and detailed `.prof` files will be saved to the output directory.

## ⚙️ Configuration

You can control the profiling behavior using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MMORE_PROFILING_ENABLED` | Set to `true` to enable profiling globally. | `false` |
| `MMORE_PROFILING_OUTPUT_DIR` | Directory where `.prof` files will be saved. | `./profiling_output` |
| `MMORE_PROFILING_SORT_BY` | Metric to sort results by (`cumulative`, `time`, `calls`, `pcalls`). | `cumulative` |
| `MMORE_PROFILING_MAX_RESULTS` | Number of rows to display in the console summary. | `50` |

## 🔍 Analyzing Results

### Console Output
After a profiled run completes, you will see a table like this in your terminal:
```text
         123456 function calls in 1.234 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.234    1.234 src/mmore/run_process.py:43(process)
      ...
```
- **ncalls**: number of calls
- **tottime**: total time spent in the function (excluding sub-calls)
- **cumtime**: cumulative time spent in the function (including sub-calls)

### Visualizing `.prof` Files
For deeper analysis, use tools like `snakeviz` or `tuna` to visualize the generated `.prof` files found in your `MMORE_PROFILING_OUTPUT_DIR`.

#### Using SnakeViz
```bash
uv pip install snakeviz
snakeviz ./profiling_output/process_1700000000.prof
```

## 💻 Programmatic Usage

If you are developing custom components or scripts, you can use the profiling tools directly from `mmore.profiler`.

### 1. Decorator
Profile a specific function using the `@profile_function` decorator.

The `sort_by` argument is passed directly to Python's profiling statistics sorter. Common values include `cumulative`, `time`, `calls`, and `pcalls`.

```python
from mmore.profiler import profile_function

@profile_function(sort_by="time", max_results=20)
def my_heavy_function():
    # ... code ...
    pass
```

### 2. Context Manager
Profile a specific block of code using the `profile_context` context manager.

```python
from mmore.profiler import profile_context

def complex_operation():
    # ... setup ...
    
    with profile_context("critical_section"):
        # ... critical code to profile ...
        pass
```

### 3. Manual Control
Use the `Profiler` class for manual start/stop control.

```python
from mmore.profiler import Profiler

profiler = Profiler(enabled=True)
profiler.start()
# ... code to profile ...
profiler.stop(name="manual_session")
```

### 4. Simple Timing
If you only need to measure execution time (wall clock), use `time_function` or `time_context`.

```python
from mmore.profiler import time_function, time_context

@time_function()
def quick_check():
    pass

with time_context("database_query"):
    # ... query ...
    pass
```


## See also

- [Process](../getting_started/process.md)
- [Distributed processing](distributed_processing.md)
