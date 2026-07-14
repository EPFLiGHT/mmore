import logging
import os
from dataclasses import dataclass
from datetime import datetime
from operator import itemgetter
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union, cast

import torch
import torch.multiprocessing as mp
from dask.distributed import Client, as_completed
from tqdm import tqdm

from ..type import MultimodalSample
from ..ux import init_worker
from .crawler import DispatcherReadyResult, FileDescriptor, URLDescriptor
from .execution_state import ExecutionState
from .processors.base import (
    AutoProcessor,
    Processor,
    ProcessorConfig,
    ProcessorRegistry,
)
from .processors.url_processor import URLProcessor

logger = logging.getLogger(__name__)


class ComputeDescriptor:
    @staticmethod
    def get_desc():
        num_gpus = 0
        gpu_size = None

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_size = torch.cuda.get_device_properties(0).total_memory
                # All GPUs are assumed to have the same size
                logging.info(
                    f"Detected {num_gpus} GPUs with {gpu_size} bytes of memory."
                )

        return {
            "num_gpus": num_gpus,
            "gpu_size": gpu_size,
        }


@dataclass
class DispatcherConfig:
    """
    A configuration class for the dispatcher.

    Save the results to the output path.
    Following structure is used:

    output_path
    ├── processors
    |   ├── Processor_type_1
    |   |   └── results.jsonl
    |   ├── Processor_type_2
    |   |   └── results.jsonl
    |   ├── ...
    |
    └── merged
        └── merged_results.jsonl

    """

    output_path: str
    use_fast_processors: bool = True
    distributed: bool = False
    scheduler_file: Optional[str] = None
    processor_config: Optional[Dict] = None
    process_batch_sizes: Optional[List[Dict[str, float]]] = None
    batch_multiplier: int = 1
    extract_images: bool = False
    # When set, processing is pinned to this single device (used by the indexer
    # API to run one job per GPU). None keeps the default behavior.
    device: Optional[str] = None

    def __post_init__(self):
        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def from_dict(config: Dict) -> "DispatcherConfig":
        """Create a DispatcherConfig object from a dictionary."""
        return DispatcherConfig(
            output_path=config["output_path"],
            use_fast_processors=config.get("use_fast_processors", True),
            distributed=config.get("distributed", False),
            scheduler_file=config.get("scheduler_file"),
            processor_config=config.get("processor_config"),
            process_batch_sizes=config.get("process_batch_sizes"),
            batch_multiplier=config.get("batch_multiplier", 1),
            extract_images=config.get("extract_images", False),
        )

    @staticmethod
    def from_yaml(yaml_path: str):
        import yaml

        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
            return DispatcherConfig.from_dict(config)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"[Dispatcher] Error processing file {yaml_path}")
            raise e

    def to_dict(self) -> Dict:
        """Convert the DispatcherConfig object to a dictionary."""
        return {
            "use_fast_processors": self.use_fast_processors,
            "distributed": self.distributed,
            "scheduler_file": self.scheduler_file,
            "output_path": self.output_path,
            "processor_config": self.processor_config,
            "process_batch_sizes": self.process_batch_sizes,
            "batch_multiplier": self.batch_multiplier,
            "extract_images": self.extract_images,
        }

    def __str__(self) -> str:
        """Return a string representation of the DispatcherConfig object."""
        return (
            f"DispatcherConfig("
            f"use_fast_processors={self.use_fast_processors}, "
            f"distributed={self.distributed}, "
            f"scheduler_file={self.scheduler_file}, "
            f"output_path={self.output_path}, "
            f"processor_config={self.processor_config}, "
            f"process_batch_sizes={self.process_batch_sizes}, "
            f"batch_multiplier={self.batch_multiplier}"
            f"extract_images={self.extract_images}"
            f")"
        )


class _LazyPool:
    """A multiprocessing pool created on first map() call.

    Processors that override process_batch (e.g. PDF, media) never call map(), so a
    job that only uses them never spawns any worker.
    """

    def __init__(self, processes: int):
        self._processes = processes
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            logger.debug(f"Initializing shared pool with {self._processes} workers...")
            self._pool = mp.Pool(processes=self._processes, initializer=init_worker)
        return self._pool

    def map(self, func, iterable):
        return self._ensure_pool().map(func, iterable)

    def imap(self, func, iterable):
        return self._ensure_pool().imap(func, iterable)

    def close(self):
        if self._pool is not None:
            self._pool.close()

    def join(self):
        if self._pool is not None:
            self._pool.join()


class Dispatcher:
    """
    Takes a converted crawl result and dispatches it to the appropriate processor.
    """

    def __init__(
        self,
        result: DispatcherReadyResult,
        config: DispatcherConfig,
        start_cluster=False,
    ):
        self.result = result
        self.config = config
        self.start_cluster = start_cluster
        self.intermediate_map = {}

    def _bucket_files(self) -> None:
        """
        Categorize files and URLs into the appropriate processors.
        """

        processor_map = {
            processor: [] for processor in ProcessorRegistry.get_processors()
        }

        for file_path_list in self.result.file_paths.values():
            for file in file_path_list:
                processor = AutoProcessor.from_file(file)
                logger.debug(
                    f"Assigned file {file.file_path} to processor: {processor}"
                )
                processor_map[processor].append(file)

        url_processor = URLProcessor
        processor_map[url_processor].extend(self.result.urls)

        self.intermediate_map = processor_map

    def _dispatch_local(
        self, task_lists: List[Tuple[Type[Processor], List[FileDescriptor]]]
    ) -> Iterator[List[MultimodalSample]]:
        """
        Dispatches the tasks locally.
        """
        ExecutionState.initialize(distributed_mode=False)
        processor_configs = self.config.processor_config or {}

        instantiated_processors: Dict[Type[Processor], Processor] = {}

        global_pool = _LazyPool(os.cpu_count() or 1)

        try:
            for processor_type, files in task_lists:
                if processor_type not in instantiated_processors:
                    processor_config = processor_configs.get(
                        processor_type.__name__, []
                    )

                    # Might need to check that the list isnt empty
                    if processor_config:
                        processor_config = {
                            list(d.keys())[0]: list(d.values())[0]
                            for d in processor_config
                        }
                    else:
                        processor_config = {}

                    processor_config["output_path"] = self.config.output_path
                    if self.config.device is not None:
                        processor_config["device"] = self.config.device

                    full_config = ProcessorConfig(
                        extract_images=self.config.extract_images,
                        custom_config=processor_config,
                    )

                    logger.debug(f"Initializing processor: {processor_type.__name__}")
                    new_proc_instance = processor_type(full_config)
                    new_proc_instance.set_shared_pool(global_pool)
                    instantiated_processors[processor_type] = new_proc_instance

                proc_instance = instantiated_processors[processor_type]

                logger.debug(
                    f"Processing batch of {len(files)} files with {proc_instance.__class__.__name__}"
                )

                res = proc_instance(
                    cast(List[Union[FileDescriptor, URLDescriptor]], files),
                    self.config.use_fast_processors,
                )
                self.save_individual_processor_results(res, processor_type.__name__)
                yield res
        finally:
            logger.debug("Closing shared global pool")
            global_pool.close()
            global_pool.join()

    def _dispatch_distributed(
        self, task_lists: List[Tuple[Type[Processor], List[FileDescriptor]]]
    ) -> List[List[MultimodalSample]]:
        kwargs = {}
        if self.config.scheduler_file:
            absolute_scheduler_path = os.path.join(
                os.getcwd(), self.config.scheduler_file
            )
            if not os.path.exists(absolute_scheduler_path):
                logger.error(f"Scheduler file {absolute_scheduler_path} does not exist")
            kwargs["scheduler_file"] = absolute_scheduler_path

        client = Client(**kwargs)
        ExecutionState.initialize(distributed_mode=True, client=client)

        futures = []
        processor_configs = self.config.processor_config or {}

        for processor_type, files in task_lists:
            processor_config = processor_configs.get(processor_type.__name__, [])
            if processor_config:
                processor_config = {
                    list(d.keys())[0]: list(d.values())[0] for d in processor_config
                }
            else:
                processor_config = {}
            processor_config["output_path"] = self.config.output_path

            logger.info(
                f"Dispatching in distributed (to some worker) {len(files)} files to {processor_type.__name__}"
            )

            processor_config = ProcessorConfig(
                extract_images=self.config.extract_images,
                custom_config=processor_config,
            )

            def process_files(
                files, processor_config, processor_name, processor_class, use_fast
            ) -> Tuple[List[MultimodalSample], str]:
                client = Client(**kwargs)
                if ExecutionState._use_dask is None:
                    ExecutionState.initialize(distributed_mode=True, client=client)

                worker_count = os.cpu_count() or 1
                task_pool = mp.Pool(processes=worker_count, initializer=init_worker)

                try:
                    proc_instance = processor_class(processor_config)
                    proc_instance.set_shared_pool(task_pool)
                    results = proc_instance(files, use_fast)

                    return results, processor_name

                finally:
                    task_pool.close()
                    task_pool.join()

            try:
                future = client.submit(
                    process_files,
                    files,
                    processor_config,
                    processor_type.__name__,
                    processor_type,
                    self.config.use_fast_processors,
                )
                futures.append(future)
            except Exception as e:
                logger.error(
                    f"Error dispatching task to {processor_type.__name__}: {e}"
                )

        results = []
        for future, (result, processor_name) in tqdm(
            as_completed(futures, with_results=True), total=len(futures)
        ):
            try:
                results.append(result)
                self.save_individual_processor_results(result, processor_name)
            except Exception as e:
                logger.error(f"Error gathering result: {e}")

        return results

    def _clear_per_processor_results(self) -> None:
        """Clear per-processor result JSONL files.
        This is needed because :meth:`MultimodalSample.to_jsonl` uses append by default."""
        if not self.config.output_path:
            return
        processors_dir = os.path.join(self.config.output_path, "processors")
        if not os.path.isdir(processors_dir):
            return
        for processor_name in os.listdir(processors_dir):
            results_path = os.path.join(processors_dir, processor_name, "results.jsonl")
            if os.path.exists(results_path):
                os.remove(results_path)

    def dispatch(self) -> List[List[MultimodalSample]]:
        """
        Dispatches the result to the appropriate processor.
        """
        self._clear_per_processor_results()

        def batch_list(
            lst: List, obj_batch_size: int, processor: Type[Processor]
        ) -> List[List]:
            """
            Creates optimized batches using best-fit decreasing algorithm.

            Args:
                lst: List of objects to batch
                obj_batch_size: Maximum allowed batch size
                processor: Processor that can determine object sizes

            Returns:
                List of batched objects optimized for size
            """
            # Create (object, size) tuples and sort by size descending
            items = [(obj, processor.get_file_len(obj)) for obj in lst]
            items = [item for item in items if item[1] != -1]

            items.sort(key=itemgetter(1), reverse=True)

            batches = [[]]  # List of object lists
            batch_sizes = [0]  # Parallel array tracking batch sizes

            for obj, size in items:
                best_fit_idx = -1
                min_remaining = obj_batch_size

                # Find best fitting-batch
                for i, batch_size in enumerate(batch_sizes):
                    remaining = obj_batch_size - (batch_size + size)
                    if 0 <= remaining < min_remaining:
                        min_remaining = remaining
                        best_fit_idx = i

                if best_fit_idx >= 0:
                    batches[best_fit_idx].append(obj)
                    batch_sizes[best_fit_idx] += size
                else:
                    batches.append([obj])
                    batch_sizes.append(size)

            return batches

        self._bucket_files()

        batch_sizes = self.config.process_batch_sizes or {}
        batch_sizes = {list(d.keys())[0]: int(list(d.values())[0]) for d in batch_sizes}

        task_lists = []
        for processor, file_list in self.intermediate_map.items():
            if len(file_list) > 0:
                batched_files = batch_list(
                    file_list,
                    self.config.batch_multiplier
                    * batch_sizes.get(processor.__name__, 100),
                    processor,
                )
                task_lists.extend([(processor, batch) for batch in batched_files])
        results = []
        if self.config.distributed:
            results = self._dispatch_distributed(task_lists)
        else:
            results = list(self._dispatch_local(task_lists))

        ExecutionState.shutdown()

        return results

    def __call__(self) -> List[List[MultimodalSample]]:
        return self.dispatch()

    def save_individual_processor_results(
        self, results: List[MultimodalSample], cls_name
    ) -> None:
        if not self.config.output_path:
            return

        processed_at = datetime.now().isoformat()
        for sample in results:
            sample.metadata.processed_at = processed_at
            sample.metadata.processor_type = cls_name

        processor_output_path = os.path.join(
            self.config.output_path, "processors", cls_name
        )
        os.makedirs(processor_output_path, exist_ok=True)
        output_file = os.path.join(processor_output_path, "results.jsonl")
        MultimodalSample.to_jsonl(output_file, results)

        logger.debug(f"Results saved to {output_file}")
