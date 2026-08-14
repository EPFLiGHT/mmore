import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from ...type import MultimodalSample
from ..incremental import (
    is_reusable_postprocess,
    load_previous_postprocess_results,
    merge_results,
)
from ..utils import jsonl_path, save_samples
from . import BasePostProcessor, BasePostProcessorConfig, load_postprocessor

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    output_path: str
    save_each_step: bool = False
    save_every: int = 100

    def __post_init__(self):
        dirname = os.path.dirname(self.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)


@dataclass
class PPPipelineConfig:
    pp_modules: List[BasePostProcessorConfig]
    output: OutputConfig
    previous_results: Optional[str] = None


class PPPipeline:
    def __init__(
        self,
        *processors: BasePostProcessor,
        previous_results_path: Optional[str] = None,
        output_config: Optional[OutputConfig] = None,
    ):
        if output_config is None:
            output_config = OutputConfig(output_path="./results")

        self.post_processors = processors
        self.previous_results_path = previous_results_path
        self.output_config = output_config

        # Log the pipeline
        self._log_plan()

    def __add__(self, other):
        return PPPipeline(
            *self.post_processors,
            *other.post_processors,
            previous_results_path=self.previous_results_path,
            output_config=self.output_config,
        )

    def _log_plan(self) -> None:
        n = len(self.post_processors)
        steps = " > ".join(p.name for p in self.post_processors)
        logger.debug(f"Pipeline ({n} step{'s' if n != 1 else ''}): {steps}")

    @classmethod
    def from_config(cls, config: PPPipelineConfig):
        post_processors = [
            load_postprocessor(pp_config) for pp_config in config.pp_modules
        ]
        return cls(
            *post_processors,
            previous_results_path=config.previous_results,
            output_config=config.output,
        )

    def __call__(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        return self.run(samples)

    def run(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        """
        Run the post-processing pipeline on a list of multimodal samples.
        The post-processors are applied in sequence.

        Args:
            samples (List[MultimodalSample]): List of multimodal samples.

        Returns:
            List[MultimodalSample]: Post-processed multimodal samples.
        """
        if self.previous_results_path is not None:
            return self._run_incremental(samples)
        return self._run_full(samples)

    def _run_full(self, samples: List[MultimodalSample]) -> List[MultimodalSample]:
        """Run all processors on all samples."""
        input_number = len(samples)
        output_dir = os.path.dirname(self.output_config.output_path) or "."
        for i, processor in enumerate(self.post_processors):
            tmp_save_path = None
            if self.output_config.save_each_step:
                tmp_save_path = os.path.join(
                    output_dir,
                    f"{i + 1}___{processor.name}.jsonl",
                )
            samples = processor.batch_process(
                samples,
                tmp_save_path=tmp_save_path,
                save_every=self.output_config.save_every,
            )

        processed_at = datetime.now().isoformat()
        for sample in samples:
            sample.metadata.processed_at = processed_at

        out_path = jsonl_path(self.output_config.output_path)
        save_samples(samples, out_path)
        logger.debug(
            f"Done: {input_number} docs -> {len(samples)} samples saved to {out_path}"
        )
        return samples

    def _run_incremental(
        self, samples: List[MultimodalSample]
    ) -> List[MultimodalSample]:
        """Run processors only on samples from new/changed source documents."""
        output_dir = os.path.dirname(self.output_config.output_path) or "."

        assert self.previous_results_path is not None and os.path.exists(
            self.previous_results_path
        ), f"Previous results file not found: {self.previous_results_path}"
        previous = load_previous_postprocess_results(self.previous_results_path)

        # Group input samples by file_path
        index: dict[str, MultimodalSample] = {}
        for sample in samples:
            index[sample.metadata.file_path] = sample

        current_file_paths = set(index.keys())

        reusable_file_paths: set[str] = set()
        to_process_file_paths: set[str] = set()
        for fp, sample in index.items():
            input_processed_at = sample.metadata.processed_at
            if input_processed_at is None or not is_reusable_postprocess(
                fp, input_processed_at, previous
            ):
                # When no processed_at on input or not reusable, post process it as a new one
                to_process_file_paths.add(fp)
            else:
                reusable_file_paths.add(fp)

        n_deleted = len(set(previous.keys()) - set(index.keys()))
        logger.debug(
            f"Post-process pipeline: {len(reusable_file_paths)} reused, "
            f"{len(to_process_file_paths)} to process, {n_deleted} deleted"
        )

        # Collect reused samples from previous results
        reused: dict[str, List[MultimodalSample]] = {
            fp: previous[fp] for fp in sorted(reusable_file_paths)
        }

        if not to_process_file_paths:
            merged_samples = merge_results(reused, [], current_file_paths)
            out_path = jsonl_path(self.output_config.output_path)
            save_samples(merged_samples, out_path)
            logger.debug(f"Done: {len(merged_samples)} samples saved to {out_path}")
            return merged_samples

        # Collect samples to process
        samples_to_process = [index[fp] for fp in sorted(to_process_file_paths)]

        # Run through pipeline
        processed = samples_to_process
        for i, processor in enumerate(self.post_processors):
            tmp_save_path = None
            if self.output_config.save_each_step:
                tmp_save_path = os.path.join(
                    output_dir,
                    f"{i + 1}___{processor.name}_incremental.jsonl",
                )
            processed = processor.batch_process(
                processed,
                tmp_save_path=tmp_save_path,
                save_every=self.output_config.save_every,
            )

        # Add processed_at to newly processed samples
        processed_at = datetime.now().isoformat()
        for sample in processed:
            sample.metadata.processed_at = processed_at

        merged_samples = merge_results(reused, processed, current_file_paths)
        out_path = jsonl_path(self.output_config.output_path)
        save_samples(merged_samples, out_path)
        logger.debug(f"Done: {len(merged_samples)} samples saved to {out_path}")
        return merged_samples
