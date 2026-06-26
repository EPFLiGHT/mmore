from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ....process.post_processor import BasePostProcessor
from ....type import MultimodalSample
from ....ux import progress


@dataclass
class BaseFilterConfig:
    type: str
    name: Optional[str] = None
    args: Any = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.type


class BaseFilter(BasePostProcessor):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @abstractmethod
    def filter(self, sample: MultimodalSample) -> bool | Tuple[bool, str]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        pass

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        res = self.filter(sample)
        if res:
            return [sample]
        else:
            return []

    def batch_filter(
        self, batch: List[MultimodalSample]
    ) -> List[bool | Tuple[bool, str]]:
        """
        Overwrite this method to implement batched filtering. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the filter result for each document

        """
        return list(map(self.filter, progress(batch, desc=self.name, unit="sample")))

    def batch_process(
        self,
        samples: List[MultimodalSample],
        tmp_save_path: Optional[str] = None,
        save_every: int = 100,
        **kwargs,
    ) -> List[MultimodalSample]:
        """
        Process a batch of samples.
        Args:
            samples: a list of samples to process
            tmp_save_path: path to save intermediate results (inherited from the base class, but useless for filtering)
            save_every: frequency of saving intermediate results (inherited from the base class, but useless for filtering)
            kwargs: additional arguments to pass to the process method

        Returns: a list of processed samples
        """
        res = self.batch_filter(samples)
        return [s for s, r in zip(samples, res) if r]
