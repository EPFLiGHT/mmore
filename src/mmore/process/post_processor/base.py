import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...type import MultimodalSample
from ...ux import progress
from ..utils import save_samples


@dataclass
class BasePostProcessorConfig:
    type: str
    name: Optional[str] = None
    args: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name is None:
            self.name = self.type


class BasePostProcessor(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __call__(
        self, sample: MultimodalSample, **kwargs
    ) -> MultimodalSample | List[MultimodalSample]:
        return self.process(sample, **kwargs)

    @abstractmethod
    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            List[MultimodalSample]: The processed sample(s).
        """
        pass

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
            tmp_save_path: if provided, intermediate results will be saved to this path every 100 samples
            kwargs: additional arguments to pass to the process method

        Returns: a list of processed samples
        """
        res = []
        if tmp_save_path:
            # Clear the file if it exists
            open(tmp_save_path, "w").close()

        current_batch = []
        bar = progress(samples, desc=self.name, unit="sample")
        for s in bar:
            bar.set_postfix_str(os.path.basename(s.metadata.file_path or ""))
            new = self.process(s, **kwargs)
            current_batch += new

            if len(current_batch) >= save_every:
                if tmp_save_path:
                    save_samples(current_batch, tmp_save_path, append_mode=True)
                res += current_batch
                current_batch = []

        if current_batch:
            if tmp_save_path:
                save_samples(current_batch, tmp_save_path, append_mode=True)

            res += current_batch

        return res
