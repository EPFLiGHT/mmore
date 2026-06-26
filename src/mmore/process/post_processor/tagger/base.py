from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ....type import MultimodalSample
from ....ux import progress
from .. import BasePostProcessor


@dataclass
class BaseTaggerConfig:
    type: str
    name: Optional[str] = None
    metadata_key: Optional[str] = None
    args: Any = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.name is None:
            self.name = self.type
        if self.metadata_key is None:
            self.metadata_key = self.type


class BaseTagger(BasePostProcessor):
    name: str
    metadata_key: str

    def __init__(self, name: str, metadata_key: str):
        self.name = name
        self.metadata_key = metadata_key

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @abstractmethod
    def tag(self, sample: MultimodalSample) -> Any:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        pass

    def batch_tag(self, batch: List[MultimodalSample]) -> List[Any]:
        """
        Overwrite this method to implement batched filtering. Batches have size `self.batch_size`, except possibly the last one.
        Args:
            batch: a list of Document to process

        Returns: a list, the same size as `batch`, containing the filter result for each document

        """
        return list(map(self.tag, progress(batch, desc=self.name, unit="sample")))

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        tag = self.tag(sample)
        sample.metadata.extra[self.metadata_key] = tag
        return [sample]
