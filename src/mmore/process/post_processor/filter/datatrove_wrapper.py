from typing import Any, Dict, List, Tuple, Union, cast

import nltk
from datatrove.data import Document, Media
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FastTextClassifierFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LambdaFilter,
    LanguageFilter,
    RegexFilter,
    SamplerFilter,
    UnigramLogProbFilter,
    URLFilter,
)
from datatrove.pipeline.filters.base_filter import BaseFilter as DatatroveBaseFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter

from ....type import MultimodalSample
from ....ux import progress
from .base import BaseFilter, BaseFilterConfig

nltk.download("punkt_tab", quiet=True)

FILTERS_MAP = {
    "filter_language": LanguageFilter,
    "filter_gopher-repetition": GopherRepetitionFilter,
    "filter_gopher-quality": GopherQualityFilter,
    "filter_fineweb": FineWebQualityFilter,
    "filter_c4": C4QualityFilter,
    "sampler": SamplerFilter,
    "filter_regex": RegexFilter,
    "filter_fasttext": FastTextClassifierFilter,
    "filter_lambda": LambdaFilter,
    "filter_unigram-logprob": UnigramLogProbFilter,
    "filter_url": URLFilter,
}
DATATROVE_FILTERS = list(FILTERS_MAP.keys())


def load_datatrove_filter(
    filter_name: str, filter_args: Dict[str, Any]
) -> DatatroveBaseFilter:
    if filter_name not in FILTERS_MAP:
        raise ValueError(f"Unsupported filter: {filter_name}")
    if "exclusion_writer" in filter_args and isinstance(
        filter_args["exclusion_writer"], str
    ):
        filter_args["exclusion_writer"] = JsonlWriter(filter_args["exclusion_writer"])
    return FILTERS_MAP[filter_name](**filter_args)


class DatatroveFilter(BaseFilter):
    datatrove_filter: DatatroveBaseFilter

    def __init__(self, name: str, datatrove_filter: DatatroveBaseFilter):
        super().__init__(name)
        self.datatrove_filter = datatrove_filter

    @classmethod
    def from_config(cls, config: BaseFilterConfig) -> "DatatroveFilter":
        datatrove_filter = load_datatrove_filter(config.type, config.args)
        return cls(name=datatrove_filter.name, datatrove_filter=datatrove_filter)

    @staticmethod
    def sample_to_doc(sample: MultimodalSample) -> Document:
        def type_as_int(x):
            return {"image": 0, "video": 1, "audio": 2}[x]

        return Document(
            text=sample.text,
            id=sample.id,
            media=[
                Media(type=type_as_int(modality.type), url=modality.value)
                for modality in sample.modalities
            ],
            metadata=cast(
                Dict[str, Union[str, int, float, bool]],
                sample.metadata.to_dict(),
            ),
        )

    def filter(self, sample: MultimodalSample) -> bool | Tuple[bool, str]:
        """Abstract method for processing a sample.

        Args:
            sample (MultimodalSample): The sample to process.

        Returns:
            bool: Whether the doc should be kept.
            str: If the document must be ignored, the reason.
        """
        # Filter the document
        res = self.datatrove_filter.filter(DatatroveFilter.sample_to_doc(sample))
        if isinstance(res, bool):
            return res
        else:
            return res[0]
        # return self.datatrove_filter.filter(DatatroveFilter.sample_to_doc(sample))

    def batch_filter(self, batch):
        """Abstract method for processing a batch of samples.

        Args:
            batch (List[MultimodalSample]): The batch to process.

        Returns:
            List[bool]: Whether each document should be kept.
        """
        batch = progress(
            [DatatroveFilter.sample_to_doc(sample) for sample in batch],
            desc=self.name,
            unit="sample",
        )
        return self.datatrove_filter.filter_batch(cast(List[Document], batch))
