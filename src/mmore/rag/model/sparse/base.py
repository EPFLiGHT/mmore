from dataclasses import dataclass

from langchain_milvus.utils.sparse import BaseSparseEmbedding

from ....ux import loading_model
from .splade import SpladeSparseEmbedding

_SPLADE_MODELS = ["naver/splade-cocondenser-selfdistil"]
_names = {"splade": "naver/splade-cocondenser-selfdistil"}
loaders = {"SPLADE": SpladeSparseEmbedding}


@dataclass
class SparseModelConfig:
    model_name: str
    is_multimodal: bool = False

    def __post_init__(self):
        if self.model_name.lower() in _names:
            self.model_name = _names.get(self.model_name, self.model_name)

    @property
    def model_type(self) -> str:
        if self.model_name in _SPLADE_MODELS:
            return "SPLADE"
        else:
            raise NotImplementedError()


class SparseModel(BaseSparseEmbedding):
    @classmethod
    def from_config(
        cls, config: SparseModelConfig, device: str | None = None
    ) -> BaseSparseEmbedding:
        with loading_model(f"the keyword-matching model ({config.model_name})"):
            return loaders.get(config.model_type, SpladeSparseEmbedding)(
                model_name=config.model_name, device=device
            )
