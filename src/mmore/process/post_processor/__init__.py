from typing import cast

from ...utils import load_config
from .base import BasePostProcessor, BasePostProcessorConfig
from .filter import FILTER_TYPES, load_filter
from .filter.base import BaseFilterConfig
from .tagger import TAGGER_TYPES, load_tagger
from .tagger.base import BaseTaggerConfig

__all__ = ["BasePostProcessor", "BasePostProcessorConfig", "load_postprocessor"]


def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type in FILTER_TYPES:
        return load_filter(cast(BaseFilterConfig, config))

    elif config.type in TAGGER_TYPES:
        return load_tagger(cast(BaseTaggerConfig, config))

    elif config.type == "chunker":
        from .chunker import MultimodalChunker, MultimodalChunkerConfig

        config_chunk = load_config(config.args, MultimodalChunkerConfig)
        return MultimodalChunker.from_config(config_chunk)

    elif config.type == "ner":
        from .ner import NERecognizer, NERExtractorConfig

        config_ner = load_config(config.args, NERExtractorConfig)
        return NERecognizer.from_config(config_ner)

    elif config.type == "translator":
        from .translator import TranslatorConfig, TranslatorPostProcessor

        config_translator = load_config(config.args, TranslatorConfig)
        return cast(
            BasePostProcessor, TranslatorPostProcessor.from_config(config_translator)
        )

    elif config.type == "metafuse":
        from .metafuse import MetaDataInfusor, MetaDataInfusorConfig

        config_metafuse = load_config(config.args, MetaDataInfusorConfig)
        return cast(BasePostProcessor, MetaDataInfusor.from_config(config_metafuse))

    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")
