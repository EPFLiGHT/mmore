"""
Utilities for loading ColVision models

Provides automatic model/processor class resolution from a HuggingFace model name.
"""

import logging
import re
from typing import Tuple, Type

import torch

from ..ux import loading_model

logger = logging.getLogger(__name__)

# Patterns checked in order — more specific patterns first.
_MODEL_REGISTRY = [
    (r"colqwen2\.5|colqwen2_5", "ColQwen2_5", "ColQwen2_5_Processor"),
    (r"colqwen2", "ColQwen2", "ColQwen2Processor"),
    (r"colgemma|colnetra", "ColGemma3", "ColGemmaProcessor3"),
    (r"colsmol|colidefics3", "ColIdefics3", "ColIdefics3Processor"),
    (r"colpali", "ColPali", "ColPaliProcessor"),
]

SUPPORTED_MODELS = {
    "ColPali": ["vidore/colpali-v1.2", "vidore/colpali-v1.3"],
    "ColQwen2": ["vidore/colqwen2-v0.1", "vidore/colqwen2-v1.0"],
    "ColQwen2.5": ["vidore/colqwen2.5-v0.1", "vidore/colqwen2.5-v0.2"],
    "ColGemma3": ["Cognitive-Lab/ColNetraEmbed"],
    "ColSmol": ["vidore/colSmol-256M", "vidore/colSmol-500M"],
}


def get_device() -> str:
    """Select the available device for model inference."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_device_cache(device: str) -> None:
    """Free cached memory for the active accelerator, if any."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def resolve_model_classes(model_name: str) -> Tuple[Type, Type]:
    """
    Resolve the model and processor classes from the HuggingFace model name.

    Returns:
        Tuple of (ModelClass, ProcessorClass)

    Raises:
        ValueError: if the model name does not match any known pattern.
    """
    try:
        import colpali_engine.models as models_module
    except ImportError as e:
        raise ImportError(
            "colpali_engine is required for ColVision. Install with: pip install mmore[colvision]"
        ) from e

    name_lower = model_name.lower()

    for pattern, model_cls_name, proc_cls_name in _MODEL_REGISTRY:
        if re.search(pattern, name_lower):
            model_cls = getattr(models_module, model_cls_name)
            proc_cls = getattr(models_module, proc_cls_name)
            logger.info(
                f"Resolved model '{model_name}' → {model_cls_name} + {proc_cls_name}"
            )
            return model_cls, proc_cls

    supported_examples = {
        fam: examples[0] for fam, examples in SUPPORTED_MODELS.items()
    }
    raise ValueError(
        f"Unknown model '{model_name}'. Supported families: "
        f"{list(SUPPORTED_MODELS.keys())}. Examples: {supported_examples}"
    )


def load_model_and_processor(
    model_name: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a ColVision model and its processor, resolved from the model name.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    model_cls, proc_cls = resolve_model_classes(model_name)
    logger.debug(f"Loading model: {model_name} ({model_cls.__name__})")
    with loading_model(model_name):
        model = model_cls.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device
        ).eval()
        processor = proc_cls.from_pretrained(model_name)
    return model, processor
