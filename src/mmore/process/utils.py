"""
Utility functions for processing files, images, PDFs, and text.
These functions can be used across various processors for data extraction,
cleaning, splitting, and aggregation.
"""

import json
import logging
import os
from typing import List

import numpy as np
from cleantext import clean
from PIL import Image

from ..type import MultimodalSample

logger = logging.getLogger(__name__)


def jsonl_path(path: str, filename: str = "final.jsonl") -> str:
    if path.endswith(".jsonl"):
        return path
    return os.path.normpath(os.path.join(path, filename))


def clean_text(text: str) -> str:
    """
    Clean a given text using `cleantext` library. https://pypi.org/project/clean-text/

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    return clean(
        text=text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
        no_urls=False,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="This is a URL",
        replace_with_email="email@email.com",
        replace_with_phone_number="",
        replace_with_number="123",
        replace_with_digit="0",
        replace_with_currency_symbol="$",
        lang="en",
    )


def clean_image(
    image: Image.Image, min_width=512, min_height=512, variance_threshold=100
) -> bool:
    """
    Validates an image based on size and variance (whether its one-colored).

    Args:
        image (PIL.Image.Image): The image to validate.
        min_width (int, optional): The minimum width an image must have to be considered valid. Defaults to 512.
        min_height (int, optional): The minimum height an image must have to be considered valid. Defaults to 512.
        variance_threshold (int, optional): The minimum variance in pixel intensity required. Images with lower variance are considered "empty". Defaults to 100.

    Returns:
        bool: True if the image meets all criteria, False otherwise.
    """
    if image is None:
        return False

    w, h = image.size

    # Check size criteria
    if w < min_width or h < min_height:
        return False

    # Check variance threshold
    gray = image.convert("L")
    arr = np.array(gray)
    variance = arr.var()
    if variance < variance_threshold:
        return False

    return True


def save_samples(
    samples: List[MultimodalSample], path: str, append_mode: bool = False
) -> None:
    """
    Save multimodal samples to a JSONL file.

    Args:
        samples (List[MultimodalSample]): List of multimodal samples.
        path (str): Path to save the samples.
        append_mode (bool, optional): If True, append to the existing file; if False, overwrite it. Defaults to False.
    """
    try:
        mode = "a" if append_mode else "w"
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, mode) as f:
            for result in samples:
                f.write(json.dumps(result.to_dict()) + "\n")
    except OSError as e:
        logger.error("Failed to save samples to %s: %s", path, e)
        raise
    except (TypeError, ValueError) as e:
        logger.error(
            "Failed to serialize sample to JSON when saving to %s: %s", path, e
        )
        raise
    except AttributeError as e:
        logger.error("Invalid sample encountered when saving to %s: %s", path, e)
        raise
    else:
        logger.debug(f"Results saved to {path}")
