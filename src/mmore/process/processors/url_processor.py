import io
import logging
import re

import requests
import trafilatura
from PIL import Image

from ...type import DocumentMetadata, MultimodalSample, URLDescriptor
from ..utils import clean_text
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class URLProcessor(Processor):
    def __init__(self, config=None):
        """
        Initialize the URLProcessor.

        :param config: ProcessorConfig object with configuration settings.
        """
        super().__init__(config=config or ProcessorConfig())
        self.ocr_models = None  # Models will be loaded per process
        self.driver = None  # WebDriver will be initialized per process

    @classmethod
    def accepts(cls, file) -> bool:
        return isinstance(file, URLDescriptor)

    def process_fast(self, file_path: str) -> MultimodalSample:
        try:  # wrap in try because urls can be buggy
            downloaded = trafilatura.fetch_url(file_path)
            if not downloaded:
                raise ValueError(f"Failed to fetch content from URL: {file_path}")
            result = trafilatura.extract(downloaded, include_images=True)
            if not result:
                raise ValueError(f"Failed to extract content from URL: {file_path}")

            embedded_images = []
            # replace all ![] with <attachment>
            all_text = re.sub(r"!\[.*\]\(.*\)", self.config.attachment_tag, result)

            if self.config.extract_images:
                images = re.findall(r"!\[.*\]\(.*\)", result)
            else:
                images = []

            for image in images:
                try:
                    image_url_search = re.search(r"\(.*\)", image)
                    if image_url_search is None:
                        raise Exception("Unable to find a valid URL of the image")

                    image_url = image_url_search.group(0)[1:-1]
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(image_url, headers=headers, timeout=5)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content)).convert("RGB")
                    embedded_images.append(img)

                except Exception as e:
                    logger.error(f"Failed to process image {image}: {e}")

            all_text = [clean_text(all_text)]
            return self.create_sample(
                all_text, embedded_images, DocumentMetadata(file_path=file_path)
            )
        except Exception as e:
            logger.error(f"Failed to process URL {file_path}: {e}")
            return self.create_sample([], [], DocumentMetadata(file_path=file_path))

    def process(self, file_path: str, fast: bool = False) -> MultimodalSample:
        return self.process_fast(file_path)
