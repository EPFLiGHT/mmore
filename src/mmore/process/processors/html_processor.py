import io
import logging
import re
from typing import List

import requests
from markdownify import markdownify as md
from PIL import Image

from ...type import DocumentMetadata, FileDescriptor, MultimodalSample
from ..utils import clean_text
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class HTMLProcessor(Processor):
    def __init__(self, config=None):
        """
        A processor for HTML files. Extracts text content and optionally embedded images.

        Attributes:
            files (List[FileDescriptor]): List of HTML files to be processed.
            config (ProcessorConfig): Config for the processor, which includes options such as
                                    the placeholder tag for embedded images (e.g., "<attachment>").
        """
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [".html", ".htm"]

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a single HTML file. Extracts text content and embedded images.

        Args:
            file_path (str): Path to the HTML file.

        Returns:
            MultimodalSample: A dictionary containing processed text and embedded images.
        """

        def _extract_images_from_markdown(markdown_text: str) -> List[Image.Image]:
            """
            Extract images from a markdown string.

            Args:
                markdown_text (str): The markdown string.
                file_path (Optional[str]): Path to the markdown source (to resolve local image paths).

            Returns:
                List[Image.Image]: A list of PIL images.
            """
            image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
            image_paths = image_pattern.findall(markdown_text)

            images = []

            def download_image(url):
                headers = {
                    "User-Agent": "YourAppName/1.0 (your.email@example.com) Python requests"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type:
                    raise ValueError(
                        f"Content at {url} is not an image (type: {content_type})"
                    )

                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                return image

            for src in image_paths:
                try:
                    if src.startswith("http") or src.startswith("//"):
                        url = src if src.startswith("http") else "https:" + src
                        image = download_image(url)

                    else:
                        raise ValueError("can't download static files, sorry")

                    images.append(image)

                except Exception as e:
                    logger.debug(f"Failed to load image {src}: {e}")

            return images

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            logger.error(f"Failed to open HTML file {file_path}: {e}")
            return self.create_sample([], [], DocumentMetadata(file_path=file_path))

        markdown = md(html, heading_style="ATX")

        if self.config.extract_images:
            embedded_images = _extract_images_from_markdown(markdown)

        else:
            embedded_images = []

        # If extract_images is enabled, optionally replace image markdown with a placeholder
        if self.config.extract_images:
            # Replace all image markdown with the placeholder
            markdown = re.sub(r"!\[.*?\]\(.*?\)", self.config.attachment_tag, markdown)

        # Clean the markdown text
        cleaned_markdown = clean_text(markdown).strip()
        all_text = [cleaned_markdown] if cleaned_markdown else []

        return self.create_sample(
            all_text, embedded_images, DocumentMetadata(file_path=file_path)
        )
