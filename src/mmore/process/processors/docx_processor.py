import io
import logging
import mimetypes
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Optional, cast

import mammoth
from mammoth.documents import Image as m_Image
from markdownify import markdownify
from PIL import Image

from ...type import (
    DocumentMetadata,
    FileDescriptor,
    MultimodalRawInput,
    MultimodalSample,
)
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class DOCXProcessor(Processor):
    def __init__(self, config=None):
        """
        A processor for handling Microsoft Word documents (.docx). Extracts text content and embedded images.

        Attributes:
            files (List[FileDescriptor]): List of DOCX files to be processed.
            config (ProcessorConfig): Config for the processor, which includes options such as
                                    the placeholder tag for embedded images (e.g., "<attachment>").
        """
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is a DOCX file, False otherwise.
        """
        return file.file_extension.lower() in [".docx"]

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a single DOCX file. Extracts text content and embedded images.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            MultimodalSample: A sample containing the extracted text and images.

        The method parses the DOCX file, cleans and extracts textual content from paragraphs,
        and extracts embedded images. Images in the paragraphs are replaced with a placeholder tag
        defined in the processor configuration (e.g., "<attachment>").
        """

        all_images = []

        def _convert_image(image: m_Image) -> Dict[str, Any]:
            if not self.config.extract_images:
                return {"src": ""}

            content_type = cast(Optional[str], image.content_type)

            with cast(Callable[[], BinaryIO], image.open)() as image_bytes:
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes.read()))

                    if content_type is None:
                        raise ValueError("Invalid content type")

                    # Generate unique image path and save the image there
                    image_path = Path(
                        os.path.join(str(image_output_dir), str(uuid.uuid4()))
                    )

                    extension = mimetypes.guess_extension(content_type)
                    if extension is None:
                        raise ValueError(
                            "Unable to determine the extension of the image"
                        )

                    image_path = image_path.with_suffix(extension)

                    pil_image.save(image_path)
                    logger.debug(f"Saving image to {image_path}")
                    all_images.append(
                        MultimodalRawInput(type="image", value=str(image_path))
                    )

                    return {"src": "", "alt": self.config.attachment_tag}

                except Exception as e:
                    logger.debug(
                        f"Failed to load image with MIME type {content_type}: {e}"
                    )
                    return {"src": "", "alt": ""}

        image_output_dir = self.config.custom_config.get("image_output_dir", None)

        # If no image_output_dir is specified then create a temporary output dir
        if image_output_dir is None:
            image_output_dir = tempfile.mkdtemp(prefix="mmore_docx_")
            logger.debug(f"Saving files in {image_output_dir}")

        try:
            with open(file_path, "rb") as docx_fileobj:
                result = mammoth.convert_to_html(
                    docx_fileobj,
                    convert_image=mammoth.images.img_element(_convert_image),
                )

        except Exception as e:
            logger.warning(f"Failed to convert {file_path}: {e}")
            return self.create_sample([], [], DocumentMetadata(file_path=file_path))

        markdown = markdownify(result.value)

        sample = MultimodalSample(
            text=re.sub(r"!\[<([^>]+)>\]\(\)", r"<\1>", markdown),
            modalities=all_images,
            metadata=DocumentMetadata(file_path=file_path),
        )

        return sample
