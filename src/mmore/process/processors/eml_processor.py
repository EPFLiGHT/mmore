import email
import io
import logging
from email import policy

from PIL import Image

from ...type import DocumentMetadata, FileDescriptor, MultimodalSample
from ..utils import clean_text
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class EMLProcessor(Processor):
    """
    A processor for handling email files (.eml). Extracts email headers, text content, and embedded images.

    Attributes:
        files (List[FileDescriptor]): List of EML files to be processed.
        config (ProcessorConfig): Configuration for the processor, including options such as the
                                   placeholder tag for embedded images (e.g., "<attachment>").
    """

    def __init__(self, config=None):
        """
        Args:
            files (List[FileDescriptor]): List of files to process.
            config (ProcessorConfig, optional): Configuration for the processor. Defaults to None.
        """
        super().__init__(config=config or ProcessorConfig())

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        """
        Args:
            file (FileDescriptor): The file descriptor to check.

        Returns:
            bool: True if the file is an EML file, False otherwise.
        """
        return file.file_extension.lower() in [".eml"]

    def process(self, file_path: str) -> MultimodalSample:
        """
        Process a single EML file. Extracts text content, email headers, and embedded images.

        Args:
            file_path (str): Path to the EML file.

        Returns:
            dict: A dictionary containing processed text, embedded images, and metadata.

        The method parses the EML file, extracts email headers, text content, and embedded images.
        Embedded images are replaced with a placeholder tag from the processor configuration.
        """

        try:
            with open(file_path, "rb") as f:
                msg = email.message_from_bytes(f.read(), policy=policy.default)
        except Exception as e:
            logger.error(f"Failed to open EML file {file_path}: {e}")
            return self.create_sample([], [], DocumentMetadata(file_path=file_path))

        all_text = []
        embedded_images = []

        # extract email headers
        headers = [
            f"From: {msg.get('From', '')}",
            f"To: {msg.get('To', '')}",
            f"Subject: {msg.get('Subject', '')}",
            f"Date: {msg.get('Date', '')}",
        ]
        all_text.extend([clean_text(header) for header in headers if header])

        for part in msg.walk():
            # extract text
            if part.get_content_type() == "text/plain":
                try:
                    text = part.get_content()
                    cleaned = clean_text(text)
                    if cleaned.strip():
                        all_text.append(cleaned)
                except Exception as e:
                    logger.error(f"Error extracting text from EML: {e}")

            # extract images only if passed argument in config is True
            elif part.get_content_type().startswith("image/"):
                if self.config.extract_images:
                    try:
                        image_data = part.get_payload(decode=True)
                        if isinstance(image_data, bytes):
                            image = Image.open(io.BytesIO(image_data)).convert("RGB")
                        else:
                            raise ValueError(
                                "Image data extracted is not made of bytes"
                            )
                        embedded_images.append(image)
                        all_text.append(
                            self.config.attachment_tag
                        )  # default token is "<attachment>"
                    except Exception as e:
                        logger.error(f"Error extracting image from EML: {e}")
                else:
                    embedded_images = []

        return self.create_sample(
            all_text, embedded_images, DocumentMetadata(file_path=file_path)
        )
