"""PDF fast-mode extraction using pymupdf4llm, opt-in through `processor_selection`.
Reference : https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html

Select it per extension in the dispatcher config:

    dispatcher_config:
      use_fast_processors: true
      processor_selection:
        ".pdf": PDFPyMuPDF4LLMProcessor
      processor_config:
        PDFPyMuPDF4LLMProcessor:
          - image_dpi: 300  # optional, resolution figures are rendered at
"""

import base64
import io
import logging
import re
from typing import List, Optional, Tuple, cast

import pymupdf
import pymupdf4llm
from PIL import Image, UnidentifiedImageError

from ...type import MultimodalSample
from ..utils import clean_image, clean_text
from . import pdf_processor
from .pdf_processor import PDFMetadata

# pymupdf4llm writes images as markdown links, they need to be replaced by attachment tags.
MD_IMG_REGEX = re.compile(r"!\[]\(([^)]*)\)")

DEFAULT_IMAGE_DPI = 300


class PDFPyMuPDF4LLMProcessor(pdf_processor.PDFProcessor):
    """`PDFProcessor` whose fast mode extracts Markdown via pymupdf4llm."""

    OPT_IN = True

    def process_fast(self, file_path: str) -> MultimodalSample:
        """Extract the PDF as per-page Markdown instead of the raw text layer."""
        pdf_doc = pymupdf.Document(file_path)
        all_text_parts: List[str] = []
        embedded_images: List[Image.Image] = []
        paragraph_starts: List[
            Tuple[int, int, int]
        ] = []  # (char_offset, page_num, para_index)

        def _decode_image(link: str) -> Optional[Image.Image]:
            """Decode a base64 data image."""
            try:
                data = link.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

            except (IndexError, ValueError, UnidentifiedImageError) as e:
                logging.error(f"Could not decode embedded image: {e}")
                return None

        current_position = 0

        md_pages = cast(
            List[dict],
            pymupdf4llm.to_markdown(
                pdf_doc,
                page_chunks=True,
                # encode images as base64 encoded string in markdown
                embed_images=self.config.extract_images,
                dpi=self.config.custom_config.get("image_dpi", DEFAULT_IMAGE_DPI),
            ),
        )

        for page_num, page in enumerate(md_pages):
            text = cast(str, page["text"])

            # replace markdown encoded images by attachment tag
            for link in MD_IMG_REGEX.findall(text):
                tag = ""
                if self.config.extract_images:
                    image = _decode_image(link)
                    if image is not None and clean_image(image):
                        embedded_images.append(image)
                        tag = self.config.attachment_tag
                text = text.replace(f"![]({link})", tag, 1)

            # keep_two_line_breaks preserves the blank lines that separate
            # Markdown blocks, which the paragraph_starts split relies on.
            text = clean_text(text, keep_two_line_breaks=True)

            if not text.strip():
                continue

            para_idx = 0
            offset_in_page = 0
            for segment in text.split("\n\n"):
                if segment.strip():
                    paragraph_starts.append(
                        (current_position + offset_in_page, page_num, para_idx)
                    )
                    para_idx += 1
                offset_in_page += len(segment) + 2  # +2 for the "\n\n" separator

            all_text_parts.append(text)
            current_position += len(text)

        paragraph_starts.append((current_position, -1, -1))
        metadata = PDFMetadata(file_path=file_path, paragraph_starts=paragraph_starts)

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, metadata)
