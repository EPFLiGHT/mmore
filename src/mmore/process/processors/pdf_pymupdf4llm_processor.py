"""PDF fast-mode extraction using pymupdf4llm, opt-in through `processor_selection`.

Select it per extension in the dispatcher config:

    dispatcher_config:
      use_fast_processors: true
      processor_selection:
        ".pdf": PDFPyMuPDF4LLMProcessor
"""

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

# pymupdf4llm writes Markdown image links; they are stripped and replaced by attachment tags.
MD_IMG_REGEX = re.compile(r"!\[]\(([^)]*)\)")


class PDFPyMuPDF4LLMProcessor(pdf_processor.PDFProcessor):
    """`PDFProcessor` whose fast mode extracts Markdown via pymupdf4llm."""

    def process_fast(self, file_path: str) -> MultimodalSample:
        """Extract the PDF as per-page Markdown instead of the raw text layer."""
        pdf_doc = pymupdf.Document(file_path)
        all_text_parts: List[str] = []
        embedded_images: List[Image.Image] = []
        paragraph_starts: List[
            Tuple[int, int, int]
        ] = []  # (char_offset, page_num, para_index)

        def _extract_image(
            pdf_doc: pymupdf.Document, xref: int
        ) -> Optional[Image.Image]:
            """Extract an embedded image XObject at its native resolution as RGB."""
            try:
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image.get("image")

                if image_bytes is None:
                    logging.error(f"No image data found for xref {xref}")

                return Image.open(io.BytesIO(cast(bytes, image_bytes))).convert("RGB")

            except KeyError as e:
                logging.error(f"KeyError while extracting image: {e}")
                return None

            except UnidentifiedImageError as e:
                logging.error(
                    f"UnidentifiedImageError: Could not identify image file for xref {xref}: {e}"
                )
                return None

            except Exception as e:
                logging.error(
                    f"Unexpected error while extracting image for xref {xref}: {e}"
                )
                return None

        current_position = 0

        md_pages = pymupdf4llm.to_markdown(
            pdf_doc,
            page_chunks=True,
            write_images=False,
        )

        # track xref to detect images that are reused across the pdf
        seen_xrefs: set[int] = set()

        for page_num, page in enumerate(md_pages):
            text = cast(str, page["text"])
            text = MD_IMG_REGEX.sub("", text)

            # keep_two_line_breaks preserves the blank lines that separate
            # Markdown blocks, which the paragraph_starts split relies on.
            text = clean_text(text, keep_two_line_breaks=True)

            if self.config.extract_images:
                for img_info in pdf_doc[page_num].get_images(full=True):
                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    image = _extract_image(pdf_doc, xref)

                    if image is not None and clean_image(image):
                        embedded_images.append(image)
                        text = (
                            f"{text}\n\n{self.config.attachment_tag}"
                            if text.strip()
                            else self.config.attachment_tag
                        )

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
