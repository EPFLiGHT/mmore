import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pymupdf
from PIL import Image

from ...type import FileDescriptor, MultimodalSample
from .base import Processor, ProcessorConfig
from .pdf_processor import PDFMetadata

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
DEFAULT_DPI = 200
DEFAULT_PROMPT = (
    "Extract the full content of this PDF page as clean GitHub-flavored Markdown.\n"
    "- Preserve headings, paragraphs, lists, and table structure.\n"
    "- Transcribe text verbatim; do not summarize.\n"
    "- For figures or charts, write a short bracketed caption like [Figure: ...].\n"
    "- Do not wrap the output in code fences. Output only the Markdown."
)

IMG_MD_REGEX = r"!\[[^\]]*\]\([^)]+\)"


@dataclass
class NemotronVLMMetadata(PDFMetadata):
    backend: str = "nemotron-vlm"
    model: str = DEFAULT_MODEL

    def to_dict(self) -> Dict[str, Any]:
        metadata = super().to_dict()
        metadata["backend"] = self.backend
        metadata["model"] = self.model
        return metadata


class NemotronVLMProcessor(Processor):
    """PDF processor that rasterizes pages and prompts NVIDIA's Nemotron Nano
    12B V2 VL via the OpenAI-compatible endpoint at integrate.api.nvidia.com.

    Selected through dispatcher_config.processor_selection by setting the
    preference in config.yaml.

    Requires `NVIDIA_API_KEY`.
    """

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        self._client = None
        cc = self.config.custom_config
        self._model = cc.get("nemotron_model", DEFAULT_MODEL)
        self._dpi = int(cc.get("nemotron_dpi", DEFAULT_DPI))
        self._prompt = cc.get("nemotron_prompt", DEFAULT_PROMPT)
        self._max_tokens = int(cc.get("nemotron_max_tokens", 4096))
        self._temperature = float(cc.get("nemotron_temperature", 0.0))

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK is required for NemotronVLMProcessor. "
                "Install with `pip install openai`."
            ) from e
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY env var is not set. Required for NemotronVLMProcessor."
            )
        self._client = OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)
        return self._client

    def _rasterize(self, file_path: str) -> List[bytes]:
        """Render each PDF page to PNG bytes at the configured DPI."""
        doc = pymupdf.Document(file_path)
        zoom = self._dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pages_png: List[bytes] = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)  # type: ignore[attr-defined]
            pages_png.append(pix.tobytes("png"))
        doc.close()
        return pages_png

    def _call_vlm(self, png_bytes: bytes) -> str:
        client = self._get_client()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        completion = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return completion.choices[0].message.content or ""

    def process(self, file_path: str) -> MultimodalSample:
        self._get_client()
        page_pngs = self._rasterize(file_path)
        page_texts: List[Tuple[int, str]] = []
        images: List[Image.Image] = []
        extract_images = self.config.extract_images

        for page_idx, png in enumerate(page_pngs):
            try:
                md = self._call_vlm(png)
            except Exception as e:
                logger.error(
                    f"Nemotron VLM failed on page {page_idx} of {file_path}: {e}"
                )
                md = ""

            if extract_images:
                images.append(Image.open(io.BytesIO(png)).convert("RGB"))
                md = re.sub(IMG_MD_REGEX, self.config.attachment_tag, md)

            page_texts.append((page_idx, md))

        paragraph_starts, full_text = self._build_pagination(page_texts)

        metadata = NemotronVLMMetadata(
            file_path=file_path,
            paragraph_starts=paragraph_starts,
            model=self._model,
        )
        return self.create_sample([full_text], images, metadata)

    @staticmethod
    def _build_pagination(
        page_texts: List[Tuple[int, str]],
    ) -> Tuple[List[Tuple[int, int, int]], str]:
        paragraph_starts: List[Tuple[int, int, int]] = []
        current_position = 0
        parts: List[str] = []
        for page_id, page_content in page_texts:
            para_idx = 0
            offset_in_page = 0
            for segment in page_content.split("\n\n"):
                if segment.strip():
                    paragraph_starts.append(
                        (current_position + offset_in_page, page_id, para_idx)
                    )
                    para_idx += 1
                offset_in_page += len(segment) + 2
            parts.append(page_content)
            current_position += len(page_content)
        paragraph_starts.append((current_position, -1, -1))
        return paragraph_starts, "".join(parts)
