import io
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from multiprocessing import Manager, Process, set_start_method
from typing import Any, Dict, List, Optional, Tuple, cast

import pymupdf
import torch
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL import Image, UnidentifiedImageError

from ...type import DocumentMetadata, FileDescriptor, MultimodalSample
from ...ux import is_verbose, loading_model, progress
from ..utils import clean_image, clean_text
from .base import Processor, ProcessorConfig

IMG_REGEX = r"!\[\]\(_page_\d+_[A-Za-z0-9_]+\.(jpeg|jpg|png|gif)\)"


@dataclass
class PDFMetadata(DocumentMetadata):
    paragraph_starts: List[Tuple[int, int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        metadata = super().to_dict()
        if self.paragraph_starts:
            metadata["paragraph_starts"] = self.paragraph_starts

        return metadata


class PDFProcessor(Processor):
    artifact_dict = None
    # One model set per device, so several GPUs can process in parallel
    artifacts_by_device: Dict[str, Any] = {}
    _load_lock = threading.Lock()

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())
        self.converter = None

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() == ".pdf"

    @staticmethod
    def _get_artifacts(device: Optional[str] = None):
        if device is None:
            cached = PDFProcessor.artifact_dict
        else:
            cached = PDFProcessor.artifacts_by_device.get(device)
        if cached is not None:
            return cached

        with PDFProcessor._load_lock:
            if device is None:
                if PDFProcessor.artifact_dict is None:
                    with loading_model("the PDF reading model"):
                        PDFProcessor.artifact_dict = create_model_dict()
                return PDFProcessor.artifact_dict
            if device not in PDFProcessor.artifacts_by_device:
                PDFProcessor.artifacts_by_device[device] = create_model_dict(
                    device=device
                )
            return PDFProcessor.artifacts_by_device[device]

    @staticmethod
    def load_models(
        disable_image_extraction: bool = False, device: Optional[str] = None
    ):
        artifact_dict = PDFProcessor._get_artifacts(device)

        marker_config = {
            "disable_image_extraction": disable_image_extraction,
            "languages": None,
            "use_llm": False,
            "disable_multiprocessing": False,
            "paginate_output": True,
            "disable_tqdm": not is_verbose(),
        }
        if device is not None:
            marker_config["device"] = str(device)
        config_parser = ConfigParser(marker_config)
        converter = PdfConverter(
            artifact_dict=artifact_dict,
            config=config_parser.generate_config_dict(),
        )

        converter.initialize_processors(list(converter.default_processors))

        return converter

    # overwriting the process_batch
    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        device = self.config.custom_config.get("device")
        if device is not None:
            self.converter = PDFProcessor.load_models(
                disable_image_extraction=not self.config.extract_images,
                device=device,
            )
            results = []
            for file_path in files_paths:
                try:
                    results.append(self.process(file_path))
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {str(e)}")
            return results

        if fast_mode:  # No GPU available - fallback to default
            return super().process_batch(files_paths, fast_mode, num_workers)
        else:
            if not torch.cuda.is_available():
                num_gpus = 1
            else:
                num_gpus = torch.cuda.device_count()

            # 1 GPU available or length of files_paths is less than 10 we just do single-GPU
            if num_gpus == 1 or len(files_paths) < 10:
                if self.converter is None:
                    self.converter = PDFProcessor.load_models(
                        disable_image_extraction=not self.config.custom_config.get(
                            "extract_images", True
                        )
                    )

                results = []
                bar = progress(files_paths, desc=self.__class__.__name__, unit="file")
                for file_path in bar:
                    bar.set_postfix_str(os.path.basename(file_path))
                    try:
                        res = self.process(file_path)
                        results.append(res)
                    except Exception as e:
                        logging.error(f"Failed to process {file_path}: {str(e)}")

                return results
            else:  # Multiple GPUs available
                batches = self._split_files(files_paths, num_gpus)

                try:
                    set_start_method("spawn", force=True)
                except RuntimeError:
                    pass

                manager = Manager()
                output_queue = manager.Queue()
                error_queue = manager.Queue()
                processes = []

                for i, batch in enumerate(batches):
                    if not batch:
                        continue
                    gpu_id = i % num_gpus
                    p = Process(
                        target=self._process_parallel,
                        args=(
                            batch,
                            gpu_id,
                            self.config.extract_images,
                            output_queue,
                            error_queue,
                        ),
                    )
                    processes.append(p)
                    p.start()

                results = []

                while any(p.is_alive() for p in processes):
                    if not error_queue.empty():
                        error = error_queue.get()
                        raise RuntimeError(f"Child process failed: {error}")
                    while not output_queue.empty():
                        results.extend(output_queue.get())

                while not output_queue.empty():
                    results.extend(output_queue.get())

                if not error_queue.empty():
                    error = error_queue.get()
                    raise RuntimeError(f"Child process failed: {error}")

                return results

    # Regex matching marker page separators: \n\n{page_id}----...\n\n
    _PAGE_SEP_RE = re.compile(r"\n\n\{(\d+)\}-{3,}\n\n")

    def process(self, file_path: str) -> MultimodalSample:
        if self.converter is None:
            self.converter = PDFProcessor.load_models(
                disable_image_extraction=not self.config.custom_config.get(
                    "extract_images", True
                )
            )

        rendered = self.converter(file_path)
        text, _, images = text_from_rendered(rendered)
        text = re.sub(str(IMG_REGEX), "<attachment>", cast(str, text))
        images = list(images.values())

        paragraph_starts, text = self._parse_pagination(cast(str, text))

        metadata = PDFMetadata(file_path=file_path)
        if paragraph_starts:
            metadata.paragraph_starts = paragraph_starts

        return self.create_sample([text], images, metadata)

    @classmethod
    def _parse_pagination(
        cls, text: str
    ) -> Tuple[
        List[Tuple[int, int, int]],
        str,
    ]:
        """Parse marker pagination separators to build paragraph_starts,
        then strip the separators from the text."""
        separators = list(cls._PAGE_SEP_RE.finditer(text))
        if not separators:
            return [], text

        page_texts: List[Tuple[int, str]] = []  # (page_id, page_content)
        prev_end = 0
        for match in separators:
            page_id = int(match.group(1))
            page_content = text[prev_end : match.start()]
            page_texts.append((page_id, page_content))
            prev_end = match.end()
        trailing = text[prev_end:]
        if trailing.strip():
            last_page_id = int(separators[-1].group(1)) + 1
            page_texts.append((last_page_id, trailing))

        paragraph_starts: List[Tuple[int, int, int]] = []
        current_position = 0

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

            current_position += len(page_content)

        paragraph_starts.append((current_position, -1, -1))

        clean_text = "".join(content for _, content in page_texts)

        return paragraph_starts, clean_text

    def process_fast(self, file_path: str) -> MultimodalSample:
        pdf_doc = pymupdf.Document(file_path)
        all_text_parts = []
        embedded_images = []
        paragraph_starts: List[
            Tuple[int, int, int]
        ] = []  # (char_offset, page_num, para_index)
        current_position = 0

        def _extract_images(pdf_doc, xref) -> Optional[Image.Image]:
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

        for page_num, page in enumerate(pdf_doc):  # pyright: ignore[reportArgumentType]
            text = clean_text(page.get_text())  # type: ignore[attr-defined]

            if text.strip():
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

            if self.config.extract_images:
                for img_info in page.get_images(full=False):
                    image = _extract_images(pdf_doc, img_info[0])
                    if image and clean_image(image):
                        # clean image filters images below size 512x512 and variance below 100, these are defaults and can be changed
                        embedded_images.append(image)
                        attachment_text = self.config.attachment_tag
                        all_text_parts.append(attachment_text)
                        current_position += len(attachment_text)
            else:
                embedded_images = []

        paragraph_starts.append((current_position, -1, -1))
        metadata = PDFMetadata(file_path=file_path, paragraph_starts=paragraph_starts)

        full_text = "".join(all_text_parts)
        return self.create_sample([full_text], embedded_images, metadata)

    # Functions for parallelizing across GPUs
    def _split_files(self, files_paths, num_batches):
        file_sizes = [(file, self.get_file_size(file)) for file in files_paths]
        sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)

        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        for file, size in sorted_files:
            min_index = batch_sizes.index(min(batch_sizes))
            batches[min_index].append(file)
            batch_sizes[min_index] += size

        batches = [batch for batch in batches if batch]
        return batches

    def _process_parallel(
        self, files_paths, gpu_id, extract_images, output_queue, error_queue
    ):
        try:
            torch.cuda.set_device(gpu_id)

            if PDFProcessor.artifact_dict is None:
                PDFProcessor.artifact_dict = create_model_dict(device=f"cuda:{gpu_id}")

            marker_config = {
                "disable_image_extraction": not extract_images,
                "languages": None,
                "use_llm": False,
                "disable_multiprocessing": False,
                "device": f"cuda:{gpu_id}",
                "disable_tqdm": not is_verbose(),
            }

            config_parser = ConfigParser(marker_config)
            self.converter = PdfConverter(
                artifact_dict=PDFProcessor.artifact_dict,
                config=config_parser.generate_config_dict(),
            )

            batch_results = []
            for file in files_paths:
                try:
                    result = self.process(file)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process {file}: {str(e)}")
                    batch_results.append(None)  # handle partial failures

            output_queue.put(batch_results)

        except Exception as e:
            error_queue.put(f"GPU {gpu_id} failed: {str(e)}")
            raise e
        finally:
            torch.cuda.empty_cache()
            if hasattr(self, "converter"):
                del self.converter
