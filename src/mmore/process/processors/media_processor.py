import logging
import os
import tempfile
import threading
from typing import List

import numpy as np
import torch
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from torch._C import device as torch_device
from transformers.pipelines import pipeline as pipeline_t

from ...type import DocumentMetadata, FileDescriptor, MultimodalSample
from ...ux import is_verbose, loading_model, progress
from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class MediaProcessor(Processor):
    @staticmethod
    def _get_available_devices():
        if torch.cuda.is_available():
            return [torch_device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available():
            return [torch_device("mps")]
        return [torch_device("cpu")]

    devices = _get_available_devices()
    pipelines = []
    # One pipeline per device, so several GPUs can transcribe in parallel
    pipelines_by_device: dict = {}
    _load_lock = threading.Lock()

    def __init__(self, config=None):
        super().__init__(config=config or ProcessorConfig())

    def _device_pipeline(self, device: str, fast_mode: bool = False):
        # Here fast_mode is not used in the cache indexing key as it won't
        # change across runs once defined in the configuration files
        cached = MediaProcessor.pipelines_by_device.get(device)
        if cached is not None:
            return cached

        with MediaProcessor._load_lock:
            if device not in MediaProcessor.pipelines_by_device:
                model_name = (
                    self.config.custom_config.get("fast_model", "openai/whisper-tiny")
                    if fast_mode
                    else self.config.custom_config.get(
                        "normal_model", "openai/whisper-large-v3-turbo"
                    )
                )
                MediaProcessor.pipelines_by_device[device] = pipeline_t(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=torch_device(device),
                    return_timestamps=True,
                )
            return MediaProcessor.pipelines_by_device[device]

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".mp3",
            ".flac",
            ".wav",
        ]

    @staticmethod
    def load_models(
        self=None,  # pyright: ignore[reportSelfClsParameterName]
        fast_mode=False,
    ):
        if self:
            model_name = (
                self.config.custom_config.get("fast_model", "openai/whisper-tiny")
                if fast_mode
                else self.config.custom_config.get(
                    "normal_model", "openai/whisper-large-v3-turbo"
                )
            )
        else:
            model_name = (
                "openai/whisper-tiny" if fast_mode else "openai/whisper-large-v3-turbo"
            )

        try:
            MediaProcessor.pipelines = []
            with loading_model(f"the speech-to-text model ({model_name})"):
                for device in MediaProcessor.devices:
                    pipe = pipeline_t(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=device,
                        return_timestamps=True,
                    )
                    MediaProcessor.pipelines.append(pipe)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            MediaProcessor.pipelines = []

    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        device = self.config.custom_config.get("device")
        if device is not None:
            pipe = self._device_pipeline(device, fast_mode)
            results = []
            for file in files_paths:
                try:
                    results.append(self._process_file(file, pipe, fast_mode))
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
            return results

        if not self.pipelines:
            self.load_models(fast_mode=fast_mode)

        file_chunks = self.evenly_split_across_gpus(files_paths, len(self.devices))

        results = []
        bar = progress(
            total=len(files_paths), desc=self.__class__.__name__, unit="file"
        )
        for pipeline, chunk in zip(self.pipelines, file_chunks):
            for file in chunk:
                bar.set_postfix_str(os.path.basename(file))
                try:
                    result = self._process_file(file, pipeline, fast_mode)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
                bar.update(1)
        bar.close()
        return results

    def _process_file(self, file_path, pipeline, fast_mode):
        all_text = self._extract_text(file_path, pipeline, fast_mode)
        if self.config.extract_images:
            images = self._extract_images(file_path)
        else:
            images = []

        return self.create_sample(
            [all_text], images, DocumentMetadata(file_path=file_path)
        )

    def process(self, file_path: str, fast: bool = False) -> MultimodalSample:
        if not self.pipelines:
            self.load_models(fast_mode=fast)
        if not self.pipelines:
            raise RuntimeError("Failed to load any processing pipelines.")

        pipeline = self.pipelines[0]
        all_text = self._extract_text(file_path, pipeline, fast_mode=fast)
        images = self._extract_images(file_path) if self.config.extract_images else []
        return self.create_sample(
            [all_text], images, DocumentMetadata(file_path=file_path)
        )

    def _extract_text(self, file_path: str, pipeline, fast_mode=False) -> str:
        def _prepare_audio_file(file_path: str, ext: str, temp_audio):
            mp_logger = "bar" if is_verbose() else None
            try:
                if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    with VideoFileClip(file_path) as clip:
                        if clip.audio is None:
                            raise ValueError("No audio track found in video.")
                        clip.audio.write_audiofile(
                            temp_audio.name, codec="pcm_s16le", logger=mp_logger
                        )
                elif ext in [".mp3", ".flac", ".wav"]:
                    with AudioFileClip(file_path) as audio_clip:
                        audio_clip.write_audiofile(
                            temp_audio.name, codec="pcm_s16le", logger=mp_logger
                        )
                temp_audio.flush()
            except Exception as e:
                logger.error(f"Error preparing audio file {file_path}: {e}")
                raise e

        ext = os.path.splitext(file_path)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                _prepare_audio_file(file_path, ext, temp_audio)
                result = pipeline(temp_audio.name)
                return result.get("text", "")
        except Exception as e:
            logger.error(f"Error transcribing {file_path}: {e}")
            return ""

    def _extract_images(self, file_path: str) -> List[Image.Image]:
        def _extract_video_frames(file_path: str) -> List[Image.Image]:
            images = []
            try:
                with VideoFileClip(file_path) as clip:
                    if clip.duration is None or clip.duration <= 0:
                        raise ValueError("Invalid video duration.")
                    duration = clip.duration
                    sample_rate = self.config.custom_config.get("sample_rate", 10)
                    num_thumbnails = max(1, int(duration / sample_rate))
                    for i in range(num_thumbnails):
                        t = min(i * sample_rate, duration - 0.1)
                        frame = np.asarray(clip.get_frame(t))
                        image = Image.fromarray(frame).convert("RGB")
                        images.append(image)
                logger.debug(f"Extracted {len(images)} images from {file_path}.")
            except Exception as e:
                logger.error(f"Error extracting images from {file_path}: {e}")
            return images

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp3", ".flac", ".wav"]:
            logger.debug(f"No images to extract from {file_path}.")
            return []
        return _extract_video_frames(file_path)

    @staticmethod
    def evenly_split_across_gpus(x_list, num_gpus):
        x_per_gpu = len(x_list) // num_gpus
        remainder = len(x_list) % num_gpus
        chunks = []
        start = 0
        for i in range(num_gpus):
            end = start + x_per_gpu + (1 if i < remainder else 0)
            chunks.append(x_list[start:end])
            start = end
        return chunks
