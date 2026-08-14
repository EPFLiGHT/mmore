import base64
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest
import yaml
from PIL import Image

from mmore.process.processors.base import (
    AutoProcessor,
    ProcessorConfig,
    ProcessorRegistry,
)
from mmore.process.processors.nemotron_vlm_processor import (
    NVIDIA_BASE_URL,
    NemotronVLMMetadata,
    NemotronVLMProcessor,
)
from mmore.process.processors.pdf_processor import PDFMetadata, PDFProcessor
from mmore.type import FileDescriptor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_CONFIGS = [
    "production-config/process/config.yaml",
    "examples/process/config.yaml",
    "examples/cc/process_config.yaml",
]


def make_file_descriptor(extension: str) -> FileDescriptor:
    return FileDescriptor(
        file_path=f"document{extension}",
        file_name=f"document{extension}",
        file_size=0,
        created_at="",
        modified_at="",
        file_extension=extension,
    )


def make_processor(
    tmp_path: Path,
    *,
    extract_images: bool = False,
    attachment_tag: str = "<attachment>",
    custom_config: dict | None = None,
) -> NemotronVLMProcessor:
    config = {"output_path": str(tmp_path), **(custom_config or {})}
    return NemotronVLMProcessor(
        ProcessorConfig(
            attachement_tag=attachment_tag,
            extract_images=extract_images,
            custom_config=config,
        )
    )


def make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def install_fake_openai(monkeypatch) -> MagicMock:
    constructor = MagicMock()
    module = ModuleType("openai")
    setattr(module, "OpenAI", constructor)
    monkeypatch.setitem(sys.modules, "openai", module)
    return constructor


@pytest.mark.parametrize(
    ("extension", "expected"),
    [(".pdf", True), (".PDF", True), (".txt", False)],
)
def test_accepts_only_pdf_case_insensitively(extension, expected):
    file = make_file_descriptor(extension)

    assert NemotronVLMProcessor.accepts(file) is expected


def test_processor_selection_can_choose_each_pdf_processor(monkeypatch):
    file = make_file_descriptor(".pdf")
    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [PDFProcessor, NemotronVLMProcessor],
    )

    assert (
        AutoProcessor.from_file(file, preferred_processor="PDFProcessor")
        is PDFProcessor
    )
    assert (
        AutoProcessor.from_file(file, preferred_processor="NemotronVLMProcessor")
        is NemotronVLMProcessor
    )


def test_custom_configuration_is_loaded(tmp_path):
    processor = make_processor(
        tmp_path,
        custom_config={
            "nemotron_model": "custom/model",
            "nemotron_dpi": "144",
            "nemotron_prompt": "Custom prompt",
            "nemotron_max_tokens": "123",
            "nemotron_temperature": "0.25",
        },
    )

    assert processor._model == "custom/model"
    assert processor._dpi == 144
    assert processor._prompt == "Custom prompt"
    assert processor._max_tokens == 123
    assert processor._temperature == 0.25


def test_get_client_requires_nvidia_api_key(monkeypatch, tmp_path):
    install_fake_openai(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    processor = make_processor(tmp_path)

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        processor._get_client()


def test_get_client_uses_nvidia_endpoint_and_caches_client(monkeypatch, tmp_path):
    constructor = install_fake_openai(monkeypatch)
    client = object()
    constructor.return_value = client
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    processor = make_processor(tmp_path)

    first_client = processor._get_client()
    second_client = processor._get_client()

    assert first_client is client
    assert second_client is client
    constructor.assert_called_once_with(
        api_key="test-key",
        base_url=NVIDIA_BASE_URL,
    )


def test_call_vlm_sends_expected_request_without_network(tmp_path):
    processor = make_processor(
        tmp_path,
        custom_config={
            "nemotron_model": "custom/model",
            "nemotron_prompt": "Extract this page",
            "nemotron_max_tokens": 321,
            "nemotron_temperature": 0.2,
        },
    )
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="# Extracted"))]
    )
    processor._client = client
    png = b"fake-png-bytes"

    result = processor._call_vlm(png)

    assert result == "# Extracted"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "custom/model"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 321
    assert request["messages"][0]["content"][0] == {
        "type": "text",
        "text": "Extract this page",
    }
    assert request["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        },
    }


def test_rasterize_returns_one_png_per_pdf_page(tmp_path):
    pdf_path = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    document.new_page(width=72, height=144)
    document.new_page(width=72, height=144)
    document.save(pdf_path)
    document.close()
    processor = make_processor(tmp_path, custom_config={"nemotron_dpi": 72})

    pages = processor._rasterize(str(pdf_path))

    assert len(pages) == 2
    for page in pages:
        with Image.open(io.BytesIO(page)) as image:
            assert image.format == "PNG"
            assert image.size == (72, 144)


def test_build_pagination_tracks_pages_paragraphs_and_sentinel():
    paragraph_starts, full_text = NemotronVLMProcessor._build_pagination(
        [(0, "First\n\nSecond"), (1, "Third")]
    )

    assert full_text == "First\n\nSecondThird"
    assert paragraph_starts == [
        (0, 0, 0),
        (7, 0, 1),
        (13, 1, 0),
        (18, -1, -1),
    ]


def test_process_without_images_keeps_markdown_and_has_no_modalities(
    monkeypatch, tmp_path
):
    processor = make_processor(tmp_path, extract_images=False)
    processor._client = object()
    monkeypatch.setattr(processor, "_rasterize", lambda _: [b"page"])
    monkeypatch.setattr(
        processor,
        "_call_vlm",
        lambda _: "Text ![figure](chart.png)",
    )

    sample = processor.process("document.pdf")

    assert sample.text == "Text ![figure](chart.png)"
    assert sample.modalities == []
    assert isinstance(sample.metadata, PDFMetadata)
    assert isinstance(sample.metadata, NemotronVLMMetadata)
    assert sample.metadata.file_path == "document.pdf"
    assert sample.metadata.backend == "nemotron-vlm"
    assert sample.metadata.model == processor._model


def test_process_with_images_replaces_markdown_and_creates_modality(
    monkeypatch, tmp_path
):
    processor = make_processor(
        tmp_path,
        extract_images=True,
        attachment_tag="<page-image>",
    )
    processor._client = object()
    monkeypatch.setattr(processor, "_rasterize", lambda _: [make_png_bytes()])
    monkeypatch.setattr(
        processor,
        "_call_vlm",
        lambda _: "Text ![figure](chart.png)",
    )

    sample = processor.process("document.pdf")

    assert sample.text == "Text <page-image>"
    assert len(sample.modalities) == 1
    assert sample.modalities[0].type == "image"
    assert Path(sample.modalities[0].value).is_file()
    assert sample.metadata.to_dict()["backend"] == "nemotron-vlm"
    assert sample.metadata.paragraph_starts[-1] == (len(sample.text), -1, -1)


def test_process_fails_before_rasterization_when_client_is_unavailable(
    monkeypatch, tmp_path
):
    processor = make_processor(tmp_path)
    get_client = MagicMock(side_effect=RuntimeError("missing API key"))
    rasterize = MagicMock()
    monkeypatch.setattr(processor, "_get_client", get_client)
    monkeypatch.setattr(processor, "_rasterize", rasterize)

    with pytest.raises(RuntimeError, match="missing API key"):
        processor.process("document.pdf")

    rasterize.assert_not_called()


def test_process_logs_failed_page_and_continues(monkeypatch, tmp_path, caplog):
    processor = make_processor(tmp_path)
    processor._client = object()
    monkeypatch.setattr(processor, "_rasterize", lambda _: [b"first", b"second"])
    call_vlm = MagicMock(side_effect=[RuntimeError("page failed"), "Second page"])
    monkeypatch.setattr(processor, "_call_vlm", call_vlm)

    sample = processor.process("document.pdf")

    assert sample.text == "Second page"
    assert "Nemotron VLM failed on page 0" in caplog.text
    assert sample.metadata.paragraph_starts == [
        (0, 1, 0),
        (len("Second page"), -1, -1),
    ]


@pytest.mark.parametrize("relative_path", PROCESS_CONFIGS)
def test_process_configs_default_to_pdf_and_define_nemotron(relative_path):
    config = yaml.safe_load((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    dispatcher_config = config["dispatcher_config"]

    assert dispatcher_config["processor_selection"][".pdf"] == "PDFProcessor"
    nemotron_config = {
        key: value
        for item in dispatcher_config["processor_config"]["NemotronVLMProcessor"]
        for key, value in item.items()
    }
    assert nemotron_config == {
        "nemotron_model": "nvidia/nemotron-nano-12b-v2-vl",
        "nemotron_dpi": 200,
        "nemotron_max_tokens": 4096,
        "nemotron_temperature": 0.0,
    }
