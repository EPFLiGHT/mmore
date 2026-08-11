import logging

from mmore.process.crawler import DispatcherReadyResult
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.process.processors.base import AutoProcessor, ProcessorRegistry
from mmore.process.processors.url_processor import URLProcessor
from mmore.type import FileDescriptor

# ---------------------------------------------------------------------------
# Definition of dummy processors: ProcessorA, ProcessorB, IncompatibleProcessor
# ---------------------------------------------------------------------------


class ProcessorA:
    @classmethod
    def accepts(cls, file):
        return file.file_extension == ".dummy"


class ProcessorB:
    @classmethod
    def accepts(cls, file):
        return file.file_extension == ".dummy"


class IncompatibleProcessor:
    @classmethod
    def accepts(cls, file):
        return file.file_extension == ".other"


# ---------------------------------------------------------------------------
# Helper function used to instantiate dummy files without creating real files
# ---------------------------------------------------------------------------


def make_file_descriptor(file_name, file_extension):
    return FileDescriptor(
        file_path=file_name,
        file_name=file_name,
        file_size=0,
        created_at="",
        modified_at="",
        file_extension=file_extension,
    )


# ---------------------------------------------------------------------------
# Preferred processor selection when multiple processors accept the file
# ---------------------------------------------------------------------------


def test_preferred_processor_is_selected(monkeypatch):
    dummy_file = make_file_descriptor("test.dummy", ".dummy")

    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [ProcessorA, ProcessorB],
    )

    selected_processor = AutoProcessor.from_file(
        dummy_file,
        preferred_processor="ProcessorB",
    )

    assert selected_processor is ProcessorB


# ---------------------------------------------------------------------------
# Default selection when no preferred processor is configured
# ---------------------------------------------------------------------------


def test_first_compatible_processor_is_selected_without_preference(monkeypatch):
    dummy_file = make_file_descriptor("test.dummy", ".dummy")

    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [ProcessorA, ProcessorB],
    )

    selected_processor = AutoProcessor.from_file(dummy_file)

    assert selected_processor is ProcessorA


# ---------------------------------------------------------------------------
# No processor selection when none of the registered processors accept the file
# ---------------------------------------------------------------------------


def test_no_processor_is_selected_when_none_accepts(monkeypatch, caplog):
    unsupported_file = make_file_descriptor("test.unsupported", ".unsupported")

    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [ProcessorA, ProcessorB],
    )

    with caplog.at_level(logging.WARNING):
        selected_processor = AutoProcessor.from_file(unsupported_file)

    assert selected_processor is None
    assert "No registered processor found" in caplog.text


# ---------------------------------------------------------------------------
# No fallback when the preferred processor does not accept the file
# ---------------------------------------------------------------------------


def test_incompatible_preferred_processor_does_not_fallback(monkeypatch):
    dummy_file = make_file_descriptor("test.dummy", ".dummy")

    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [ProcessorA, IncompatibleProcessor],
    )

    selected_processor = AutoProcessor.from_file(
        dummy_file,
        preferred_processor="IncompatibleProcessor",
    )

    assert selected_processor is None


# ---------------------------------------------------------------------------
# Dispatcher selection with multiple files, including an unsupported file
# ---------------------------------------------------------------------------


def test_dispatcher_buckets_multiple_files_and_ignores_unsupported(
    monkeypatch, tmp_path
):
    first_dummy_file = make_file_descriptor("first.dummy", ".dummy")
    unsupported_file = make_file_descriptor("test.unsupported", ".unsupported")
    second_dummy_file = make_file_descriptor("second.dummy", ".dummy")

    monkeypatch.setattr(
        ProcessorRegistry,
        "_registry",
        [ProcessorA, ProcessorB, URLProcessor],
    )

    result = DispatcherReadyResult(
        urls=[],
        file_paths={"local": [first_dummy_file, unsupported_file, second_dummy_file]},
    )
    config = DispatcherConfig(
        output_path=str(tmp_path),
        processor_selection={".dummy": "ProcessorB"},
    )
    dispatcher = Dispatcher(result=result, config=config)

    dispatcher._bucket_files()

    assert dispatcher.intermediate_map[ProcessorA] == []
    assert dispatcher.intermediate_map[ProcessorB] == [
        first_dummy_file,
        second_dummy_file,
    ]
    assert all(
        unsupported_file not in files for files in dispatcher.intermediate_map.values()
    )
