from mmore.process.processors.base import AutoProcessor, ProcessorRegistry
from mmore.type import FileDescriptor

# ---------------------------------------------------------------------------
# Definition of two dummy processors : ProcessorA, ProcessorB
# ---------------------------------------------------------------------------

class ProcessorA:
    @classmethod
    def accepts(cls, file):
        return file.file_extension == ".dummy"


class ProcessorB:
    @classmethod
    def accepts(cls, file):
        return file.file_extension == ".dummy"


# ---------------------------------------------------------------------------
# Dummy file instantiation, creating list of processors with ProcessorA coming
# first, and setting preference for processorB for the processor selection
# ---------------------------------------------------------------------------


def test_preferred_processor_is_selected(monkeypatch):
    dummy_file = FileDescriptor(
        file_path="test.dummy",
        file_name="test.dummy",
        file_size=0,
        created_at="",
        modified_at="",
        file_extension=".dummy",
    )

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