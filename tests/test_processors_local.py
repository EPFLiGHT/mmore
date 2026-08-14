import os

from marker.output import MarkdownOutput

from mmore.process.processors.base import ProcessorConfig
from mmore.process.processors.docx_processor import DOCXProcessor
from mmore.process.processors.eml_processor import EMLProcessor
from mmore.process.processors.md_processor import MarkdownProcessor
from mmore.process.processors.media_processor import MediaProcessor
from mmore.process.processors.pdf_processor import PDFProcessor
from mmore.process.processors.pptx_processor import PPTXProcessor
from mmore.process.processors.spreadsheet_processor import SpreadsheetProcessor
from mmore.process.processors.txt_processor import TextProcessor
from mmore.process.processors.url_processor import URLProcessor
from mmore.type import FileDescriptor

"""
If you get an error when running tests with pytest, Run tests with: PYTHONPATH=$(pwd) pytest tests/test_processors_local.py.
This is required because the project follows a "src" layout, and setting PYTHONPATH ensures Python correctly resolves "src.mmore..." imports.
"""

SAMPLES_DIR = "examples/sample_data/"


def get_file_descriptor(file_path):
    return FileDescriptor.from_filename(file_path)


# ------------------ DOCX Processor Tests ------------------
def test_docx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = DOCXProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_docx_no_image_extraction():
    sample_file = os.path.join(SAMPLES_DIR, "docx", "ums.docx")
    # Assert that the sample file exists.
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    # Disable image extraction by setting "extract_images" to False.
    config = ProcessorConfig(
        extract_images=False,
        custom_config={
            "output_path": "tmp",
            "attachment_tag": "<attachment>",
        },
    )
    processor = DOCXProcessor(config=config)

    # Process the DOCX file.
    result = processor.process(sample_file)

    # Combine the extracted text into a single string for easy checking.
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )

    # Ensure that the attachment placeholder is not present.
    assert "<attachment>" not in combined_text, (
        "Attachment tag should not appear when image extraction is disabled."
    )
    # Verify that no images were extracted.
    assert len(result.modalities) == 0, (
        "Expected no images when image extraction is disabled."
    )


#  ------------------ EML Processor Tests ------------------
def test_eml_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = EMLProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_eml_headers_present():
    """
    Test that the processed EML file includes email headers: From, To, Subject, and Date
    """
    sample_file = os.path.join(SAMPLES_DIR, "eml", "sample.eml")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = EMLProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    # Ensure that the text field is not empty
    assert result.text, "Text should not be empty"
    # Combine text segments into one string
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    expected_headers = ["From:", "To:", "Subject:", "Date:"]
    # Assert each header is present
    for header in expected_headers:
        assert header in combined_text, f"{header} not found in the processed text"


# ------------------ Markdown Processor Tests ------------------
def test_md_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = MarkdownProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_md_image_extraction():
    """
    Test that the processor correctly extracts images and replaces image tags with the attachment placeholder.
    The sample md file contains one local image and one remote image. We expect: the processed text to contain two occurrences of the attachment placeholder and the modalities list to contain two entries
    """
    sample_file = os.path.join(SAMPLES_DIR, "md", "test.md")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    custom_attachment_tag = "<attachment>"
    # Set extract images to True explicitly
    config = ProcessorConfig(extract_images=True, custom_config={"output_path": "tmp"})
    processor = MarkdownProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    # Count the number of attachment placeholders inserted in text
    placeholder_count = combined_text.count(custom_attachment_tag)
    assert placeholder_count == 2, (
        f"Expected 2 attachment placeholders, found {placeholder_count}"
    )
    # Assert that modalities is a list and that two images were extracted
    assert isinstance(result.modalities, list), "Modalities should be a list"
    assert len(result.modalities) == 2, (
        f"Expected 2 images in modalities, found {len(result.modalities)}"
    )


# ------------------ Media Processor Tests ------------------
def test_media_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(
        custom_config={"normal_model": "model-name", "output_path": "tmp"}
    )
    processor = MediaProcessor(config=config)
    # Overriding load_models function with a lambda to bypass the actual model loading and heavy dependencies
    processor.load_models = lambda self=None, fast_mode=False: setattr(
        processor, "pipelines", [lambda x: {"text": "dummy transcription"}]
    )
    processor.load_models()
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_media_process_batch():
    # Create a list of sample files
    sample_file = os.path.join(SAMPLES_DIR, "media", "video.mp4")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    files = [sample_file, sample_file, sample_file]
    # Create a configuration and processor instance
    config = ProcessorConfig(
        custom_config={"normal_model": "model-name", "output_path": "tmp"}
    )
    processor = MediaProcessor(config=config)
    # Overriding load_models function with a lambda to bypass the actual model loading and heavy dependencies
    processor.load_models = lambda self=None, fast_mode=False: setattr(
        processor,
        "pipelines",
        [lambda x: {"text": "dummy transcription"} for _ in processor.devices],
    )
    processor.load_models()
    # Call process_batch with a dummy num_workers value
    results = processor.process_batch(files, fast_mode=False, num_workers=1)
    # Verify that each file in the batch produces a result with non-empty text and a list of modalities.
    assert len(results) == len(files), (
        "Number of results should match number of files processed."
    )
    for result in results:
        assert result.text, "Text should not be empty"
        assert isinstance(result.modalities, list), "Modalities should be a list"


# ------------------ PPTX Processor Tests ------------------
def test_pptx_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = PPTXProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_pptx_extract_notes():
    """
    Verify that PPTXProcessor correctly extracts slide notes.
    """
    sample_file = os.path.join(SAMPLES_DIR, "pptx", "ada.pptx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = PPTXProcessor(config=config)

    result = processor.process(sample_file)
    # Combine the text segments for easy searching
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )

    expected_text = "Data analysis has multiple facets and approaches"
    assert expected_text in combined_text, (
        f"Expected notes not found in extracted text: {combined_text}"
    )


# ------------------ Spreadsheet Processor Tests ------------------
def test_spreadsheet_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = SpreadsheetProcessor(config=config)
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_spreadsheet_multi_sheet_content():
    """
    Test that SpreadsheetProcessor correctly extracts text from multiple sheets
    in a spreadsheet that contains no images.
    """
    sample_file = os.path.join(SAMPLES_DIR, "spreadsheet", "survey.xlsx")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    config = ProcessorConfig(extract_images=True, custom_config={"output_path": "tmp"})
    processor = SpreadsheetProcessor(config=config)

    result = processor.process(sample_file)
    # Verify text was extracted
    assert result.text, "Expected some text in spreadsheet."

    # Convert extracted text to a single string for easy searching
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )

    # 1) Confirm that the names of each sheet appear in the extracted text
    expected_sheet_names = ["Form Responses 1"]
    for sheet_name in expected_sheet_names:
        assert sheet_name in combined_text, (
            f"Didn't find '{sheet_name}' in extracted text."
        )

    # 2) Check for specific cell content that should exist in the file
    expected_snippets = [
        "What is your current educational enrollment status?",
        "What is your gender?",
        "Master's Degree",
        "Female",
    ]
    for snippet in expected_snippets:
        assert snippet in combined_text, (
            f"Expected '{snippet}' not found in spreadsheet text."
        )

    # 3) Since there are no images, confirm modalities is empty
    assert isinstance(result.modalities, list), "Modalities should be a list."
    assert len(result.modalities) == 0, "Expected no images in this spreadsheet."


## ------------------ PDF Processor Tests ------------------
def test_pdf_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(
        extract_images=True,
        custom_config={
            "attachment_tag": "<attachment>",
            "output_path": "tmp",
        },
    )
    processor = PDFProcessor(config=config)
    processor.converter = (  # pyright: ignore[reportAttributeAccessIssue]
        lambda file_path: MarkdownOutput(
            markdown="dummy rendered content with ![](dummy.jpg)",
            images={"dummy.jpg": "dummy image data"},
            metadata={},
        )
    )
    # Process file
    result = processor.process(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_pdf_process_fast():
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    # Assert that the file exists beforre attempting to process it
    assert os.path.exists(sample_file), f"Sample file not found {sample_file}"
    config = ProcessorConfig(
        extract_images=True,
        custom_config={
            "attachment_tag": "<attachment>",
            "output_path": "tmp",
        },
    )
    processor = PDFProcessor(config=config)
    processor.converter = (  # pyright: ignore[reportAttributeAccessIssue]
        lambda file_path: "dummy rendered content with ![](dummy.jpg)"
    )
    # Process file
    result = processor.process_fast(sample_file)
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list), "Modalities should be a list"

    # ------------------ Text Processor Tests ------------------


def test_text_process_standard():
    sample_file = os.path.join(SAMPLES_DIR, "txt", "test.txt")
    # Assert that the text sample file exists.
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    config = ProcessorConfig(custom_config={"output_path": "tmp"})
    processor = TextProcessor(config=config)
    result = processor.process(sample_file)
    # Verify that some text is extracted and no image modalities are returned.
    assert result.text, "Text should not be empty"
    assert isinstance(result.modalities, list) and len(result.modalities) == 0, (
        "Modalities should be an empty list"
    )


# ------------------ URL Processor Tests ------------------
def test_url_process_standard():
    sample_url = "http://example.com"
    config = ProcessorConfig(
        extract_images=False,
        custom_config={
            "output_path": "tmp",
            "attachment_tag": "<attachment>",
        },
    )
    processor = URLProcessor(config=config)
    result = processor.process(sample_url)
    combined_text = (
        " ".join(result.text) if isinstance(result.text, list) else result.text
    )
    # Expect that the text from example.com contains "illustrative examples".
    assert "This domain" in combined_text, (
        "Expected 'This domain' in extracted text from http://example.com"
    )
    assert isinstance(result.modalities, list), "Modalities should be a list"


def test_url_process_invalid():
    sample_url = "http://thisurldoesnotexist.tld"
    config = ProcessorConfig(
        extract_images=False,
        custom_config={
            "output_path": "tmp",
            "attachment_tag": "<attachment>",
        },
    )
    processor = URLProcessor(config=config)
    result = processor.process(sample_url)
    # If URL processing fails, expect empty text and no modalities.
    assert not result.text, "Expected empty text for invalid URL"
    assert isinstance(result.modalities, list) and len(result.modalities) == 0, (
        "Expected no modalities for invalid URL"
    )
