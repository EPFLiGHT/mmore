import logging

from ..ux import setup_logging

WEBSEARCH_EMOJI = "🌐"
logger = setup_logging("WebSearch", WEBSEARCH_EMOJI)

# Keep a full DEBUG log on disk for debugging
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    file_handler = logging.FileHandler("shared_log_file.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
