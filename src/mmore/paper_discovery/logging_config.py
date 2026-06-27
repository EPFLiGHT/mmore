import logging

PAPER_EMOJI = "📄"
logger = logging.getLogger("PAPER_DISCOVERY")

logging.basicConfig(
    format=f"[PaperDiscovery {PAPER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
