import logging

def setup_logging(level=logging.INFO):
    """
    Setup logging for the toolkit.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )
    logging.info("Logging is set up and ready to go.")
