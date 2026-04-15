import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_cropguard_configured", False):
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
    logger._cropguard_configured = True
    return logger
