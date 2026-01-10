import logging


def setup_logger(logger_name: str, log_level: int) -> logging.Logger:
    """Set up a logger that logs to both console and a file."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logger.debug(f"Logger '{logger_name}' set up with level {logging.getLevelName(log_level)}")
        
    return logger