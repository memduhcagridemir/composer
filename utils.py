import logging


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
