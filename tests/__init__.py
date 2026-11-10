import logging

# Keep test output readable: the box loggers would otherwise print retries/warnings to stderr.
_box_logger = logging.getLogger("box")
_box_logger.addHandler(logging.NullHandler())
_box_logger.propagate = False
