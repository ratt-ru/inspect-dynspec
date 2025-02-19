import sys
import logging

CONSOLE = None

def create_logger():
    """Create a console logger"""
    log = logging.getLogger(__name__)
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    global CONSOLE
    CONSOLE = logging.StreamHandler(sys.stdout)
    CONSOLE.setLevel(logging.INFO)
    CONSOLE.setFormatter(cfmt)
    log.addHandler(CONSOLE)
    return log

def set_console_logging_level(level: int):
    CONSOLE.setLevel(level)

LOGGER = create_logger()