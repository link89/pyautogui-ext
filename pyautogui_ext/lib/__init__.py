import logging
import os

logging.basicConfig(level=os.environ.get('LOG_LEVEL', logging.INFO))
