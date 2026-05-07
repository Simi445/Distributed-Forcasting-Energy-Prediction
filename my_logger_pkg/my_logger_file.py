import logging
from logging.handlers import TimedRotatingFileHandler
import os
LOGS_DIR = os.environ.get('LOGS_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs')))
SERVICE_NAME = os.environ.get('SERVICE_NAME', 'app')
LOG_NAME = os.path.join(LOGS_DIR, f"{SERVICE_NAME}.log")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

my_logger:logging.Logger = logging.getLogger(__name__)
my_logger.setLevel(logging.INFO)
fileHandler = TimedRotatingFileHandler(
    LOG_NAME, 
    when='midnight', 
    interval=1, 
    backupCount=30
)
fileHandler.setFormatter(logging.Formatter(LOG_FORMAT))
my_logger.addHandler(fileHandler)
