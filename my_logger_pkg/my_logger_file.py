import logging
from logging.handlers import TimedRotatingFileHandler
import os
LOG_NAME = f"logs/data_to_parquet.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

os.makedirs('logs', exist_ok=True)
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
