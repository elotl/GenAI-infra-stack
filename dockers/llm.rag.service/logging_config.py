import logging
import os
from logging.handlers import TimedRotatingFileHandler

from config import LoggingSettings

settings = LoggingSettings()
os.makedirs(
    os.path.dirname(settings.log_file_path), exist_ok=True
)  # Ensure log directory exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d -- %(message)s",
    handlers=[
        # Log to file, rotate every 1H and store files from last 24 hrs * 7 days files == 168H data
        TimedRotatingFileHandler(settings.log_file_path, when="h", interval=1, backupCount=168),
        logging.StreamHandler(),  # Also log to console
    ],
)

logger = logging.getLogger(__name__)  # Use __name__ here
logger.info("Logger initialized")

