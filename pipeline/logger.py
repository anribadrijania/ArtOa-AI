import logging
import os
from datetime import datetime
from fastapi import HTTPException
import pytz

LOG_MODE = os.getenv("LOG_MODE", "information")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Tbilisi")


def get_current_time():
    timezone = pytz.timezone(TIMEZONE)
    return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")


# Logger Configuration
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Always log everything internally

formatter = logging.Formatter("%(levelname)s: %(message)s | Timestamp: [%(asctime)s]")

# Console Handler (Only in Debug Mode)
if LOG_MODE == "debug":
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Log everything to console
    logger.addHandler(console_handler)

# File Handler (Log Everything to File)
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # Log everything to the file
logger.addHandler(file_handler)


# Logging Functions
def log_info(message):
    logger.info(message)


def log_warning(message):
    logger.warning(message)


def log_error(message):
    logger.error(message)


def log_debug(message):
    logger.debug(message)
