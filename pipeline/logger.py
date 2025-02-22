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
logger.setLevel(logging.DEBUG if LOG_MODE == "debug" else logging.INFO)

formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Debug mode → logs everything to the console
if LOG_MODE == "debug":
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)

# Information mode → logs only info, warning, and error messages to a file
if LOG_MODE == "information":
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def log_info(message):
    logger.info(f"{message} | Timestamp: {get_current_time()}")


def log_warning(message):
    logger.warning(f"{message} | Timestamp: {get_current_time()}")


def log_error(message):
    logger.error(f"{message} | Timestamp: {get_current_time()}")


def log_debug(message):
    if LOG_MODE == "debug":
        logger.debug(f"{message} | Timestamp: {get_current_time()}")


def data_request(image_url, prompt, tags, box, n):
    if image_url == "":
        log_error("Empty image URL.")
        raise HTTPException(status_code=400, detail="Empty image URL in the request.")
    if prompt == "":
        log_error("Empty prompt.")
        raise HTTPException(status_code=400, detail="Empty prompt in the request.")
    if box is []:
        log_error("Empty box coordinates.")
        raise HTTPException(status_code=400, detail="Empty box coordinates in the request.")
    if len(box) != 4:
        log_error("Invalid number of box coordinates.")
        raise HTTPException(status_code=400, detail="Invalid number of box coordinates in the request.")
    if not all(isinstance(coord, (int, float)) for coord in box):
        log_error("Invalid box coordinates.")
        raise HTTPException(status_code=400, detail="Box coordinates must be integers or floats.")

    log_info(f"Request received successfully: {image_url}, {prompt}, {tags}, {box}, {n}")