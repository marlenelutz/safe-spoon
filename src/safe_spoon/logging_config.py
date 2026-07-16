"""Unified logging configuration for the safe_spoon package."""

import logging

from safe_spoon.utils.common import init_logger

PACKAGE_LOGGER_NAME = "safe_spoon"


def configure_logging(config_path: str) -> logging.Logger:
    return init_logger(config_path, name=PACKAGE_LOGGER_NAME)
