import json
import logging
import os
import pathlib
import pickle
from datetime import datetime
import sys
from typing import Any, Dict, Optional

import yaml


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def log_or_print(
    message: str,
    level: str = "info",
    logger: Optional[logging.Logger] = None
) -> None:
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)


def load_yaml_config_file(
    config_file: str,
    section: str,
    logger: logging.Logger
) -> Dict:
    if not pathlib.Path(config_file).exists():
        log_or_print(f"Config file not found: {config_file}", level="error", logger=logger)
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    section_dict = config.get(section, {})

    if section == {}:
        log_or_print(f"Section {section} not found in config file.", level="error", logger=logger)
        raise ValueError(f"Section {section} not found in config file.")

    log_or_print(f"Loaded config file {config_file} and section {section}.", logger=logger)

    return section_dict


def init_logger(
    config_file: str,
    name: str = None
) -> logging.Logger:
    logger_config = load_yaml_config_file(config_file, "logger", logger=None)
    name = name if name else logger_config.get("logger_name", "default_logger")
    log_level = logger_config.get("log_level", "INFO").upper()
    dir_logger = pathlib.Path(logger_config.get("dir_logger", "logs"))
    N_log_keep = int(logger_config.get("N_log_keep", 5))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    dir_logger.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be saved in {dir_logger}")

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{name}_log_{current_date}.log"
    log_file_path = dir_logger / log_file_name

    log_files = sorted(
        dir_logger.glob("*.log"),
        key=lambda f: f.stat().st_mtime, reverse=True)
    if len(log_files) >= N_log_keep:
        for old_file in log_files[N_log_keep - 1:]:
            old_file.unlink()

    if logger_config.get("file_log", True):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if logger_config.get("console_log", True):
        console_handler = FlushingStreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger


def unpickler(file: str) -> object:
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob: object) -> int:
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0


def get_unique_id(prefix: str = "") -> str:
    import uuid
    return f"{prefix}{uuid.uuid4().hex}"


def write_json_atomic(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
