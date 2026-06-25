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


def load_annotation_unit_config(config_path: str = "config/config.yaml") -> dict:
    """Load annotation-unit parameters from the project config file.

    Falls back to sensible defaults if the file does not exist or cannot
    be parsed, so the server can always start without a config file present.
    """
    cfg: dict = {}
    # Resolve relative paths against the project root (parent of this file's package)
    resolved = pathlib.Path(config_path)
    if not resolved.is_absolute():
        _pkg_root = pathlib.Path(__file__).parent.parent.parent.parent
        resolved = _pkg_root / config_path
    try:
        with open(resolved, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pass
    pw = cfg.get("priority_weights", {})
    ui = cfg.get("ui_display", {})
    ui_views = ui.get("views", {})
    _default_labels = [
        {"field": "trash", "label": "Not a real request"},
        {"field": "demo",  "label": "Mentions personal details"},
    ]
    raw_labels = cfg.get("query_labels", _default_labels)
    query_labels = [
        {"field": str(lf["field"]), "label": str(lf["label"])}
        for lf in raw_labels
        if isinstance(lf, dict) and "field" in lf and "label" in lf
    ] or _default_labels
    return {
        "min_size":      int(cfg.get("min_size",      50)),
        "purity_factor": float(cfg.get("purity_factor", 5.0)),
        "pw_mixture":    float(pw.get("topic_mixture",  0.5)),
        "pw_size":       float(pw.get("size",           0.3)),
        "pw_balance":    float(pw.get("merge_balance",  0.2)),
        "ui_display": {
            "topic_words_max": int(ui.get("topic_words_max", 10)),
            "theme_bars_max": int(ui.get("theme_bars_max", 5)),
            "theme_bar_min_weight": float(ui.get("theme_bar_min_weight", 0.1)),
            "group_preview_queries": int(ui.get("group_preview_queries", 3)),
            "representative_queries_max": int(ui.get("representative_queries_max", 40)),
            "topic_top_docs_generated": int(ui.get("topic_top_docs_generated", 20)),
            "theme_typical_queries_max": int(ui.get("theme_typical_queries_max", 20)),
            "views": {
                "themes": bool(ui_views.get("themes", True)),
                "groups": bool(ui_views.get("groups", True)),
                "guidelines": bool(ui_views.get("guidelines", False)),
            },
        },
        "query_labels": query_labels,
    }


def write_json_atomic(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)