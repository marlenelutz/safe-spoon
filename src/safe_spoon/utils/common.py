import logging
import pathlib
import zlib
from datetime import datetime
import sys
import json
import yaml


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_COLOR_PALETTE = [
    "\033[36m",  # cyan
    "\033[32m",  # green
    "\033[33m",  # yellow
    "\033[34m",  # blue
    "\033[35m",  # magenta
    "\033[91m",  # bright red
    "\033[92m",  # bright green
    "\033[93m",  # bright yellow
    "\033[94m",  # bright blue
    "\033[95m",  # bright magenta
    "\033[96m",  # bright cyan
]
_COLOR_RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    """Colors each console line by logger name (i.e. by module), so output
    from different parts of the package is visually distinguishable when
    several modules log interleaved (e.g. pipeline + embeddings + clustering
    in one run).

    The color is picked with a stable hash of the logger name rather than
    order-of-first-use, so a given module gets the same color on every run.
    Console-only: file output stays plain so log files don't fill up with
    escape codes.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _COLOR_PALETTE[zlib.crc32(record.name.encode()) % len(_COLOR_PALETTE)]
        return f"{color}{message}{_COLOR_RESET}"


def init_logger(
    config_path: str,
    name: str = None
) -> logging.Logger:
    """Initialize a logger based on the configuration file. This logger will log messages to both the console and a file, with the file being rotated based on the number of log files to keep.
    """

    cfg: dict = {}
    try:
        cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    except FileNotFoundError:
        pass

    logger_config = cfg.get("logger", {})
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
        fmt = '%(name)s - %(levelname)s - %(message)s'
        if sys.stdout.isatty():
            console_handler.setFormatter(ColorFormatter(fmt))
        else:
            console_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(console_handler)

    return logger


def load_yaml_config_file(config_path: str = "config/config.yaml") -> dict:
    """Load all parameters from the project config file."""
    cfg: dict = {}
    try:
        cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    except FileNotFoundError:
        pass

    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    lda = cfg.get("lda", {})
    clustering = cfg.get("clustering", {})
    embedding = cfg.get("embedding", {})
    pw = cfg.get("priority_weights", {})
    llm = cfg.get("llm", {})
    ui_display = cfg.get("ui_display") or {}

    def _str(v, default):
        return str(v) if v not in (None, "") else default

    llm_provider = _str(llm.get("provider"), "openai")
    llm_model = _str(llm.get("model"),  "gpt-5.4-mini-2026-03-17")
    llm_server = llm.get("server") or None
    llm_api_key = llm.get("api_key") or None

    # Rubric-suggestion LLM calls inherit the top-level llm.* settings
    # whenever llm.rubric_suggestion.* is null/omitted.
    rubric_cfg = llm.get("rubric_suggestion") or {}
    rubric_provider = _str(rubric_cfg.get("provider"), llm_provider)
    rubric_model = _str(rubric_cfg.get("model"), llm_model)
    rubric_server = rubric_cfg.get("server") or llm_server

    optimize_range = data.get("optimize_range") or lda.get(
        "optimize_range") or [30, 50, 5]

    return {
        "input_file": paths.get("input_file", "data/dataset/automatically-labeled-data/high_risk_automatically_labelled_filtered_cleaned.csv"),
        "reference_corpus": paths.get("reference_corpus", "data/reference_corpus.csv"),
        "reference_corpus_preprocessed": paths.get("reference_corpus_preprocessed", "data/reference_corpus_preprocessed.csv"),
        "output_json": paths.get("output_json", "data/output/viz_v5_data.json"),
        "labels_csv": paths.get("labels_csv", "data/output/labels.csv"),
        "annotation_db": paths.get("annotation_db", "data/output/annotation.db"),

        # Data
        "content_col": data.get("content_col", "content"),
        "label_col": data.get("label_col", "high_risk_label"),
        "categories": data.get("categories") or [],
        "active_categories": data.get("active_categories") or [],
        "oov_threshold": float(data.get("oov_threshold", 0.15)),
        "similarity_threshold": float(data.get("similarity_threshold", 0.95)),

        # LDA
        "n_topics": int(lda.get("n_topics", 30)),
        "lda_iters": int(lda.get("iters", 1500)),
        "lda_alpha": float(lda.get("alpha", 0.1)),
        "lda_eta": float(lda.get("eta", 0.01)),
        "min_df": int(lda.get("min_df", 10)),
        "max_df": float(lda.get("max_df", 0.6)),
        "spacy_model": lda.get("spacy_model", "en_core_web_lg"),
        "retrain": bool(lda.get("retrain", False)),
        "optimize": bool(lda.get("optimize", False)),
        "optimize_range": list(optimize_range),

        # Embedding model
        "embedding_model": embedding.get("model", "all-MiniLM-L6-v2"),
        "embedding_batch_size": int(embedding.get("batch_size", 64)),

        # Clustering / annotation-unit stopping rules
        "linkage_method": clustering.get("linkage_method", "average"),
        "n_repr_queries": int(clustering.get("n_repr_queries", 50)),
        "n_cut_levels": int(clustering.get("n_cut_levels", 40)),
        "min_size": int(clustering.get("min_size", 20)),
        "max_rel_dist": float(clustering.get("max_rel_dist", 0.40)),
        "purity_factor": float(clustering.get("purity_factor", 15.0)),
        "max_purity_cap": float(clustering.get("max_purity_cap", 0.80)),
        "min_cohesion": float(clustering.get("min_cohesion", 0.0)),
        "pw_mixture": float(pw.get("topic_mixture", 0.35)),
        "pw_size": float(pw.get("size", 0.25)),
        "pw_balance": float(pw.get("merge_balance", 0.15)),
        "pw_safety": float(pw.get("safety_signal", 0.25)),

        # LLM settings
        "llm_provider": llm_provider,
        "llm_model":    llm_model,
        "llm_server":   llm_server,
        "llm_api_key":  llm_api_key,
        "rubric_provider": rubric_provider,
        "rubric_model":    rubric_model,
        "rubric_server":   rubric_server,

        # UI display settings
        "top_docs_per_topic": int(cfg.get("top_docs_per_topic") or ui_display.get("topic_top_docs_generated", 10)),
        "query_labels": cfg.get("query_labels") or [
            {"field": "trash", "label": "Not a real request"},
            {"field": "demo",  "label": "Mentions personal details"},
        ],
        "ui_display": {
            "topic_words_max": int(ui_display.get("topic_words_max", 10)),
            "theme_bars_max": int(ui_display.get("theme_bars_max", 5)),
            "theme_bar_min_weight": float(ui_display.get("theme_bar_min_weight", 0.08)),
            "group_preview_queries": int(ui_display.get("group_preview_queries", 3)),
            "representative_queries_max": int(ui_display.get("representative_queries_max", 40)),
            "theme_typical_queries_max": int(ui_display.get("theme_typical_queries_max", 8)),
            "views": {
                "themes": bool(ui_display.get("views").get("themes", True)),
                "groups": bool(ui_display.get("views").get("groups", True)),
                "guidelines": bool(ui_display.get("views").get("guidelines", True)),
            },
        },
    }

def save_json(payload: dict, output_path) -> None:
    """Write payload as compact JSON to output_path.

    Parent directories are created automatically.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))