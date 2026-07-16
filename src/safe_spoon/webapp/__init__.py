from pathlib import Path

from flask import Flask

from safe_spoon.utils.common import load_yaml_config_file
from safe_spoon.webapp import (
    annotation_units_routes,
    core_routes,
    labels_routes,
    rubric_routes,
)
from safe_spoon.logging_config import configure_logging
from safe_spoon.webapp.state import AppState

# static/ hangs off the repo root, not off this package's directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def create_app(config_path: str = "config/config.yaml") -> Flask:
    configure_logging(config_path)

    app = Flask(__name__, static_folder=str(_REPO_ROOT / "static"))

    cfg = load_yaml_config_file(config_path)
    state = AppState(cfg)
    app.extensions["safespoon"] = state

    app.register_blueprint(core_routes.bp)
    app.register_blueprint(labels_routes.bp)
    app.register_blueprint(annotation_units_routes.bp)
    app.register_blueprint(rubric_routes.bp)

    return app
