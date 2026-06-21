"""Visualization helpers: JSON serialization and HTML templating."""

import json
import os
from pathlib import Path


def save_json(payload: dict, output_path) -> None:
    """Write payload as compact JSON to output_path.

    Parent directories are created automatically.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def render_html(
    payload: dict,
    template_path,
    output_path,
    placeholder: str = "PAYLOAD_PLACEHOLDER",
) -> None:
    """Inject payload as JSON into an HTML template and write the result.

    The template file must contain placeholder exactly once; it is replaced
    with the JSON-encoded payload.

    Parameters
    ----------
    payload:
        Data to embed in the HTML.
    template_path:
        Path to the HTML template file.
    output_path:
        Destination path for the rendered HTML file.
    placeholder:
        String inside the template that will be replaced with the payload JSON.
    """
    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = template_path.read_text(encoding="utf-8")
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = html.replace(placeholder, payload_json)

    output_path.write_text(html, encoding="utf-8")
