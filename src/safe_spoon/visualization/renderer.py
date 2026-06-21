import json
from pathlib import Path


def save_json(payload: dict, output_path) -> None:
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
    template_path = Path(template_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = template_path.read_text(encoding="utf-8")
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = html.replace(placeholder, payload_json)
    output_path.write_text(html, encoding="utf-8")
