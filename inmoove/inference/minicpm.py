"""Lazy adapter for text and image conversations with MiniCPM-V."""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any


_JSON_REPLY_INSTRUCTION = """Return exactly one valid JSON object and nothing else.
Do not output Markdown, code fences, labels, explanations, or any text before or after the JSON object.
Use exactly these three fields and no placeholder values:
{"response": "안녕하세요. 반가워요!", "expression": "neutral", "intensity": 1}
The response value must be a natural-language reply to the user.
The expression value must be exactly one of these JSON strings: "neutral", "happy", "sad".
Never output placeholder text such as "emotion label" or "expression label".
The intensity value must be a JSON integer: 1, 2, or 3. Do not use a string or decimal value."""


def _import_pipeline() -> Callable[..., Callable[..., object]]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "MiniCPM inference dependencies are unavailable; install "
            "requirements-minicpm.txt"
        ) from exc
    return pipeline


def _assistant_content(result: object) -> str:
    """Return the last assistant message's content in a pipeline result."""
    if isinstance(result, list):
        assistant_messages = [
            item for item in result
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if assistant_messages:
            return _content_text(assistant_messages[-1].get("content"))
        return _assistant_content(result[-1]) if result else ""
    if isinstance(result, dict):
        generated = result.get("generated_text")
        if generated is not None:
            return _assistant_content(generated)
        if result.get("role") == "assistant":
            return _content_text(result.get("content"))
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _open_local_image(image_path: str | Path) -> object:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image file does not exist: {path}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for local image inference") from exc

    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except OSError as exc:
        raise ValueError(f"could not open image file: {path}") from exc


def _json_prompt(message: str) -> str:
    return f"{_JSON_REPLY_INSTRUCTION}\n\nUser request: {message}"


def _parse_structured_response(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("MiniCPM response must be a valid JSON object") from exc

    if not isinstance(parsed, dict):
        raise ValueError("MiniCPM response must be a JSON object")

    required_fields = {"response", "expression", "intensity"}
    missing_fields = required_fields - parsed.keys()
    if missing_fields:
        raise ValueError(
            "MiniCPM response is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    expression = parsed["expression"]
    if not isinstance(expression, str) or expression not in {"neutral", "happy", "sad"}:
        raise ValueError(
            "MiniCPM response expression must be one of: neutral, happy, sad"
        )

    intensity = parsed["intensity"]
    if isinstance(intensity, bool) or not isinstance(intensity, int) or intensity not in {
        1,
        2,
        3,
    }:
        raise ValueError("MiniCPM response intensity must be an integer from 1 to 3")

    return parsed


class MiniCPMTextChat:
    def __init__(
        self,
        model_id: str = "openbmb/MiniCPM-V-4.6",
        pipeline_factory: Callable[..., Callable[..., object]] | None = None,
    ) -> None:
        self._model_id = model_id
        self._pipeline_factory = pipeline_factory
        self._pipeline: Callable[..., object] | None = None

    def _load_pipeline(self) -> Callable[..., object]:
        if self._pipeline is None:
            factory = (
                self._pipeline_factory
                if self._pipeline_factory is not None
                else _import_pipeline()
            )
            try:
                self._pipeline = factory("image-text-to-text", model=self._model_id)
            except ImportError as exc:
                raise RuntimeError(
                    "MiniCPM inference dependencies are unavailable; install "
                    "requirements-minicpm.txt"
                ) from exc
        return self._pipeline

    def reply(self, message: str) -> str:
        if not message or not message.strip():
            raise ValueError("message must not be blank")

        messages = [{"role": "user", "content": [{"type": "text", "text": message}]}]
        result = self._load_pipeline()(text=messages, max_new_tokens=128, do_sample=False)
        content = _assistant_content(result)
        if not content:
            raise RuntimeError("MiniCPM pipeline returned no final assistant content")
        return content

    def reply_json(self, message: str) -> dict[str, Any]:
        """Answer a text prompt as a parsed JSON object."""
        if not message or not message.strip():
            raise ValueError("message must not be blank")

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": _json_prompt(message)}],
            }
        ]
        result = self._load_pipeline()(text=messages, max_new_tokens=128, do_sample=False)
        content = _assistant_content(result)
        if not content:
            raise RuntimeError("MiniCPM pipeline returned no final assistant content")
        return _parse_structured_response(content)

    def reply_with_image(self, image_path: str | Path, message: str) -> str:
        """Answer a question about one local image file."""
        if not message or not message.strip():
            raise ValueError("message must not be blank")

        image = _open_local_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": message},
                ],
            }
        ]
        result = self._load_pipeline()(text=messages, max_new_tokens=128, do_sample=False)
        content = _assistant_content(result)
        if not content:
            raise RuntimeError("MiniCPM pipeline returned no final assistant content")
        return content

    def reply_with_image_json(
        self, image_path: str | Path, message: str
    ) -> dict[str, Any]:
        """Answer a question about one local image as a parsed JSON object."""
        if not message or not message.strip():
            raise ValueError("message must not be blank")

        image = _open_local_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _json_prompt(message)},
                ],
            }
        ]
        result = self._load_pipeline()(text=messages, max_new_tokens=128, do_sample=False)
        content = _assistant_content(result)
        if not content:
            raise RuntimeError("MiniCPM pipeline returned no final assistant content")
        return _parse_structured_response(content)
