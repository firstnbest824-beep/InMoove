"""Lazy adapter for text-only conversations with MiniCPM-V."""

from collections.abc import Callable
from typing import Any


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
        for item in reversed(result):
            content = _assistant_content(item)
            if content:
                return content
        return ""
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
            factory = self._pipeline_factory or _import_pipeline()
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
