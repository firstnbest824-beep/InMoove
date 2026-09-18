"""Run the keyboard-text MiniCPM to MockHead baseline.

This executable deliberately uses only ``MockHead``.  It never opens a serial
port or creates a ``RealHead``, so running it cannot move a servo.
"""

from pathlib import Path
import sys
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inmoove.face import FaceExpression, HeadInterface, MockHead, map_expression
from inmoove.inference import MiniCPMTextChat


class JsonChat(Protocol):
    """The minimal MiniCPM interface used by this application."""

    def reply_json(self, message: str) -> dict:
        """Return a response, expression label, and intensity."""


def _print_expression(
    reply: dict,
    face_expression: FaceExpression,
    output_fn: Callable[[str], None],
) -> None:
    output_fn(f"Robot response: {reply['response']}")
    output_fn(f"Expression: {reply['expression']}")
    output_fn(f"Intensity: {reply['intensity']}")
    output_fn("")
    output_fn("FaceExpression:")

    for name, value in zip(
        face_expression.get_field_names(), face_expression.get_all_values()
    ):
        output_fn(f"{name} = {value}")


def run_turn(
    chat: JsonChat,
    head: HeadInterface,
    message: str,
    output_fn: Callable[[str], None] = print,
) -> FaceExpression:
    """Process one non-blank keyboard message through the face pipeline."""
    reply = chat.reply_json(message)
    face_expression = map_expression(reply["expression"], reply["intensity"])
    head.set_expression(face_expression)
    _print_expression(reply, face_expression, output_fn)
    return face_expression


def run_cli(
    chat: JsonChat,
    head: HeadInterface,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Read text until ``/quit``, sending only valid turns to the head."""
    while True:
        try:
            message = input_fn("You: ").strip()
        except EOFError:
            return

        if message == "/quit":
            return
        if not message:
            continue

        try:
            run_turn(chat, head, message, output_fn=output_fn)
        except ValueError as error:
            output_fn(f"Face expression error: {error}")


def main() -> None:
    """Run the safe, software-only keyboard baseline."""
    run_cli(MiniCPMTextChat(), MockHead(verbose=False))


if __name__ == "__main__":
    main()
