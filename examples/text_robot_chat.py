"""Run keyboard-text MiniCPM with a safe MockHead default.

Use ``--jaw-hardware`` only after manually uploading the CH15-only Arduino
test sketch. The default path never opens a serial port or moves a servo.
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inmoove.face import FaceExpression, HeadInterface, MockHead, map_expression
from inmoove.hardware import Ch15SerialTransport, JawCommandResult, JawHardwareHead
from inmoove.inference import MiniCPMTextChat


class JsonChat(Protocol):
    """The minimal MiniCPM interface used by this application."""

    def reply_json(self, message: str) -> dict:
        """Return a response, expression label, and intensity."""


JawHeadFactory = Callable[[], JawHardwareHead]


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


def _print_jaw_hardware_result(
    result: JawCommandResult, output_fn: Callable[[str], None]
) -> None:
    """Print the one CH15-only command produced for a valid AI response."""
    output_fn("Jaw hardware:")
    output_fn(f"normalized jaw = {result.normalized_jaw}")
    output_fn(f"original target pulse = {result.original_target_pulse}")
    output_fn(f"amplified target pulse = {result.amplified_target_pulse}")
    for exchange in result.exchanges:
        output_fn(f"PC -> Arduino: {exchange.command}")
        output_fn(f"Arduino -> PC: {exchange.arduino_response}")


def run_turn(
    chat: JsonChat,
    head: HeadInterface,
    message: str,
    output_fn: Callable[[str], None] = print,
    jaw_head: JawHardwareHead | None = None,
) -> FaceExpression:
    """Process one non-blank keyboard message through the face pipeline."""
    reply = chat.reply_json(message)
    face_expression = map_expression(reply["expression"], reply["intensity"])
    jaw_result = None
    if jaw_head is not None:
        # ``JawHardwareHead`` sends only CH15_SET,<pulse>; it cannot emit an
        # ANGLES packet or reference any of CH0~CH14.
        jaw_result = jaw_head.set_expression(face_expression)
    head.set_expression(face_expression)
    _print_expression(reply, face_expression, output_fn)
    if isinstance(jaw_result, JawCommandResult):
        _print_jaw_hardware_result(jaw_result, output_fn)
    return face_expression


def run_cli(
    chat: JsonChat,
    head: HeadInterface,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    jaw_head: JawHardwareHead | None = None,
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
            run_turn(chat, head, message, output_fn=output_fn, jaw_head=jaw_head)
        except ValueError as error:
            output_fn(f"Face expression error: {error}")
            if jaw_head is not None:
                # No later AI turn may command CH15 after malformed data; the
                # enclosing hardware application's finally block sends STOP.
                return


def _create_jaw_hardware_head() -> JawHardwareHead:
    """Construct the opt-in CH15-only head without opening serial yet."""
    return JawHardwareHead(transport=Ch15SerialTransport())


def run_application(
    chat: JsonChat,
    *,
    jaw_hardware: bool = False,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    jaw_head_factory: JawHeadFactory = _create_jaw_hardware_head,
) -> None:
    """Run the MockHead baseline, optionally mirroring jaw only to CH15."""
    mock_head = MockHead(verbose=False)
    if not jaw_hardware:
        run_cli(chat, mock_head, input_fn=input_fn, output_fn=output_fn)
        return

    jaw_head = jaw_head_factory()
    enabled = False
    try:
        # Opens serial and sends exactly one CH15_TEST_ENABLE. No position is
        # sent until a valid MiniCPM reply has mapped to FaceExpression.
        jaw_head.enable()
        enabled = True
        run_cli(
            chat,
            mock_head,
            input_fn=input_fn,
            output_fn=output_fn,
            jaw_head=jaw_head,
        )
    finally:
        if enabled:
            jaw_head.stop()


def main(argv: list[str] | None = None) -> None:
    """Run MockHead by default; use an explicit flag for jaw-only hardware."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jaw-hardware",
        action="store_true",
        help="mirror only FaceExpression.jaw to provisional PCA9685 CH15",
    )
    args = parser.parse_args(argv)
    run_application(MiniCPMTextChat(), jaw_hardware=args.jaw_hardware)


if __name__ == "__main__":
    main()
