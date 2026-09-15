"""Run a keyboard chat session with MiniCPM."""

from typing import Callable

from inmoove.inference import MiniCPMTextChat


def run_cli(
    chat: MiniCPMTextChat,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Read user messages and print MiniCPM replies until the user exits."""
    while True:
        try:
            message = input_fn("You: ").strip()
        except EOFError:
            return

        if message == "/quit":
            return
        if message:
            output_fn(f"AI: {chat.reply(message)}")


if __name__ == "__main__":
    run_cli(MiniCPMTextChat())
