"""Ask MiniCPM one question about one local image."""

import argparse
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inmoove.inference import MiniCPMTextChat


def run_once(
    chat: MiniCPMTextChat,
    image_path: Path,
    message: str,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Print one MiniCPM answer grounded in ``image_path``."""
    output_fn(f"AI: {chat.reply_with_image(image_path, message)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask MiniCPM a question about one local image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to an image file")
    parser.add_argument("--message", required=True, help="Question or instruction for the image")
    args = parser.parse_args(argv)

    try:
        run_once(MiniCPMTextChat(), args.image, args.message)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
