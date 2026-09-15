"""Temporary software-baseline mapping from LLM labels to FaceExpression.

This module is intentionally limited to normalized logical 16D values.  It is
not a servo calibration: replace these presets after the physical i2Head
calibration and RealHead/ServoCalibration layers are available.
"""

from .expression import FaceExpression


_NEUTRAL_16D = (0.5,) * 16

# These terminal vectors are copied exactly from examples/simple_demo.py.
# They remain temporary software-baseline values until hardware calibration.
_HAPPY_3_16D = (
    0.5,
    0.6,
    0.5,
    0.6,
    0.85,
    0.15,
    0.85,
    0.15,
    0.6,
    0.6,
    0.75,
    0.75,
    0.4,
    0.4,
    0.7,
    0.3,
)
_SAD_3_16D = (
    0.5,
    0.3,
    0.5,
    0.3,
    0.6,
    0.4,
    0.6,
    0.4,
    0.3,
    0.3,
    0.2,
    0.2,
    0.7,
    0.7,
    0.3,
    0.2,
)

_TERMINAL_16D = {
    "neutral": _NEUTRAL_16D,
    "happy": _HAPPY_3_16D,
    "sad": _SAD_3_16D,
}


def _validate_values(values: tuple[float, ...]) -> None:
    expected_length = len(FaceExpression.get_field_names())
    if len(values) != expected_length:
        raise ValueError(
            f"FaceExpression mapping must contain {expected_length} values, got {len(values)}"
        )
    if any(value < FaceExpression.MIN_VALUE or value > FaceExpression.MAX_VALUE for value in values):
        raise ValueError("FaceExpression mapping contains a value outside 0.0~1.0")


def map_expression(expression: str, intensity: int) -> FaceExpression:
    """Map a supported LLM expression label and intensity to FaceExpression.

    ``happy`` and ``sad`` intensities linearly interpolate from neutral at 0.5
    to their existing simple-demo vector at intensity 3. ``neutral`` remains
    the all-0.5 vector at every supported intensity.
    """
    if expression not in _TERMINAL_16D:
        raise ValueError(f"unsupported expression: {expression!r}")
    if isinstance(intensity, bool) or not isinstance(intensity, int) or intensity not in (1, 2, 3):
        raise ValueError("intensity must be an integer from 1 to 3")

    terminal = _TERMINAL_16D[expression]
    if expression == "neutral":
        values = _NEUTRAL_16D
    else:
        ratio = intensity / 3
        values = tuple(
            neutral + (target - neutral) * ratio
            for neutral, target in zip(_NEUTRAL_16D, terminal)
        )

    _validate_values(values)
    return FaceExpression(**dict(zip(FaceExpression.get_field_names(), values)))
