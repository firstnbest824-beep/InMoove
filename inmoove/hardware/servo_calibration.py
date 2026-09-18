"""Named, per-servo conversion from normalized values to physical angles."""

from dataclasses import dataclass
import math
from numbers import Real


def _require_finite_number(name: str, value: Real) -> float:
    """Return a finite numeric value and reject booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")

    return result


@dataclass(frozen=True)
class ServoCalibration:
    """The measured safe angle limits and direction for one named servo.

    No default angle limits are supplied: callers must provide measured values
    for each physical servo before creating a ``RealHead``.  A normalized value
    outside 0.0~1.0 is rejected rather than silently clamped, matching
    ``FaceExpression`` validation.
    """

    min_angle: float
    max_angle: float
    reversed: bool = False

    def __post_init__(self) -> None:
        min_angle = _require_finite_number("min_angle", self.min_angle)
        max_angle = _require_finite_number("max_angle", self.max_angle)

        if min_angle > max_angle:
            raise ValueError("min_angle must not exceed max_angle")

        object.__setattr__(self, "min_angle", min_angle)
        object.__setattr__(self, "max_angle", max_angle)

    def to_angle(self, value: float) -> float:
        """Convert one validated 0.0~1.0 value into this servo's angle range."""
        normalized = _require_finite_number("normalized value", value)
        if not 0.0 <= normalized <= 1.0:
            raise ValueError("normalized value must be within 0.0~1.0")

        if self.reversed:
            normalized = 1.0 - normalized

        return self.min_angle + normalized * (self.max_angle - self.min_angle)
