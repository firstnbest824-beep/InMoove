"""Provisional normalized-value to PCA9685 pulse conversion for CH15/jaw."""

from dataclasses import dataclass
import math
from numbers import Real


CH15_PROVISIONAL_MIN_PULSE = 290
CH15_PROVISIONAL_MAX_PULSE = 320


@dataclass(frozen=True)
class NormalizedPulseCalibration:
    """Convert a normalized actuator value to an integer PCA9685 pulse count.

    This is deliberately distinct from ``ServoCalibration``: pulse counts are
    not degrees. The allowed range is the currently observed CH15 test window,
    not a final mechanical limit.
    """

    closed_pulse: int
    open_pulse: int

    def __post_init__(self) -> None:
        for name, pulse in (
            ("closed_pulse", self.closed_pulse),
            ("open_pulse", self.open_pulse),
        ):
            if isinstance(pulse, bool) or not isinstance(pulse, int):
                raise TypeError(f"{name} must be an integer pulse count")
            if not CH15_PROVISIONAL_MIN_PULSE <= pulse <= CH15_PROVISIONAL_MAX_PULSE:
                raise ValueError(
                    f"{name} must be within the provisional CH15 range "
                    f"{CH15_PROVISIONAL_MIN_PULSE}..{CH15_PROVISIONAL_MAX_PULSE}"
                )

    def to_pulse(self, value: float) -> int:
        """Convert a validated 0.0~1.0 jaw value into a bounded pulse count."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("normalized value must be a real number")

        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("normalized value must be within 0.0..1.0")

        pulse = round(
            self.closed_pulse + normalized * (self.open_pulse - self.closed_pulse)
        )
        if not CH15_PROVISIONAL_MIN_PULSE <= pulse <= CH15_PROVISIONAL_MAX_PULSE:
            raise ValueError("converted pulse is outside the provisional CH15 range")
        return pulse


# Current manual observations only: jaw=0.0 closes at 320; jaw=1.0 opens at
# 290. Replace this after measured mechanical calibration, rather than treating
# these pulse counts as final limits or servo angles.
PROVISIONAL_CH15_JAW_CALIBRATION = NormalizedPulseCalibration(
    closed_pulse=320,
    open_pulse=290,
)
