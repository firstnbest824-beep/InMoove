"""CH15-only HeadInterface implementation for provisional jaw testing."""

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Protocol

from inmoove.face.expression import FaceExpression
from inmoove.face.interface import HeadInterface

from .jaw_calibration import (
    CH15_PROVISIONAL_MAX_PULSE,
    CH15_PROVISIONAL_MIN_PULSE,
    NormalizedPulseCalibration,
    PROVISIONAL_CH15_JAW_CALIBRATION,
)


CH15_NEUTRAL_PULSE = 305
JAW_DEMO_AMPLIFICATION = 3
JAW_DEMO_SETTLE_SECONDS = 0.2
JAW_DEMO_HOLD_SECONDS = 1.0


def amplify_jaw_demo_pulse(target_pulse: int) -> int:
    """Temporarily magnify CH15 motion away from neutral for visual demos.

    This is not physical calibration. It keeps the original normalized-to-pulse
    conversion intact, then expands only its visible displacement and clamps
    the result to the currently observed provisional CH15 range.
    """
    amplified = CH15_NEUTRAL_PULSE + JAW_DEMO_AMPLIFICATION * (
        target_pulse - CH15_NEUTRAL_PULSE
    )
    return max(
        CH15_PROVISIONAL_MIN_PULSE,
        min(CH15_PROVISIONAL_MAX_PULSE, amplified),
    )


class JawPulseTransport(Protocol):
    """The small serial boundary required by the jaw-only head."""

    def enable(self) -> None:
        """Enable Arduino CH15 manual mode."""

    def send_pulse(self, pulse: int) -> str:
        """Send one validated CH15 pulse and return the Arduino response."""

    def stop(self) -> None:
        """Release CH15 PWM output."""


@dataclass(frozen=True)
class JawCommandExchange:
    """One CH15-only command and the corresponding Arduino response."""

    command: str
    arduino_response: str


@dataclass(frozen=True)
class JawCommandResult:
    """The three-command provisional jaw demo produced for one AI response."""

    normalized_jaw: float
    original_target_pulse: int
    amplified_target_pulse: int
    exchanges: tuple[JawCommandExchange, ...]


class JawHardwareHead(HeadInterface):
    """Map only ``FaceExpression.jaw`` to provisional CH15 pulse commands."""

    def __init__(
        self,
        transport: JawPulseTransport,
        calibration: NormalizedPulseCalibration = PROVISIONAL_CH15_JAW_CALIBRATION,
        sleep_fn: Callable[[float], object] = time.sleep,
    ) -> None:
        self.transport = transport
        self.calibration = calibration
        self._sleep = sleep_fn
        self._enabled = False

    def enable(self) -> None:
        """Enable the manual CH15 path once, without sending a servo position."""
        self.transport.enable()
        self._enabled = True

    def set_expression(self, expression: FaceExpression) -> JawCommandResult:
        """Run a temporary CH15-only neutral-target-neutral visual demo."""
        if not isinstance(expression, FaceExpression):
            raise TypeError("expression must be a FaceExpression")
        if not self._enabled:
            raise RuntimeError("jaw hardware must be enabled before setting an expression")

        original_target_pulse = self.calibration.to_pulse(expression.jaw)
        amplified_target_pulse = amplify_jaw_demo_pulse(original_target_pulse)
        exchanges = [self._send_pulse(CH15_NEUTRAL_PULSE)]
        self._sleep(JAW_DEMO_SETTLE_SECONDS)
        exchanges.append(self._send_pulse(amplified_target_pulse))
        self._sleep(JAW_DEMO_HOLD_SECONDS)
        exchanges.append(self._send_pulse(CH15_NEUTRAL_PULSE))

        return JawCommandResult(
            normalized_jaw=expression.jaw,
            original_target_pulse=original_target_pulse,
            amplified_target_pulse=amplified_target_pulse,
            exchanges=tuple(exchanges),
        )

    def stop(self) -> None:
        """Release CH15 when the app exits after a successful enable."""
        if self._enabled:
            self.transport.stop()
            self._enabled = False

    def _send_pulse(self, pulse: int) -> JawCommandExchange:
        command = f"CH15_SET,{pulse}"
        return JawCommandExchange(
            command=command,
            arduino_response=self.transport.send_pulse(pulse),
        )
