"""Named 16D face-expression conversion to fixed PCA9685 channel order."""

from collections.abc import Mapping

from inmoove.face.expression import FaceExpression
from inmoove.face.interface import HeadInterface

from .serial_transport import AngleTransport
from .servo_calibration import ServoCalibration


# The insertion order is the wire order: the value is the PCA9685 channel.
# Keep this explicit mapping aligned with FaceExpression.get_field_names().
SERVO_CHANNELS = {
    "eye_left_lr": 0,
    "eye_left_ud": 1,
    "eye_right_lr": 2,
    "eye_right_ud": 3,
    "eyelid_left_upper": 4,
    "eyelid_left_lower": 5,
    "eyelid_right_upper": 6,
    "eyelid_right_lower": 7,
    "eyebrow_left": 8,
    "eyebrow_right": 9,
    "cheek_left": 10,
    "cheek_right": 11,
    "forehead_left": 12,
    "forehead_right": 13,
    "upper_lip": 14,
    "jaw": 15,
}


class RealHead(HeadInterface):
    """Apply named calibrations, then optionally send channel-ordered angles.

    ``transport`` is intentionally optional and is never constructed here.
    This makes conversion safe to use in software-only tests and prevents an
    accidental serial-port open merely by creating a ``RealHead``.
    """

    def __init__(
        self,
        calibrations: Mapping[str, ServoCalibration],
        transport: AngleTransport | None = None,
    ) -> None:
        expected_names = set(SERVO_CHANNELS)
        received_names = set(calibrations)
        if received_names != expected_names:
            raise ValueError(
                "calibration names must exactly match the 16 actuator names; "
                f"missing={sorted(expected_names - received_names)}, "
                f"unexpected={sorted(received_names - expected_names)}"
            )
        if not all(isinstance(value, ServoCalibration) for value in calibrations.values()):
            raise TypeError("every named calibration must be a ServoCalibration")

        self.calibrations = dict(calibrations)
        self.transport = transport

    def convert(self, expression: FaceExpression) -> tuple[float, ...]:
        """Return 16 calibrated angles in the explicit CH0~CH15 order."""
        if not isinstance(expression, FaceExpression):
            raise TypeError("expression must be a FaceExpression")

        return tuple(
            self.calibrations[name].to_angle(getattr(expression, name))
            for name in SERVO_CHANNELS
        )

    def set_expression(self, expression: FaceExpression) -> None:
        """Convert and explicitly forward a face expression through the injected transport."""
        if self.transport is None:
            raise RuntimeError("RealHead requires an injected transport to send angles")

        self.transport.send_angles(self.convert(expression))
