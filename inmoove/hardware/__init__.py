"""Step 5 hardware-boundary utilities for calibrated face commands.

This package does not open serial ports or drive servos on import.
"""

from .real_head import RealHead, SERVO_CHANNELS
from .jaw_calibration import NormalizedPulseCalibration, PROVISIONAL_CH15_JAW_CALIBRATION
from .jaw_head import JawCommandResult, JawHardwareHead
from .jaw_serial_transport import Ch15SerialTransport
from .serial_transport import SerialTransport, format_angles_packet
from .servo_calibration import ServoCalibration

__all__ = [
    "RealHead",
    "SERVO_CHANNELS",
    "Ch15SerialTransport",
    "JawCommandResult",
    "JawHardwareHead",
    "NormalizedPulseCalibration",
    "PROVISIONAL_CH15_JAW_CALIBRATION",
    "SerialTransport",
    "ServoCalibration",
    "format_angles_packet",
]
