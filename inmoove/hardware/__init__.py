"""Step 5 hardware-boundary utilities for calibrated face commands.

This package does not open serial ports or drive servos on import.
"""

from .real_head import RealHead, SERVO_CHANNELS
from .serial_transport import SerialTransport, format_angles_packet
from .servo_calibration import ServoCalibration

__all__ = [
    "RealHead",
    "SERVO_CHANNELS",
    "SerialTransport",
    "ServoCalibration",
    "format_angles_packet",
]
