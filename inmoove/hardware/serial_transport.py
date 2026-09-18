"""Explicit, hardware-optional transport for 16 calibrated servo angles."""

from collections.abc import Callable, Sequence
import math
from numbers import Real
from typing import Protocol


SERVO_COUNT = 16
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUD_RATE = 115200


class AngleTransport(Protocol):
    """Minimal boundary that lets ``RealHead`` be tested without hardware."""

    def send_angles(self, angles: Sequence[float]) -> None:
        """Send exactly 16 channel-ordered servo angles."""


class SerialConnection(Protocol):
    """The subset of a pyserial connection used by this module."""

    def write(self, data: bytes) -> object:
        """Write one packet."""

    def flush(self) -> object:
        """Flush buffered packet data."""

    def close(self) -> object:
        """Close the connection."""


ConnectionFactory = Callable[[str, int], SerialConnection]


def format_angles_packet(angles: Sequence[float]) -> bytes:
    """Encode a newline-delimited ``ANGLES`` packet in fixed CH0~CH15 order.

    The 0~180 degree check is a wire-protocol sanity check only.  Per-servo
    safety limits belong to ``ServoCalibration`` and are applied before this
    boundary.
    """
    values = tuple(angles)
    if len(values) != SERVO_COUNT:
        raise ValueError(f"servo packet requires exactly {SERVO_COUNT} angles")

    checked_values = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"angle at channel {index} must be a real number")

        angle = float(value)
        if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
            raise ValueError(
                f"angle at channel {index} must be a finite value within 0~180"
            )
        checked_values.append(angle)

    payload = "ANGLES," + ",".join(f"{angle:.6f}" for angle in checked_values)
    return (payload + "\n").encode("ascii")


class SerialTransport:
    """Lazy pyserial transport; it opens no port until ``send_angles`` is called."""

    def __init__(
        self,
        port: str = DEFAULT_SERIAL_PORT,
        baudrate: int = DEFAULT_BAUD_RATE,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self._connection_factory = connection_factory
        self._connection: SerialConnection | None = None

    def send_angles(self, angles: Sequence[float]) -> None:
        """Open the configured serial port lazily and send one validated packet."""
        packet = format_angles_packet(angles)
        connection = self._get_connection()
        connection.write(packet)
        connection.flush()

    def close(self) -> None:
        """Close a previously opened serial connection, if any."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _get_connection(self) -> SerialConnection:
        if self._connection is None:
            factory = self._connection_factory or self._default_connection_factory
            self._connection = factory(self.port, self.baudrate)

        return self._connection

    @staticmethod
    def _default_connection_factory(port: str, baudrate: int) -> SerialConnection:
        try:
            import serial
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "pyserial is required for real serial transport; install requirements.txt"
            ) from error

        return serial.Serial(port=port, baudrate=baudrate, timeout=1)
