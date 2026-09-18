"""Lazy CH15-only serial transport for the provisional jaw calibration mode."""

import time

from .jaw_calibration import CH15_PROVISIONAL_MAX_PULSE, CH15_PROVISIONAL_MIN_PULSE
from .serial_transport import (
    DEFAULT_BAUD_RATE,
    DEFAULT_SERIAL_PORT,
    ConnectionFactory,
    SerialConnection,
    SerialTransport,
)


class Ch15SerialTransport:
    """Send only CH15 manual-calibration commands, never ``ANGLES`` packets."""

    def __init__(
        self,
        port: str = DEFAULT_SERIAL_PORT,
        baudrate: int = DEFAULT_BAUD_RATE,
        connection_factory: ConnectionFactory | None = None,
        boot_wait_seconds: float = 2.0,
        response_timeout_seconds: float = 3.0,
    ) -> None:
        if boot_wait_seconds < 0 or response_timeout_seconds < 0:
            raise ValueError("serial boot wait and response timeout must be non-negative")
        self.port = port
        self.baudrate = baudrate
        self._connection_factory = connection_factory
        self._boot_wait_seconds = boot_wait_seconds
        self._response_timeout_seconds = response_timeout_seconds
        self._connection: SerialConnection | None = None
        self._enabled = False

    def enable(self) -> None:
        """Open the port lazily and opt in to the Arduino's CH15 test mode."""
        # Opening /dev/ttyACM0 can reset a Mega. Wait before writing so the
        # command is not confused with boot output or lost during reset.
        self._get_connection()
        time.sleep(self._boot_wait_seconds)
        self._write_command("CH15_TEST_ENABLE")
        response = self._wait_for_response("OK CH15 jaw test enabled")
        if not response.startswith("OK CH15 jaw test enabled"):
            raise RuntimeError(
                "Arduino did not confirm CH15 manual enable: "
                f"{response or 'no response'}"
            )
        self._enabled = True

    def send_pulse(self, pulse: int) -> str:
        """Send one already-calibrated CH15 pulse and return Arduino's reply."""
        if isinstance(pulse, bool) or not isinstance(pulse, int):
            raise TypeError("CH15 pulse must be an integer")
        if not CH15_PROVISIONAL_MIN_PULSE <= pulse <= CH15_PROVISIONAL_MAX_PULSE:
            raise ValueError(
                "CH15 pulse must be within the provisional range "
                f"{CH15_PROVISIONAL_MIN_PULSE}..{CH15_PROVISIONAL_MAX_PULSE}"
            )
        if not self._enabled:
            raise RuntimeError("send CH15_TEST_ENABLE before a CH15 pulse")

        self._write_command(f"CH15_SET,{pulse}")
        response = self._wait_for_response(f"OK CH15 jaw pulse set to {pulse}")
        if not response.startswith(f"OK CH15 jaw pulse set to {pulse}"):
            raise RuntimeError(
                f"Arduino did not confirm CH15 pulse {pulse}: "
                f"{response or 'no response'}"
            )
        return response

    def stop(self) -> None:
        """Release CH15 on a clean application exit without touching other channels."""
        if self._enabled:
            self._write_command("CH15_TEST_STOP")
            self._enabled = False

    def close(self) -> None:
        """Close a previously opened serial connection without opening a new one."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _write_command(self, command: str) -> None:
        connection = self._get_connection()
        connection.write((command + "\n").encode("ascii"))
        connection.flush()

    def _get_connection(self) -> SerialConnection:
        if self._connection is None:
            factory = self._connection_factory or SerialTransport._default_connection_factory
            self._connection = factory(self.port, self.baudrate)
        return self._connection

    def _wait_for_response(self, expected_prefix: str) -> str:
        """Read through boot noise until a matching response or deadline."""
        connection = self._get_connection()
        readline = getattr(connection, "readline", None)
        if not callable(readline):
            raise RuntimeError("serial connection does not support Arduino responses")

        last_response = ""
        deadline = time.monotonic() + self._response_timeout_seconds
        while True:
            response = readline()
            if isinstance(response, bytes):
                response = response.decode("utf-8", errors="replace").strip()
            else:
                response = str(response).strip()

            if response:
                last_response = response
                if response.startswith(expected_prefix) or response.startswith("ERR "):
                    return response

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return last_response

            # Real pyserial ``readline`` already waits for its timeout. This
            # small wait prevents a fake/non-blocking connection from busy-looping.
            time.sleep(min(0.01, remaining))
