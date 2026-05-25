"""Serial bridge to firmware/leg_controller/leg_controller.ino.

Wire format: ``q,<q1>,<q2>,<q3>\\n`` (radians, shoulder/wing/knee), one line per
command. Firmware echoes ``ok <q1> <q2> <q3>`` after clamping.

set_joint_targets is fire-and-forget: it does not wait for a reply. The
firmware uses SoftwareSerial on the UNO R3 for host I/O, and the
hardware-Serial DXL transactions starve that interrupt enough to drop
~15% of host bytes during fast control loops. Blocking on a per-command
reply would hang for the full readline timeout on every drop. Replies
that have already arrived are drained into ``last_reply`` opportunistically
so jog/GUI status text still works; we just never wait.

read_joint_state returns empty dicts — the firmware doesn't stream state.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from quadruped.backends.base import RobotBackend
from quadruped.config import CONFIG


class ArduinoBackend(RobotBackend):
    def __init__(
        self,
        port: Optional[str] = None,
        *,
        baud: Optional[int] = None,
        timeout: float = 2.0,
    ) -> None:
        self._port = port or os.environ.get("SERIAL_PORT")
        if not self._port:
            raise ValueError(
                "ArduinoBackend needs a serial port: pass port= or set SERIAL_PORT."
            )
        self._baud = baud if baud is not None else CONFIG.serial.baud_host
        self._timeout = timeout
        self._ser = None
        self._last_reply: str = ""

    def connect(self) -> None:
        try:
            import serial
        except ImportError:
            sys.stderr.write("Need pyserial: pip install pyserial\n")
            raise
        self._ser = serial.Serial(self._port, self._baud, timeout=self._timeout)
        self._ser.reset_input_buffer()

    def disconnect(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def set_joint_targets(self, q: dict[str, float]) -> None:
        assert self._ser is not None, "call connect() first (or use context manager)"
        joint_names = CONFIG.joint_names
        try:
            values = [float(q[name]) for name in joint_names]
        except KeyError as exc:
            raise KeyError(
                f"ArduinoBackend.set_joint_targets requires keys {joint_names}; missing {exc}"
            ) from exc
        line = "q," + ",".join(f"{v}" for v in values) + "\n"
        self._ser.write(line.encode("ascii"))
        # Opportunistic, non-blocking drain. Do not wait for THIS command's
        # reply — see module docstring.
        n = self._ser.in_waiting
        if n:
            chunk = self._ser.read(n).decode("ascii", errors="ignore")
            lines = [s.strip() for s in chunk.splitlines() if s.strip()]
            if lines:
                self._last_reply = lines[-1]

    def read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        # Firmware does not stream state today. Future: parse a queryable
        # `?` command, or read DXL present-position via the sketch.
        return {}, {}

    @property
    def last_reply(self) -> str:
        return self._last_reply

    @property
    def port(self) -> str:
        return self._port
