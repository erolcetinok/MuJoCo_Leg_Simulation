"""Serial bridge to firmware/leg_controller/leg_controller.ino.

Wire format: ``q,<q0>,<q1>,...,<q[N-1]>\\n`` (radians) where N = number of
joints in CONFIG.joints and the order is canonical YAML order (leg-major:
FL,FR,BL,BR × shoulder,wing,knee). Firmware echoes
``ok <q0> <q1> ... <q[N-1]>`` after clamping.

set_joint_targets is fire-and-forget: it does not wait for a reply. The
firmware uses SoftwareSerial on the UNO R3 for host I/O, and the
hardware-Serial DXL transactions starve that interrupt enough to drop
~15% of host bytes during fast control loops. Blocking on a per-command
reply would hang for the full readline timeout on every drop. Replies
that have already arrived are drained into ``last_reply`` opportunistically
so jog/GUI status text still works; we just never wait.

The host always sends all N joint angles in one line, even if the caller
only updated a subset — the backend keeps a `_targets` cache for the rest
(initialized to zeros). This is what lets single-leg CLIs like quad-jog
or quad-swing-hw drive a single leg without having to know the other
nine joints exist.

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
        # Per-joint cached target in canonical order; merged on every send so
        # callers can update one leg at a time.
        self._targets: dict[str, float] = {name: 0.0 for name in CONFIG.joint_names}

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
        unknown = set(q) - set(joint_names)
        if unknown:
            raise KeyError(
                f"ArduinoBackend.set_joint_targets got unknown joint keys "
                f"{sorted(unknown)}; expected subset of {joint_names}"
            )
        # Merge caller's dict over cached targets, then serialize in canonical
        # YAML order. Subset updates are fine; firmware always sees all N.
        for name, value in q.items():
            self._targets[name] = float(value)
        values = [self._targets[name] for name in joint_names]
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
