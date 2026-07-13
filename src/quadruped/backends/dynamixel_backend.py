"""U2D2 + DYNAMIXEL SDK backend — direct USB→Dynamixel bus, no Arduino/firmware.

Unlike ArduinoBackend (which streams radians to leg_controller.ino and lets the
*firmware* apply sign/offset/limits/tick-conversion), this backend talks straight
to the servos through the DYNAMIXEL SDK. So it must reproduce that firmware math
itself — using the same configs/robot.yaml direction / offset_deg / limit_rad —
so the same calibration produces identical motion to the Arduino path and to sim.

Protocol 2.0, XL430 control table. Goal positions go out in one GroupSyncWrite
per command; present current/velocity/position come back in one GroupSyncRead.

Install the SDK on the Pi:  pip install dynamixel-sdk
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from quadruped.backends.base import RobotBackend
from quadruped.config import CONFIG
from quadruped.kinematics.jacobian import foot_force as _foot_force

# --- XL430 / Protocol 2.0 control table (addr, length in bytes) -------------
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126   # start of the contiguous read block
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_BLOCK = 10       # current(2) + velocity(4) + position(4), contiguous

OP_MODE_POSITION = 3
PROTOCOL_VERSION = 2.0
TICKS_PER_REV = 4096

# XL430 unit conversions (ROBOTIS e-Manual control table)
VELOCITY_UNIT_RPM = 0.229    # present-velocity LSB → rev/min
CURRENT_UNIT_MA = 2.69       # present-current LSB → mA


def _signed(value: int, bits: int) -> int:
    """Reinterpret an unsigned register read as two's-complement signed."""
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


class DynamixelBackend(RobotBackend):
    """Drives the 12 servos over a U2D2 with the DYNAMIXEL SDK.

    profile_velocity: 0 = move to each goal as fast as possible (matches the
    firmware's streaming convention; required for smooth gait). For first
    power-on / calibration, pass a small value (e.g. 30) so the legs move
    gently instead of snapping to the commanded pose.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        *,
        baud: Optional[int] = None,
        profile_velocity: int = 0,
    ) -> None:
        self._port_name = port or os.environ.get("SERIAL_PORT")
        self._baud = baud if baud is not None else CONFIG.serial.baud_dxl
        self._profile_velocity = profile_velocity
        # Joint metadata in canonical YAML order (same order everything uses).
        self._joints = list(CONFIG.joints)
        self._by_name = {j.name: j for j in self._joints}

        # SDK handles, created in connect().
        self._port = None
        self._packet = None
        self._sync_write = None
        self._sync_read = None

        # Cached targets so callers can update one leg at a time (subset writes).
        self._targets: dict[str, float] = {j.name: 0.0 for j in self._joints}
        self._last_current: dict[str, float] = {}

    # --- radian <-> tick conversion (mirrors leg_controller.ino exactly) -----

    def _rad_to_ticks(self, joint, q: float) -> int:
        lo, hi = joint.limit_rad
        q = min(max(q, lo), hi)                       # clamp, like the firmware
        deg = joint.direction * math.degrees(q) + joint.offset_deg
        return int(round(deg * TICKS_PER_REV / 360.0))

    def _ticks_to_rad(self, joint, ticks: int) -> float:
        deg = ticks * 360.0 / TICKS_PER_REV
        return math.radians((deg - joint.offset_deg) / joint.direction)

    @staticmethod
    def _le4(value: int) -> list[int]:
        """Pack a 32-bit int little-endian for GroupSyncWrite."""
        value &= 0xFFFFFFFF
        return [value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF, (value >> 24) & 0xFF]

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        try:
            from dynamixel_sdk import (
                PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
            )
        except ImportError:
            raise ImportError("Need the DYNAMIXEL SDK: pip install dynamixel-sdk")
        if not self._port_name:
            raise ValueError("DynamixelBackend needs a serial port: pass port= or set SERIAL_PORT.")

        self._port = PortHandler(self._port_name)
        self._packet = PacketHandler(PROTOCOL_VERSION)
        if not self._port.openPort():
            raise ConnectionError(f"could not open U2D2 port {self._port_name!r}")
        if not self._port.setBaudRate(self._baud):
            raise ConnectionError(f"could not set baud {self._baud} on {self._port_name!r}")

        self._sync_write = GroupSyncWrite(
            self._port, self._packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self._sync_read = GroupSyncRead(
            self._port, self._packet, ADDR_PRESENT_CURRENT, LEN_PRESENT_BLOCK)
        for j in self._joints:
            self._sync_read.addParam(j.motor_id)

        # Configure each servo: torque off → position mode + profile velocity →
        # torque on. Torque-on holds present position (no jump) until the first
        # goal is written.
        for j in self._joints:
            self._packet.write1ByteTxRx(self._port, j.motor_id, ADDR_TORQUE_ENABLE, 0)
            self._packet.write1ByteTxRx(self._port, j.motor_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)
            self._packet.write4ByteTxRx(self._port, j.motor_id, ADDR_PROFILE_VELOCITY, self._profile_velocity)
            self._packet.write1ByteTxRx(self._port, j.motor_id, ADDR_TORQUE_ENABLE, 1)

    def disconnect(self) -> None:
        if self._port is not None:
            for j in self._joints:
                try:
                    self._packet.write1ByteTxRx(self._port, j.motor_id, ADDR_TORQUE_ENABLE, 0)
                except Exception:
                    pass
            self._port.closePort()
            self._port = None

    # --- command ------------------------------------------------------------

    def set_joint_targets(self, q: dict[str, float]) -> None:
        assert self._sync_write is not None, "call connect() first (or use context manager)"
        unknown = set(q) - set(self._targets)
        if unknown:
            raise KeyError(
                f"DynamixelBackend.set_joint_targets got unknown joint keys "
                f"{sorted(unknown)}; expected subset of {list(self._targets)}"
            )
        # Merge over cache so callers can update a single leg; firmware-parity
        # math turns radians → ticks; one GroupSyncWrite for all 12.
        for name, value in q.items():
            self._targets[name] = float(value)
        self._sync_write.clearParam()
        for j in self._joints:
            ticks = self._rad_to_ticks(j, self._targets[j.name])
            self._sync_write.addParam(j.motor_id, self._le4(ticks))
        self._sync_write.txPacket()

    def write_goal_ticks(self, name: str, ticks: int) -> None:
        """Move ONE servo to a raw tick position, bypassing the rad/offset math.
        For calibration: position a single joint exactly without disturbing the
        others and without depending on the offset we're still discovering."""
        assert self._packet is not None, "call connect() first"
        j = self._by_name[name]
        self._packet.write4ByteTxRx(self._port, j.motor_id, ADDR_GOAL_POSITION, int(ticks) & 0xFFFFFFFF)

    def set_torque(self, name: str, on: bool) -> None:
        """Enable/disable torque on one servo (so it can be hand-posed, etc.)."""
        assert self._packet is not None, "call connect() first"
        j = self._by_name[name]
        self._packet.write1ByteTxRx(self._port, j.motor_id, ADDR_TORQUE_ENABLE, 1 if on else 0)

    # --- feedback -----------------------------------------------------------

    def read_joint_state(self) -> tuple[dict[str, float], dict[str, float]]:
        """One GroupSyncRead → (qpos rad, qvel rad/s). Present current is stashed
        for present_current() (used for current-based touchdown detection)."""
        assert self._sync_read is not None, "call connect() first"
        self._sync_read.txRxPacket()
        qpos: dict[str, float] = {}
        qvel: dict[str, float] = {}
        cur: dict[str, float] = {}
        for j in self._joints:
            if not self._sync_read.isAvailable(j.motor_id, ADDR_PRESENT_POSITION, 4):
                continue
            pos_raw = _signed(self._sync_read.getData(j.motor_id, ADDR_PRESENT_POSITION, 4), 32)
            vel_raw = _signed(self._sync_read.getData(j.motor_id, ADDR_PRESENT_VELOCITY, 4), 32)
            cur_raw = _signed(self._sync_read.getData(j.motor_id, ADDR_PRESENT_CURRENT, 2), 16)
            qpos[j.name] = self._ticks_to_rad(j, pos_raw)
            # direction flips the joint's sense for velocity/current too.
            qvel[j.name] = j.direction * vel_raw * VELOCITY_UNIT_RPM * 2.0 * math.pi / 60.0
            cur[j.name] = j.direction * cur_raw * CURRENT_UNIT_MA / 1000.0  # amps
        self._last_current = cur
        return qpos, qvel

    def present_current(self) -> dict[str, float]:
        """Per-joint current (amps) from the last read_joint_state(). A spike at
        touchdown is the cheap contact signal (Stanford Pupper / mjbots trick)."""
        return dict(self._last_current)

    def foot_force(self, leg: str) -> np.ndarray:
        """Estimated Cartesian foot force (N, leg-local) from the DXL current.

        Does its OWN single read_joint_state() so position and current come from
        the same bus read: present_current() returns only that read's stashed
        current (there is no stashed position), so pairing it with a separate
        read would mismatch the two. LORIS-style proprioceptive contact sensing.

        Raises RuntimeError if the bus read came back incomplete for this leg
        (read_joint_state skips any motor whose GroupSyncRead was unavailable —
        common under DXL contention). That is a clear, catchable signal so a
        gait/contact loop can skip the cycle rather than hit a bare KeyError.
        """
        names = CONFIG.leg_joint_names(leg)   # (shoulder, wing, knee) order
        qpos, _ = self.read_joint_state()
        cur = self.present_current()
        missing = [n for n in names if n not in qpos or n not in cur]
        if missing:
            raise RuntimeError(
                f"foot_force({leg!r}): incomplete bus read, missing {missing}"
            )
        return _foot_force(
            tuple(qpos[n] for n in names),
            tuple(cur[n] for n in names),
        )

    @property
    def port(self) -> str:
        return self._port_name
