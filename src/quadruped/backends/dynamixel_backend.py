"""Future U2D2 + DynamixelSDK backend (stub).

Adopting this deletes the Arduino + SoftwareSerial + strtod parser layer:
direct USB → Dynamixel bus over U2D2 (~$50) + Power Hub (~$15), full Protocol
2.0, sync/bulk read+write at 333+ Hz, no SoftwareSerial 57600 ceiling, no AVR
``%f`` parsing surprises. Worth revisiting before Phase 5 when 12 servos make
the Arduino bridge harder to justify.

Sketch:
    from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
    port = PortHandler("/dev/cu.usbserial-FT...")
    packet = PacketHandler(2.0)
    port.openPort(); port.setBaudRate(CONFIG.serial.baud_dxl)
    # write/read goal position addresses per XL430 control table
"""
from __future__ import annotations

from quadruped.backends.base import RobotBackend


class DynamixelBackend(RobotBackend):
    def connect(self) -> None:
        raise NotImplementedError(
            "DynamixelBackend is a stub. See module docstring for the migration "
            "path (U2D2 + DynamixelSDK)."
        )

    def disconnect(self) -> None: ...
    def set_joint_targets(self, q): ...
    def read_joint_state(self): return {}, {}
