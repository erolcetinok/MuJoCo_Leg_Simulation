"""Identify the servo on the bus: plug one in, get its ID and joint name.

The cross-check between the label on a servo and what configs/robot.yaml
thinks that ID is. `dxl_scan.py` answers "are all twelve there"; this answers
"which one am I holding", one servo at a time, without a wall of NO RESPONSE
rows for the eleven that are unplugged.

    python scripts/dxl_whoami.py -p /dev/ttyUSB0    # loop: plug in, Enter, repeat
    python scripts/dxl_whoami.py --once             # single shot, then exit

Every supported bus rate is swept, so a factory-fresh servo still at 57600
reports just as clearly as a configured one at 115200.
"""
import argparse
import os
import sys

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped import provisioning
from quadruped.backends.dynamixel_backend import (
    ADDR_HARDWARE_ERROR, ADDR_PRESENT_POSITION, ADDR_PRESENT_TEMPERATURE,
    COMM_SUCCESS, PROTOCOL_VERSION, TICKS_PER_REV,
)
from quadruped.calibration import ask
from quadruped.config import CONFIG

ADDR_PRESENT_INPUT_VOLTAGE = 144

ERROR_BITS = {
    0: "input voltage",
    2: "overheating",
    3: "motor encoder",
    4: "electrical shock",
    5: "overload",
}


def joint_name_for(motor_id: int) -> str:
    for j in CONFIG.joints:
        if j.motor_id == motor_id:
            return j.name
    return "NOT IN robot.yaml"


def read_byte(packet, port, motor_id, addr):
    value, result, error = packet.read1ByteTxRx(port, motor_id, addr)
    return value if (result == COMM_SUCCESS and error == 0) else None


def report(packet, port, baud, found) -> None:
    for motor_id in sorted(found):
        name = joint_name_for(motor_id)
        pos, result, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS or error != 0:
            pos = None
        volt = read_byte(packet, port, motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
        temp = read_byte(packet, port, motor_id, ADDR_PRESENT_TEMPERATURE)
        err = read_byte(packet, port, motor_id, ADDR_HARDWARE_ERROR)

        deg = f"{pos * 360.0 / TICKS_PER_REV:.1f} deg" if pos is not None else "?"
        status = "ok" if err == 0 else (
            ";".join(n for bit, n in ERROR_BITS.items() if err & (1 << bit))
            if err else "unread")
        print(f"  ID {motor_id:>2} at {baud} baud  ->  {name}")
        print(f"     model {found[motor_id]}, pos {deg}, "
              f"{volt / 10.0 if volt is not None else '?'} V, "
              f"{temp if temp is not None else '?'} C, status {status}")

    if len(found) > 1:
        print(f"  WARNING: {len(found)} servos answered. Connect exactly one to be "
              f"sure which is which.")


def scan_once(packet, port) -> int:
    baud, found = provisioning.discover(packet, port)
    if not found:
        rates = ", ".join(str(b) for b in provisioning.SEARCH_BAUDS)
        print(f"  nothing answered (tried {rates} baud). Check power and the TTL cable.")
        return 1
    report(packet, port, baud, found)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", "-p", default=None,
                        help="U2D2 serial port (or the SERIAL_PORT env var).")
    parser.add_argument("--once", action="store_true",
                        help="Scan once and exit instead of looping on Enter.")
    args = parser.parse_args()

    try:
        from dynamixel_sdk import PortHandler, PacketHandler
    except ImportError:
        print("Need the DYNAMIXEL SDK: pip install dynamixel-sdk", file=sys.stderr)
        return 2

    port_name = args.port or os.environ.get("SERIAL_PORT")
    if not port_name:
        print("No serial port: pass --port or set SERIAL_PORT.", file=sys.stderr)
        return 2

    port = PortHandler(port_name)
    packet = PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        print(f"Could not open {port_name}", file=sys.stderr)
        return 2

    try:
        if args.once:
            return scan_once(packet, port)
        print(f"Reading {port_name}. Plug in ONE servo, then Enter (q = quit).")
        while True:
            if ask("\n> ").startswith("q"):
                return 0
            scan_once(packet, port)
    except (KeyboardInterrupt, EOFError):
        print()
        return 0
    finally:
        port.closePort()


if __name__ == "__main__":
    sys.exit(main())
