"""Ping the DYNAMIXEL bus and report what is actually on it.

The power-on smoke test for the U2D2 path: before any control code runs, prove
that every servo enumerates at the expected ID and baud, that none has latched a
hardware fault, and that the bus voltage is sane.

    python scripts/dxl_scan.py -p /dev/ttyUSB0
    python scripts/dxl_scan.py -p /dev/ttyUSB0 --sweep-baud    # nothing answered
    python scripts/dxl_scan.py -p /dev/ttyUSB0 --ids 1-3       # bench leg only

Exits 1 if any expected ID is silent, so it works as a bringup gate.
"""
import argparse
import os
import sys

# Run directly (`python scripts/x.py`) without depending on `pip install -e .`:
# put src/ on the path before importing quadruped. The editable install's .pth
# is unreliable on macOS (see docs/STATUS.md), and this costs nothing.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "src"))

from quadruped.backends.dynamixel_backend import (
    ADDR_HARDWARE_ERROR, ADDR_PRESENT_POSITION, ADDR_PRESENT_TEMPERATURE,
    COMM_SUCCESS, PROTOCOL_VERSION, TICKS_PER_REV,
)
from quadruped.config import CONFIG

ADDR_FIRMWARE_VERSION = 6
ADDR_PRESENT_INPUT_VOLTAGE = 144

# XL430-supported bus rates, most likely first. A factory-fresh servo is at
# 57600; the ones this project configures are at 115200.
SWEEP_BAUDS = (57600, 115200, 1000000, 2000000, 9600)

# Hardware Error Status bit meanings (XL430 control table, addr 70).
ERROR_BITS = {
    0: "input voltage",
    2: "overheating",
    3: "motor encoder",
    4: "electrical shock",
    5: "overload",
}


def parse_ids(spec: str) -> list:
    """`1-12` or `1,4,7` or `1-3,10` -> a sorted list of ints."""
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def describe_errors(byte: int) -> str:
    if byte == 0:
        return "ok"
    return ";".join(name for bit, name in ERROR_BITS.items() if byte & (1 << bit)) or f"0x{byte:02x}"


def name_for_id(motor_id: int) -> str:
    for j in CONFIG.joints:
        if j.motor_id == motor_id:
            return j.name
    return "-"


def scan(packet, port, ids: list) -> dict:
    """{motor_id: dict of readings} for every ID that answers a ping."""
    found = {}
    for motor_id in ids:
        model, result, error = packet.ping(port, motor_id)
        if result != COMM_SUCCESS or error != 0:
            continue
        row = {"model": model}
        for key, addr, reader in (
            ("fw", ADDR_FIRMWARE_VERSION, packet.read1ByteTxRx),
            ("err", ADDR_HARDWARE_ERROR, packet.read1ByteTxRx),
            ("temp", ADDR_PRESENT_TEMPERATURE, packet.read1ByteTxRx),
            ("volt", ADDR_PRESENT_INPUT_VOLTAGE, packet.read2ByteTxRx),
            ("pos", ADDR_PRESENT_POSITION, packet.read4ByteTxRx),
        ):
            value, result, error = reader(port, motor_id, addr)
            row[key] = value if (result == COMM_SUCCESS and error == 0) else None
        found[motor_id] = row
    return found


def print_table(ids: list, found: dict) -> None:
    print(f"{'ID':>3}  {'joint':<12} {'model':>6} {'fw':>3} "
          f"{'pos(ticks)':>10} {'pos(deg)':>9} {'V':>5} {'temp(C)':>7}  status")
    for motor_id in ids:
        row = found.get(motor_id)
        if row is None:
            print(f"{motor_id:>3}  {name_for_id(motor_id):<12} "
                  f"{'-':>6} {'-':>3} {'-':>10} {'-':>9} {'-':>5} {'-':>7}  NO RESPONSE")
            continue
        pos = row["pos"]
        deg = f"{pos * 360.0 / TICKS_PER_REV:9.1f}" if pos is not None else f"{'-':>9}"
        volt = f"{row['volt'] / 10.0:5.1f}" if row["volt"] is not None else f"{'-':>5}"
        status = describe_errors(row["err"]) if row["err"] is not None else "unread"
        print(f"{motor_id:>3}  {name_for_id(motor_id):<12} "
              f"{row['model']:>6} {row['fw'] if row['fw'] is not None else '-':>3} "
              f"{pos if pos is not None else '-':>10} {deg} {volt} "
              f"{row['temp'] if row['temp'] is not None else '-':>7}  {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", "-p", default=None,
                        help="U2D2 serial port (or the SERIAL_PORT env var).")
    parser.add_argument("--baud", "-b", type=int, default=None,
                        help=f"DXL bus baud (default {CONFIG.serial.baud_dxl} from configs/robot.yaml).")
    parser.add_argument("--ids", default=None,
                        help="IDs to probe, e.g. 1-12 or 1,4,7 (default: every motor_id in robot.yaml).")
    parser.add_argument("--sweep-baud", action="store_true",
                        help="Retry every supported bus rate until something answers. "
                             "Use when a factory-fresh servo is still at 57600.")
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

    ids = parse_ids(args.ids) if args.ids else [j.motor_id for j in CONFIG.joints]
    bauds = SWEEP_BAUDS if args.sweep_baud else (args.baud or CONFIG.serial.baud_dxl,)

    port = PortHandler(port_name)
    packet = PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        print(f"Could not open {port_name}", file=sys.stderr)
        return 2

    try:
        for baud in bauds:
            if not port.setBaudRate(baud):
                print(f"Could not set baud {baud}", file=sys.stderr)
                continue
            print(f"\nprobing {port_name} at {baud} baud, IDs {ids[0]}-{ids[-1]}")
            found = scan(packet, port, ids)
            print_table(ids, found)
            if found:
                missing = [i for i in ids if i not in found]
                if missing:
                    print(f"\nFAIL: {len(missing)} of {len(ids)} silent: {missing}")
                    return 1
                print(f"\nOK: all {len(ids)} servos responded at {baud} baud.")
                return 0
        print("\nFAIL: nothing on the bus. Check power, the U2D2 data link, and "
              "try --sweep-baud.")
        return 1
    finally:
        port.closePort()


if __name__ == "__main__":
    sys.exit(main())
