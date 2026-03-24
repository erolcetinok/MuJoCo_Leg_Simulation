#!/usr/bin/env python3
# send three joint angles to the leg controller over USB serial
# order: shoulder, wing, knee (same as MuJoCo single_leg.xml)
# requires: pyserial  (pip install pyserial)
# Usage:
# python send_angles.py <q1> <q2> <q3>   [--port PORT]
# python send_angles.py 0.0 0.0 0.0
# python send_angles.py 0.1 -0.2 0.5 --port /dev/cu.usbmodem14101
import argparse
import sys

try:
    import serial
except ImportError:
    print("Need pyserial: pip install pyserial", file=sys.stderr)
    sys.exit(1)

import argparse
import sys

try:
    import serial
except ImportError:
    print("Need pyserial: pip install pyserial", file=sys.stderr)
    sys.exit(1)


def send_angles(port: str, q1: float, q2: float, q3: float, baud: int = 115200) -> None:
    # Open serial port, send one line 'q,q1,q2,q3', read one reply line, close
    with serial.Serial(port, baud, timeout=2.0) as ser:
        line = f"q,{q1},{q2},{q3}\n"
        ser.write(line.encode("ascii"))
        reply = ser.readline().decode("ascii", errors="ignore").strip()
        if reply:
            print(reply)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send joint angles (radians) to leg controller: shoulder, wing, knee."
    )
    parser.add_argument("q1", type=float, help="Shoulder angle (rad)")
    parser.add_argument("q2", type=float, help="Wing angle (rad)")
    parser.add_argument("q3", type=float, help="Knee angle (rad)")
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port (e.g. /dev/cu.usbmodem14101 on Mac, COM3 on Windows)",
    )
    parser.add_argument("--baud", "-b", type=int, default=115200, help="Baud rate (default 115200)")
    args = parser.parse_args()

    port = args.port
    if not port:
        # Try common Mac names; user can set PORT env or use --port
        import os
        port = os.environ.get("SERIAL_PORT")
    if not port:
        print("Specify port: --port /dev/cu.usbmodem... or set SERIAL_PORT", file=sys.stderr)
        sys.exit(1)

    send_angles(port, args.q1, args.q2, args.q3, args.baud)


if __name__ == "__main__":
    main()
