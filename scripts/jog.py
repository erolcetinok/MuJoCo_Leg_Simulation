#!/usr/bin/env python3
"""Interactive joint-angle jogger for leg_controller.ino.

Type three radians (shoulder, wing, knee) separated by spaces or commas.
Each command is sent over the USB-to-TTL adapter; the port stays open for
the duration of the session, so commands are near-instant (no Arduino
reset per command). Ctrl-D or Ctrl-C to exit.

The port is the ADAPTER's, not the Arduino's onboard USB. On Mac that's
usually /dev/cu.usbserial-XXXX or /dev/cu.SLAB_USBtoUART. Run
`python scripts/usb_console.py --list` to see what's plugged in.

Usage:
  python scripts/jog.py --port /dev/cu.usbserial-A1234
  SERIAL_PORT=/dev/cu.usbserial-A1234 python scripts/jog.py

Example session:
  > 0 0 0
  ok 0.00 0.00 0.00
  > 0.5 -0.3 0.2
  ok 0.50 -0.30 0.20
  > -1 0.5 -1.5
  ok -1.00 0.50 -1.50
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import serial
except ImportError:
    print("Need pyserial: pip install pyserial", file=sys.stderr)
    sys.exit(1)


def parse_three_floats(line: str) -> tuple[float, float, float] | None:
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", "-p", default=None,
                        help="Adapter serial port (e.g. /dev/cu.usbserial-A1234)")
    parser.add_argument("--baud", "-b", type=int, default=57600)
    args = parser.parse_args()

    port = args.port or os.environ.get("SERIAL_PORT")
    if not port:
        print("Specify --port or set SERIAL_PORT", file=sys.stderr)
        sys.exit(1)

    with serial.Serial(port, args.baud, timeout=1.0) as ser:
        ser.reset_input_buffer()
        print(f"Connected to {port}. Type three radians (shoulder wing knee). Ctrl-D to exit.\n")

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            angles = parse_three_floats(line)
            if angles is None:
                print("  ? need three numbers, e.g.  0.5 -0.3 0.2", file=sys.stderr)
                continue
            q1, q2, q3 = angles
            try:
                ser.write(f"q,{q1},{q2},{q3}\n".encode("ascii"))
                reply = ser.readline().decode("ascii", errors="ignore").strip()
            except serial.SerialException as e:
                print(f"  ! serial error: {e}", file=sys.stderr)
                break
            print(f"  {reply}" if reply else "  (no reply)")


if __name__ == "__main__":
    main()
