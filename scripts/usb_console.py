#!/usr/bin/env python3
"""
Interactive USB serial console — same traffic as Arduino Serial Monitor, from the terminal.

For this project, the port is the USB-to-TTL adapter on D7/D8, not the
Arduino's onboard USB. On Mac that's usually /dev/cu.usbserial-XXXX or
/dev/cu.SLAB_USBtoUART. You still upload sketches with Arduino IDE (or
arduino-cli) via the Arduino's onboard USB — this script only replaces
Serial Monitor for talking to the running sketch.

Requires: pip install pyserial

Examples:
  python3 usb_console.py --list
  python3 usb_console.py --port /dev/cu.usbserial-A1234
  SERIAL_PORT=/dev/cu.usbserial-A1234 python3 usb_console.py

Exit miniterm: Ctrl+]  then type  q  (or see miniterm help).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def try_list_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    return list(list_ports.comports())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List serial ports and exit",
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Device path (e.g. /dev/cu.usbmodem1101 on Mac)",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=115200,
        help="Baud rate (default 115200, same as leg_controller USB serial)",
    )
    args = parser.parse_args()

    if args.list:
        ports = try_list_ports()
        if ports is None:
            print("Install pyserial: pip install pyserial", file=sys.stderr)
            sys.exit(1)
        if not ports:
            print("No serial ports found.")
            return
        for p in ports:
            print(f"{p.device}\t{p.description or ''}")
        return

    port = args.port or os.environ.get("SERIAL_PORT")
    if not port:
        ports = try_list_ports()
        if ports is None:
            print("Install pyserial: pip install pyserial", file=sys.stderr)
            sys.exit(1)
        # Prefer cu.* on Mac (call-out = safe for capture); exclude Bluetooth SPP
        candidates = [
            p for p in ports
            if p.device.startswith("/dev/cu.")
            and "Bluetooth" not in (p.description or "")
        ]
        if len(candidates) == 1:
            port = candidates[0].device
            print(f"Using {port}", file=sys.stderr)
        elif not candidates:
            print(
                "No /dev/cu.* USB serial port found. Plug in the USB-to-TTL adapter "
                "(and/or the Arduino), run with --list, then pass --port explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print("Multiple ports — pick the adapter (usually /dev/cu.usbserial-* or", file=sys.stderr)
            print("/dev/cu.SLAB_USBtoUART), NOT /dev/cu.usbmodem* which is the Arduino:\n", file=sys.stderr)
            for p in candidates:
                print(f"  {p.device}\t{p.description or ''}", file=sys.stderr)
            print("\nThen: python3 usb_console.py --port /dev/cu....", file=sys.stderr)
            sys.exit(1)

    # pyserial's miniterm (same package as serial.Serial)
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "serial.tools.miniterm",
                port,
                str(args.baud),
            ]
        )
    )


if __name__ == "__main__":
    main()
