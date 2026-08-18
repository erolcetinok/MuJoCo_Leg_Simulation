"""Assign DYNAMIXEL IDs, bus baud and return delay, one servo at a time.

Every XL430 ships as ID 1 at 57600 baud, so twelve fresh servos on one bus all
answer the same packet. This walks the whole set: connect one servo, press
Enter, it becomes the ID configs/robot.yaml wants, repeat. The Pi-side
replacement for DYNAMIXEL Wizard 2.0 and for the Arduino path's
firmware/configure_motor.ino.

    python scripts/dxl_id.py -p /dev/ttyUSB0             # all twelve, in order
    python scripts/dxl_id.py -p /dev/ttyUSB0 --leg FL    # one leg
    python scripts/dxl_id.py -p /dev/ttyUSB0 --ids 9     # redo a single servo
    python scripts/dxl_id.py -p /dev/ttyUSB0 --dry-run   # report, change nothing

ONE SERVO ON THE BUS AT A TIME. The script refuses to write while more than one
answers: with two factory-fresh servos connected, a write to ID 1 hits both.

Exits 1 if any target was left unassigned. Verify the finished chain with
scripts/dxl_scan.py.
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
from quadruped.backends.dynamixel_backend import PROTOCOL_VERSION
from quadruped.calibration import ask
from quadruped.config import CONFIG


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


def select_targets(args) -> list:
    """The joints to provision, in canonical YAML order."""
    joints = list(CONFIG.joints)
    if args.leg:
        joints = list(CONFIG.joints_for_leg(args.leg))
    if args.ids:
        wanted = set(parse_ids(args.ids))
        joints = [j for j in joints if j.motor_id in wanted]
        unknown = wanted - {j.motor_id for j in CONFIG.joints}
        if unknown:
            raise SystemExit(f"no joint in robot.yaml uses motor_id {sorted(unknown)}")
    return joints


def assign(packet, port, joint, args) -> bool:
    baud, found = provisioning.discover(packet, port)
    if not found:
        rates = ", ".join(str(b) for b in provisioning.SEARCH_BAUDS)
        print(f"  nothing answered (tried {rates} baud). Check power and the TTL cable.")
        return False

    listing = ", ".join(f"ID {i} (model {m})" for i, m in sorted(found.items()))
    print(f"  found at {baud} baud: {listing}")

    try:
        source = provisioning.sole_servo(found, prefer=args.from_id)
    except provisioning.ProvisionError as exc:
        print(f"  SKIP: {exc}")
        return False

    if joint.motor_id in found and joint.motor_id != source:
        print(f"  SKIP: ID {joint.motor_id} is already held by another servo on this bus")
        return False

    if args.dry_run:
        print(f"  dry run: would set ID {source} -> {joint.motor_id}, "
              f"baud {baud} -> {args.baud}, return delay {args.return_delay}")
        return True

    try:
        changed = provisioning.provision(
            packet, port,
            current_id=source, current_baud=baud,
            target_id=joint.motor_id, target_baud=args.baud,
            return_delay=args.return_delay,
        )
    except provisioning.ProvisionError as exc:
        print(f"  FAIL: {exc}")
        return False

    detail = "; ".join(changed) if changed else "already correct"
    print(f"  OK: ID {joint.motor_id} at {args.baud} baud ({detail})")
    print(f"  label this servo {joint.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", "-p", default=None,
                        help="U2D2 serial port (or the SERIAL_PORT env var).")
    parser.add_argument("--baud", "-b", type=int, default=CONFIG.serial.baud_dxl,
                        help="Bus baud to leave each servo at (default %(default)s "
                             "from configs/robot.yaml).")
    parser.add_argument("--leg", choices=list(CONFIG.legs), default=None,
                        help="Only this leg's three joints (default: all twelve).")
    parser.add_argument("--ids", default=None,
                        help="Only these target motor IDs, e.g. 9 or 1-3 or 1,4,7.")
    parser.add_argument("--from-id", type=int, default=None,
                        help="Which ID on the bus to reconfigure, when more than one "
                             "answers. Without it, a crowded bus is refused.")
    parser.add_argument("--return-delay", type=int,
                        default=provisioning.DEFAULT_RETURN_DELAY,
                        help="Return Delay Time register, in 2 us units "
                             "(default %(default)s; the factory value is 250).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what is on the bus and what would change; write nothing.")
    args = parser.parse_args()

    if args.baud not in provisioning.BAUD_CODES:
        print(f"{args.baud} is not an XL430 bus rate "
              f"(pick from {sorted(provisioning.BAUD_CODES)})", file=sys.stderr)
        return 2

    try:
        from dynamixel_sdk import PortHandler, PacketHandler
    except ImportError:
        print("Need the DYNAMIXEL SDK: pip install dynamixel-sdk", file=sys.stderr)
        return 2

    port_name = args.port or os.environ.get("SERIAL_PORT")
    if not port_name:
        print("No serial port: pass --port or set SERIAL_PORT.", file=sys.stderr)
        return 2

    targets = select_targets(args)
    if not targets:
        print("No targets selected.", file=sys.stderr)
        return 2

    port = PortHandler(port_name)
    packet = PacketHandler(PROTOCOL_VERSION)
    if not port.openPort():
        print(f"Could not open {port_name}", file=sys.stderr)
        return 2

    print(f"Provisioning {len(targets)} servo(s) on {port_name}.")
    print("Connect ONE servo at a time. Ctrl-C aborts; nothing is written until "
          "you press Enter.")

    done, skipped = [], []
    try:
        for index, joint in enumerate(targets):
            print(f"\n[{joint.name}] target ID {joint.motor_id}")
            reply = ask("  connect ONLY this servo, then Enter (s = skip, q = quit): ")
            if reply.startswith("q"):
                skipped.extend(targets[index:])
                break
            if reply.startswith("s"):
                skipped.append(joint)
                continue
            (done if assign(packet, port, joint, args) else skipped).append(joint)
    except KeyboardInterrupt:
        print("\naborted")
        return 1
    finally:
        port.closePort()

    print(f"\n{len(done)} of {len(targets)} assigned.")
    if skipped:
        print("not assigned: " + ", ".join(f"{j.name} (ID {j.motor_id})" for j in skipped))
        return 1
    print("Chain them all back together, then verify:")
    print(f"  python scripts/dxl_scan.py -p {port_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
