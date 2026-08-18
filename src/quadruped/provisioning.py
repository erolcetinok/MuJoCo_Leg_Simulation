"""One-time servo provisioning: ID, bus baud, and return delay.

Every XL430 ships as ID 1 at 57600 baud, so twelve fresh servos on one bus all
answer the same packet and nothing works until each has a distinct ID. This
module walks ONE connected servo from whatever it currently is to what
configs/robot.yaml expects. The Arduino path did this with
firmware/configure_motor.ino; the Pi path is scripts/dxl_id.py, and this is the
part of it tests can reach.

Order is forced by the hardware: EEPROM writes are rejected while torque is on,
and a baud change takes effect the moment the servo acknowledges it. So it goes
torque off, ID, return delay, baud last, then re-open the port at the new rate
and ping to confirm.
"""
import time

from quadruped.backends.dynamixel_backend import ADDR_TORQUE_ENABLE, COMM_SUCCESS

ADDR_ID = 7
ADDR_BAUD_RATE = 8
ADDR_RETURN_DELAY_TIME = 9

# XL430 Baud Rate register (addr 8): bus rate -> register value. Factory is 1.
BAUD_CODES = {
    9600: 0, 57600: 1, 115200: 2, 1000000: 3,
    2000000: 4, 3000000: 5, 4000000: 6, 4500000: 7,
}

# Searched in this order so a factory-fresh servo (57600) is found before the
# already-configured chain (115200) it may be sharing a bus with.
SEARCH_BAUDS = (57600, 115200, 1000000, 2000000, 9600)

# An EEPROM write needs a moment to land before the servo will answer again.
EEPROM_SETTLE_S = 0.05

# Return Delay Time is in 2 us units; 0 is the LORIS setting and the single
# biggest sync-read/write throughput win on a U2D2 (see docs/power_rationale.md).
DEFAULT_RETURN_DELAY = 0


class ProvisionError(RuntimeError):
    """A servo refused a write, or vanished mid-change."""


def _check(packet, result, error, what):
    if result != COMM_SUCCESS:
        raise ProvisionError(f"{what}: {packet.getTxRxResult(result)}")
    if error != 0:
        raise ProvisionError(f"{what}: {packet.getRxPacketError(error)}")


def _write1(packet, port, dxl_id, addr, value, what):
    result, error = packet.write1ByteTxRx(port, dxl_id, addr, value)
    _check(packet, result, error, what)
    time.sleep(EEPROM_SETTLE_S)


def ping(packet, port, dxl_id):
    """Model number, or None if the servo does not answer."""
    model, result, error = packet.ping(port, dxl_id)
    if result != COMM_SUCCESS or error != 0:
        return None
    return model


def discover(packet, port, bauds=SEARCH_BAUDS):
    """Broadcast-ping at each rate. -> (baud, {id: model}), or (None, {}).

    Returns the first rate anything answered at rather than the union across
    rates: servos split across two bauds is the exact mess this tool exists to
    get you out of, and merging them into one result would hide it.
    """
    for baud in bauds:
        if not port.setBaudRate(baud):
            continue
        found, _ = packet.broadcastPing(port)
        if found:
            # broadcastPing yields {id: [model_number, firmware_version]}.
            return baud, {i: (v[0] if isinstance(v, (list, tuple)) else v)
                          for i, v in found.items()}
    return None, {}


def sole_servo(found, prefer=None):
    """The one ID it is safe to write to. Raises if the bus is empty or crowded.

    Crowded is the failure mode this guards: two factory-fresh servos both
    answer as ID 1, and a write to ID 1 then reconfigures both at once.
    """
    if not found:
        raise ProvisionError("nothing answered on the bus")
    if prefer is not None:
        if prefer not in found:
            raise ProvisionError(f"ID {prefer} did not answer (found {sorted(found)})")
        return prefer
    if len(found) > 1:
        raise ProvisionError(
            f"{len(found)} servos on the bus: {sorted(found)}. Connect exactly one, "
            f"or name the one you mean with --from-id.")
    return next(iter(found))


def provision(packet, port, *, current_id, current_baud, target_id, target_baud,
              return_delay=DEFAULT_RETURN_DELAY):
    """Move one servo to (target_id, target_baud). Returns a list of changes made.

    Raises ProvisionError if a write is rejected, or if the servo does not
    answer at its new settings afterwards.
    """
    if target_baud not in BAUD_CODES:
        raise ProvisionError(f"{target_baud} is not an XL430 bus rate "
                             f"(pick from {sorted(BAUD_CODES)})")

    changed = []
    _write1(packet, port, current_id, ADDR_TORQUE_ENABLE, 0, "torque off")

    if target_id != current_id:
        _write1(packet, port, current_id, ADDR_ID, target_id,
                f"set ID {current_id} -> {target_id}")
        changed.append(f"id {current_id} -> {target_id}")

    if return_delay is not None:
        value, result, error = packet.read1ByteTxRx(port, target_id, ADDR_RETURN_DELAY_TIME)
        _check(packet, result, error, "read return delay")
        if value != return_delay:
            _write1(packet, port, target_id, ADDR_RETURN_DELAY_TIME, return_delay,
                    "set return delay")
            changed.append(f"return delay {value} -> {return_delay}")

    if target_baud != current_baud:
        try:
            _write1(packet, port, target_id, ADDR_BAUD_RATE, BAUD_CODES[target_baud],
                    "set baud")
        except ProvisionError:
            # This write's status packet goes out at the OLD rate and is easy to
            # lose, so a missing ack does not mean the write failed. The ping
            # below at the new rate is the real verdict.
            pass
        changed.append(f"baud {current_baud} -> {target_baud}")

    if not port.setBaudRate(target_baud):
        raise ProvisionError(f"could not set the port to {target_baud} baud")
    time.sleep(EEPROM_SETTLE_S)
    if ping(packet, port, target_id) is None:
        raise ProvisionError(f"no reply from ID {target_id} at {target_baud} baud "
                             f"after the change; power cycle and re-scan")
    return changed
