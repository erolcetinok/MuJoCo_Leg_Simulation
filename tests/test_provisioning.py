"""Servo provisioning tests, against a fake dynamixel_sdk packet/port pair.

The real SDK isn't a test dependency, and the thing worth pinning here is
sequence, not I/O: write ID before baud, never write EEPROM with torque on,
refuse a crowded bus, and treat a lost ack on the baud write as inconclusive
rather than fatal.
"""
import pytest

from quadruped.backends.dynamixel_backend import ADDR_TORQUE_ENABLE
from quadruped.provisioning import (
    ADDR_BAUD_RATE, ADDR_ID, ADDR_RETURN_DELAY_TIME, BAUD_CODES, ProvisionError,
    discover, ping, provision, sole_servo,
)

CODE_TO_BAUD = {code: baud for baud, code in BAUD_CODES.items()}
FACTORY_RETURN_DELAY = 250


class _FakePort:
    def __init__(self):
        self.baud = None
        self.baud_history = []
        self.set_baud_ok = True

    def setBaudRate(self, baud):
        if not self.set_baud_ok:
            return False
        self.baud = baud
        self.baud_history.append(baud)
        return True


class _FakePacket:
    """A bus holding `present` ({id: model}) at `at_baud`.

    Writes mutate that state the way a servo would: an ID write moves the servo,
    a baud write moves the whole bus. Addresses in `write_fails` return a comm
    error instead.
    """

    def __init__(self, present=None, at_baud=57600):
        self.present = dict(present if present is not None else {1: 1060})
        self.at_baud = at_baud
        self.writes = []
        self.write_fails = set()
        self.regs = {}

    def broadcastPing(self, port):
        if port.baud != self.at_baud:
            return {}, -3001
        return {i: [model, 52] for i, model in self.present.items()}, 0

    def ping(self, port, dxl_id):
        if port.baud != self.at_baud or dxl_id not in self.present:
            return 0, -3001, 0
        return self.present[dxl_id], 0, 0

    def write1ByteTxRx(self, port, dxl_id, addr, value):
        self.writes.append((dxl_id, addr, value))
        if addr in self.write_fails:
            return -1000, 0
        if addr == ADDR_ID:
            self.present[value] = self.present.pop(dxl_id)
        elif addr == ADDR_BAUD_RATE:
            self.at_baud = CODE_TO_BAUD[value]
        else:
            self.regs[(dxl_id, addr)] = value
        return 0, 0

    def read1ByteTxRx(self, port, dxl_id, addr):
        return self.regs.get((dxl_id, addr), FACTORY_RETURN_DELAY), 0, 0

    def getTxRxResult(self, result):
        return f"comm {result}"

    def getRxPacketError(self, error):
        return f"servo error {error}"


def _bus(present=None, at_baud=57600):
    return _FakePacket(present, at_baud), _FakePort()


# --- discovery ---------------------------------------------------------------

def test_discover_finds_the_baud_the_servo_is_actually_at():
    packet, port = _bus({1: 1060}, at_baud=115200)
    baud, found = discover(packet, port)
    assert baud == 115200
    assert found == {1: 1060}


def test_discover_reports_nothing_on_a_silent_bus():
    packet, port = _bus({}, at_baud=57600)
    assert discover(packet, port) == (None, {})


def test_discover_tries_factory_baud_first():
    """A fresh servo shares the bus with a configured chain often enough that
    the search order is load-bearing, not cosmetic."""
    packet, port = _bus({1: 1060}, at_baud=57600)
    discover(packet, port)
    assert port.baud_history[0] == 57600


def test_discover_skips_bauds_the_port_rejects():
    packet, port = _bus({1: 1060}, at_baud=115200)
    port.set_baud_ok = False
    assert discover(packet, port) == (None, {})


# --- crowded-bus policy ------------------------------------------------------

def test_sole_servo_returns_the_only_id():
    assert sole_servo({7: 1060}) == 7


def test_sole_servo_refuses_a_crowded_bus():
    with pytest.raises(ProvisionError, match="2 servos"):
        sole_servo({1: 1060, 2: 1060})


def test_sole_servo_refuses_an_empty_bus():
    with pytest.raises(ProvisionError, match="nothing answered"):
        sole_servo({})


def test_sole_servo_honours_an_explicit_choice():
    assert sole_servo({1: 1060, 2: 1060}, prefer=2) == 2


def test_sole_servo_rejects_a_choice_that_is_not_there():
    with pytest.raises(ProvisionError, match="did not answer"):
        sole_servo({1: 1060}, prefer=5)


# --- provisioning ------------------------------------------------------------

def test_provision_moves_a_factory_servo_to_its_target():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    changed = provision(packet, port, current_id=1, current_baud=57600,
                        target_id=9, target_baud=115200)
    assert packet.present == {9: 1060}
    assert packet.at_baud == 115200
    assert port.baud == 115200
    assert any("id 1 -> 9" in c for c in changed)
    assert any("baud 57600 -> 115200" in c for c in changed)


def test_provision_torques_off_before_touching_eeprom():
    """EEPROM writes are silently rejected while torque is on."""
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    provision(packet, port, current_id=1, current_baud=57600,
              target_id=9, target_baud=115200)
    assert packet.writes[0] == (1, ADDR_TORQUE_ENABLE, 0)


def test_provision_writes_id_before_baud():
    """Reversed, every later write would have to go out at the new rate."""
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    provision(packet, port, current_id=1, current_baud=57600,
              target_id=9, target_baud=115200)
    addrs = [addr for _, addr, _ in packet.writes]
    assert addrs.index(ADDR_ID) < addrs.index(ADDR_BAUD_RATE)


def test_provision_sets_return_delay_to_zero_by_default():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    provision(packet, port, current_id=1, current_baud=57600,
              target_id=9, target_baud=115200)
    assert (9, ADDR_RETURN_DELAY_TIME, 0) in packet.writes


def test_provision_leaves_return_delay_alone_when_asked():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    provision(packet, port, current_id=1, current_baud=57600,
              target_id=9, target_baud=115200, return_delay=None)
    assert not any(addr == ADDR_RETURN_DELAY_TIME for _, addr, _ in packet.writes)


def test_provision_skips_writes_that_would_change_nothing():
    packet, port = _bus({9: 1060}, at_baud=115200)
    port.setBaudRate(115200)
    packet.regs[(9, ADDR_RETURN_DELAY_TIME)] = 0
    changed = provision(packet, port, current_id=9, current_baud=115200,
                        target_id=9, target_baud=115200)
    assert changed == []
    assert [addr for _, addr, _ in packet.writes] == [ADDR_TORQUE_ENABLE]


def test_provision_survives_a_lost_ack_on_the_baud_write():
    """That status packet goes out at the old rate; losing it is routine, and
    the servo has still changed. Only the ping at the new rate decides."""
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)

    real_write = packet.write1ByteTxRx

    def drop_ack(p, dxl_id, addr, value):
        result = real_write(p, dxl_id, addr, value)
        return (-1000, 0) if addr == ADDR_BAUD_RATE else result

    packet.write1ByteTxRx = drop_ack
    provision(packet, port, current_id=1, current_baud=57600,
              target_id=9, target_baud=115200)
    assert packet.at_baud == 115200


def test_provision_raises_when_the_servo_is_silent_afterwards():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    packet.write_fails = {ADDR_ID}
    with pytest.raises(ProvisionError, match="set ID"):
        provision(packet, port, current_id=1, current_baud=57600,
                  target_id=9, target_baud=115200)


def test_provision_raises_when_the_id_write_lands_but_nothing_answers():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    packet.write_fails = {ADDR_BAUD_RATE}
    with pytest.raises(ProvisionError, match="no reply"):
        provision(packet, port, current_id=1, current_baud=57600,
                  target_id=9, target_baud=115200)


def test_provision_rejects_a_baud_the_servo_cannot_do():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    with pytest.raises(ProvisionError, match="not an XL430 bus rate"):
        provision(packet, port, current_id=1, current_baud=57600,
                  target_id=9, target_baud=250000)
    assert packet.writes == []


def test_ping_returns_none_for_a_missing_servo():
    packet, port = _bus({1: 1060}, at_baud=57600)
    port.setBaudRate(57600)
    assert ping(packet, port, 1) == 1060
    assert ping(packet, port, 4) is None


# --- the register table itself ----------------------------------------------

def test_baud_codes_match_the_xl430_control_table():
    """https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/ addr 8."""
    assert BAUD_CODES == {9600: 0, 57600: 1, 115200: 2, 1000000: 3,
                          2000000: 4, 3000000: 5, 4000000: 6, 4500000: 7}


def test_project_baud_is_reachable():
    from quadruped.config import CONFIG
    assert CONFIG.serial.baud_dxl in BAUD_CODES
