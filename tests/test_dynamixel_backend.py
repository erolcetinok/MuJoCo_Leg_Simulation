"""DynamixelBackend tests.

Two layers:
  * pure firmware-parity math (rad<->ticks), no SDK needed
  * connect/write/read plumbing, exercised against a fake dynamixel_sdk module
    injected into sys.modules (the real SDK isn't a test dependency).
"""
import math
import sys
import types

import numpy as np
import pytest

from quadruped.backends import DynamixelBackend
from quadruped.config import CONFIG


# --- firmware-parity math (no SDK) ------------------------------------------

def _firmware_ticks(j, q):
    q = min(max(q, j.limit_rad[0]), j.limit_rad[1])
    return round((j.direction * math.degrees(q) + j.offset_deg) * 4096 / 360)


def test_rad_to_ticks_matches_firmware_formula():
    b = DynamixelBackend(port="x")
    for j in CONFIG.joints:
        for q in (-1.2, -0.3, 0.0, 0.5, 1.4):
            assert b._rad_to_ticks(j, q) == _firmware_ticks(j, q)


def test_zero_rad_is_mid_scale():
    # offset_deg=180 → q=0 maps to 2048 ticks (servo centre), like the firmware.
    b = DynamixelBackend(port="x")
    assert b._rad_to_ticks(CONFIG.joint("shoulder_FL"), 0.0) == 2048


def test_tick_roundtrip_within_quantization():
    b = DynamixelBackend(port="x")
    j = CONFIG.joint("shoulder_FL")
    for q in np.linspace(-1.4, 1.4, 21):
        back = b._ticks_to_rad(j, b._rad_to_ticks(j, q))
        assert abs(back - q) < 1e-3  # one tick ≈ 0.0015 rad


def test_clamps_beyond_limit():
    b = DynamixelBackend(port="x")
    j = CONFIG.joint("wing_FL")
    assert b._rad_to_ticks(j, 5.0) == b._rad_to_ticks(j, j.limit_rad[1])
    assert b._rad_to_ticks(j, -5.0) == b._rad_to_ticks(j, j.limit_rad[0])


def test_direction_sign_inverts():
    # synthetic joint with direction=-1 to cover the sign path (config is all +1)
    j = types.SimpleNamespace(direction=-1, offset_deg=180.0, limit_rad=(-1.57, 1.57))
    b = DynamixelBackend(port="x")
    assert b._ticks_to_rad(j, b._rad_to_ticks(j, 0.7)) == pytest.approx(0.7, abs=1e-3)


# --- plumbing against a fake SDK --------------------------------------------

class _FakePort:
    def __init__(self, name): self.name = name; self.opened = False; self.baud = None
    def openPort(self): self.opened = True; return True
    def setBaudRate(self, b): self.baud = b; return True
    def closePort(self): self.opened = False


class _FakePacket:
    """Every servo answers unless it is in `silent`, and every write succeeds
    unless its (addr) is in `write_fails`."""

    silent: set = set()
    write_fails: set = set()
    reads: dict = {}

    def __init__(self, ver):
        self.writes = []
        self.pings = []

    def _fail_for(self, mid, addr):
        if mid in self.silent:
            return (-1000, 0)          # COMM_TX_FAIL
        if addr in self.write_fails:
            return (0, 1)              # dxl_error: result ok, servo unhappy
        return (0, 0)

    def ping(self, port, mid):
        self.pings.append(mid)
        return (1060, -1000, 0) if mid in self.silent else (1060, 0, 0)

    def write1ByteTxRx(self, port, mid, addr, val):
        self.writes.append((mid, addr, val)); return self._fail_for(mid, addr)

    def write4ByteTxRx(self, port, mid, addr, val):
        self.writes.append((mid, addr, val)); return self._fail_for(mid, addr)

    def read1ByteTxRx(self, port, mid, addr):
        if mid in self.silent:
            return (0, -1000, 0)
        return (self.reads.get((mid, addr), 0), 0, 0)

    def getTxRxResult(self, result): return f"comm {result}"
    def getRxPacketError(self, error): return f"servo error {error}"


class _FakeSyncWrite:
    tx_result = 0

    def __init__(self, port, packet, addr, length): self.params = {}; self.tx = 0
    def clearParam(self): self.params = {}
    def addParam(self, mid, data): self.params[mid] = list(data); return True
    def txPacket(self): self.tx += 1; return type(self).tx_result


class _FakeSyncRead:
    def __init__(self, port, packet, addr, length): self.ids = []; self.data = {}
    def addParam(self, mid): self.ids.append(mid); return True
    def txRxPacket(self): return 0
    def isAvailable(self, mid, addr, length): return mid in self.data
    def getData(self, mid, addr, length): return self.data[mid][(addr, length)]


@pytest.fixture
def fake_sdk(monkeypatch):
    # Class attributes carry the failure knobs, so reset them per test.
    monkeypatch.setattr(_FakePacket, "silent", set())
    monkeypatch.setattr(_FakePacket, "write_fails", set())
    monkeypatch.setattr(_FakePacket, "reads", {})
    monkeypatch.setattr(_FakeSyncWrite, "tx_result", 0)
    mod = types.ModuleType("dynamixel_sdk")
    mod.PortHandler = _FakePort
    mod.PacketHandler = _FakePacket
    mod.GroupSyncWrite = _FakeSyncWrite
    mod.GroupSyncRead = _FakeSyncRead
    monkeypatch.setitem(sys.modules, "dynamixel_sdk", mod)
    return mod


def test_connect_opens_port_and_configures_servos(fake_sdk):
    b = DynamixelBackend(port="/dev/fake", baud=115200)
    b.connect()
    assert b._port.opened and b._port.baud == 115200
    assert b._sync_read.ids == [j.motor_id for j in CONFIG.joints]  # all 12 registered
    # every servo got torque/mode/profile writes
    for j in CONFIG.joints:
        addrs = [addr for (mid, addr, _) in b._packet.writes if mid == j.motor_id]
        assert 64 in addrs and 11 in addrs and 112 in addrs  # torque, op mode, profile vel


def test_set_joint_targets_writes_all_twelve(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    b.set_joint_targets({"shoulder_FL": 0.0})  # subset update; rest cached at 0
    assert len(b._sync_write.params) == 12 and b._sync_write.tx == 1
    # shoulder_FL at q=0 → 2048 ticks, little-endian
    assert b._sync_write.params[CONFIG.joint("shoulder_FL").motor_id] == [0x00, 0x08, 0x00, 0x00]


def test_set_joint_targets_rejects_unknown_key(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    with pytest.raises(KeyError):
        b.set_joint_targets({"not_a_joint": 0.0})


def test_read_joint_state_decodes_pos_vel_current(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    mid = CONFIG.joint("shoulder_FL").motor_id
    b._sync_read.data = {mid: {(132, 4): 2048, (128, 4): 100, (126, 2): 50}}
    qpos, qvel = b.read_joint_state()
    assert qpos["shoulder_FL"] == pytest.approx(0.0, abs=1e-3)          # 2048 ticks → 0 rad
    assert qvel["shoulder_FL"] == pytest.approx(100 * 0.229 * 2 * math.pi / 60, abs=1e-6)
    assert b.present_current()["shoulder_FL"] == pytest.approx(50 * 2.69 / 1000, abs=1e-6)


def test_foot_force_matches_pure_function(fake_sdk):
    from quadruped.kinematics.jacobian import foot_force

    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    # seed each FL motor with known raw pos/vel/current
    raws = {"shoulder_FL": (2100, 0, 40),
            "wing_FL": (1900, 0, -30),
            "knee_FL": (2300, 0, 25)}
    b._sync_read.data = {
        CONFIG.joint(n).motor_id: {(132, 4): p, (128, 4): v, (126, 2): c}
        for n, (p, v, c) in raws.items()
    }
    force = b.foot_force("FL")

    # Recompute independently with the joint order pinned EXPLICITLY (shoulder,
    # wing, knee) — not via leg_joint_names — so a wrong ordering in the method
    # would fail this test instead of cancelling out.
    qpos, _ = b.read_joint_state()
    cur = b.present_current()
    expected = foot_force(
        (qpos["shoulder_FL"], qpos["wing_FL"], qpos["knee_FL"]),
        (cur["shoulder_FL"], cur["wing_FL"], cur["knee_FL"]),
    )
    assert np.allclose(force, expected)
    assert force.shape == (3,)


def test_disconnect_torques_off_and_closes(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    b._packet.writes.clear()
    b.disconnect()
    # every servo got a torque-disable (addr 64, val 0) and the port closed
    assert all((j.motor_id, 64, 0) in b._packet.writes for j in CONFIG.joints)
    assert b._port is None


# --- failure paths: the whole point of the hardening pass -------------------

def test_connect_pings_before_configuring(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    assert b._packet.pings == [j.motor_id for j in CONFIG.joints]


def test_connect_raises_naming_the_silent_ids(fake_sdk, monkeypatch):
    monkeypatch.setattr(_FakePacket, "silent", {5, 11})
    b = DynamixelBackend(port="/dev/fake")
    with pytest.raises(ConnectionError) as exc:
        b.connect()
    assert "5" in str(exc.value) and "11" in str(exc.value)
    assert b._port is None                       # port released, not leaked


def test_connect_raises_on_a_rejected_configuration_write(fake_sdk, monkeypatch):
    monkeypatch.setattr(_FakePacket, "write_fails", {11})   # operating mode
    b = DynamixelBackend(port="/dev/fake")
    with pytest.raises(ConnectionError) as exc:
        b.connect()
    assert "position mode" in str(exc.value)


def test_isolated_syncwrite_failure_is_tolerated(fake_sdk, monkeypatch):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    monkeypatch.setattr(_FakeSyncWrite, "tx_result", -1000)
    b.set_joint_targets({"shoulder_FL": 0.0})    # one dropped packet: keep going
    monkeypatch.setattr(_FakeSyncWrite, "tx_result", 0)
    b.set_joint_targets({"shoulder_FL": 0.0})    # recovered, counter resets
    monkeypatch.setattr(_FakeSyncWrite, "tx_result", -1000)
    b.set_joint_targets({"shoulder_FL": 0.0})
    b.set_joint_targets({"shoulder_FL": 0.0})
    with pytest.raises(ConnectionError):          # third in a row: the bus is gone
        b.set_joint_targets({"shoulder_FL": 0.0})


def test_read_joint_state_skips_motors_that_did_not_answer(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    mid = CONFIG.joint("wing_FL").motor_id
    b._sync_read.data = {mid: {(132, 4): 2048, (128, 4): 0, (126, 2): 0}}
    qpos, qvel = b.read_joint_state()
    assert set(qpos) == {"wing_FL"} and set(qvel) == {"wing_FL"}


def test_foot_force_raises_on_an_incomplete_read(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    b._sync_read.data = {CONFIG.joint("shoulder_FL").motor_id:
                         {(132, 4): 2048, (128, 4): 0, (126, 2): 0}}
    with pytest.raises(RuntimeError, match="incomplete bus read"):
        b.foot_force("FL")


def test_set_torque_all_writes_every_servo(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    b._packet.writes.clear()
    b.set_torque_all(False)
    assert all((j.motor_id, 64, 0) in b._packet.writes for j in CONFIG.joints)


def test_set_profile_velocity_retunes_every_servo(fake_sdk):
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    b._packet.writes.clear()
    b.set_profile_velocity(30)
    assert all((j.motor_id, 112, 30) in b._packet.writes for j in CONFIG.joints)
    assert b._profile_velocity == 30


def test_health_check_reports_faults_and_temperature(fake_sdk, monkeypatch):
    mid = CONFIG.joint("knee_BR").motor_id
    monkeypatch.setattr(_FakePacket, "reads", {(mid, 70): 0x20, (mid, 146): 61})
    b = DynamixelBackend(port="/dev/fake")
    b.connect()
    health = b.health_check()
    assert len(health) == 12
    assert health["knee_BR"] == (0x20, 61)       # overload bit latched, 61 C


def test_ticks_are_clamped_to_a_single_turn():
    """A miscalibrated offset must not wrap around the servo's 0..4095 range."""
    j = types.SimpleNamespace(direction=1, offset_deg=350.0, limit_rad=(-1.57, 1.57))
    b = DynamixelBackend(port="x")
    assert b._rad_to_ticks(j, 1.5) == 4095
    j_low = types.SimpleNamespace(direction=1, offset_deg=10.0, limit_rad=(-1.57, 1.57))
    assert b._rad_to_ticks(j_low, -1.5) == 0
