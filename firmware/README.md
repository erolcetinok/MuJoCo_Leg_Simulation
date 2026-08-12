# Arduino UNO bridge — hardware setup

> **Which path am I on?** This is the **fallback**. The primary hardware path is
> a Raspberry Pi 5 + U2D2 driving the servos directly over the DYNAMIXEL SDK —
> see `docs/hardware_bringup.md` for the BOM and bringup order, and
> `docs/power_and_electronics.md` for the power chain. The U2D2 path needs no
> Arduino, no firmware flashing, and gives you state readback for free
> (`--backend dxl`).
>
> Keep this document for the case where the U2D2 disappoints: the UNO bridge
> works, it's fully understood, and it's what the first leg was brought up on.

Start-to-finish bring-up for the Arduino path: up to twelve daisy-chained
Dynamixels driven by an Arduino UNO R3, commanded over USB from your Mac.
Joint order everywhere is the canonical YAML order — legs **FL, FR, BL, BR** ×
joints **shoulder, wing, knee** — the same order used by the wire format, the
`syncWrite` packet, and `configs/robot.yaml`.

If you're bringing up a single leg on the bench, wire only FL (IDs 1-3); the
sketch reports `? no servo ID <n>` for the absent motors at boot and keeps
going.

The Arduino's onboard USB is **flash-only** for this project. Host comms (commands and replies) go through a small USB-to-TTL adapter wired to the Arduino's D7/D8 pins. The reason: on UNO R3, the Dynamixel Shield and the onboard USB-UART chip both live on the same hardware UART (D0/D1) and electrically fight each other for D0 — you cannot reliably read host bytes through the Arduino's USB while the shield is driving the DXL bus. See §5 for the full explanation. The adapter sidesteps the problem cleanly by putting host comms on a second, dedicated serial path that ROBOTIS designed the shield for.

---

## 1. Parts

| Part | Notes |
|------|-------|
| **Arduino UNO R3** | ATmega328P. Onboard USB used for flashing only. |
| **ROBOTIS DYNAMIXEL Shield** | TTL half-duplex driver + screw terminals + UART switch + Dynamixel power switch. |
| **12 × XL430-W250-T** servos | Daisy-chained. IDs **1-12** in per-leg blocks (FL 1-3, FR 4-6, BL 7-9, BR 10-12), each block shoulder / wing / knee. **Protocol 2.0**, **baud 115 200**. |
| **3-pin TTL JST cables** | XL430 uses TTL only — **never** the 4-pin RS-485 port on the shield. |
| **USB-to-TTL adapter** | CP2102, CH340, or FTDI. Any cheap module from Amazon. Carries host commands and replies. |
| **3× M/F jumper wires** | "Dupont" cables to connect adapter ↔ Arduino. |
| **Bench supply** | **6.5–12.0 V** (XL430's spec — the shield itself accepts 5–24 V, but the motors don't). ~5 A is enough for one leg on the bench; a full twelve-motor robot needs far more headroom — see `docs/power_and_electronics.md`. **Do not exceed 12.0 V.** |
| **USB-B cable** | UNO R3 to your Mac (flashing). |
| **USB-A cable for the adapter** | Whatever the adapter takes (USB-A or USB-C). |

Official references (only consult if something doesn't match below):
- [ROBOTIS DYNAMIXEL Shield](https://emanual.robotis.com/docs/en/parts/interface/dynamixel_shield/) — switches, jumpers, connectors.
- [XL430-W250-T](https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/) — voltage, control table.

---

## 2. Mac-side install

1. Install [Arduino IDE 2](https://www.arduino.cc/en/software).
2. **Tools → Board → Boards Manager** → install **Arduino AVR Boards** (it's usually pre-installed; Arduino UNO appears under it).
3. **Tools → Manage Libraries** → install **Dynamixel2Arduino**.
4. From the repo root (the directory containing `pyproject.toml`, one level above `firmware/`): `pip install -e .` — installs the `quadruped` Python package and pulls in pyserial/numpy/mujoco for the host-side scripts.
5. macOS bundles drivers for CP2102 and FTDI out of the box (macOS 11+). For CH340 modules, install [WCH's signed driver](https://www.wch-ic.com/downloads/CH34XSER_MAC_ZIP.html) if your adapter doesn't enumerate.

---

## 3. Configure the servos (IDs, baud, protocol)

Each motor must end up at a **unique ID in 1-12**, **baud 115 200**, **Protocol 2.0**. Factory defaults are **ID 1, baud 57 600** — fresh XL430s on the same bus will all answer to ID 1 and collide, which is why you configure one motor at a time below.

The ID map is fixed by `configs/robot.yaml` and must match exactly:

| Leg | Shoulder | Wing | Knee |
|-----|----------|------|------|
| FL  | 1 | 2 | 3 |
| FR  | 4 | 5 | 6 |
| BL  | 7 | 8 | 9 |
| BR  | 10 | 11 | 12 |

Label each motor physically as you configure it. Twelve identical servos on a bench with no labels is a bad afternoon.

> Why 115 200 and not 1 Mbps (the XL430's nominal max)? Because the hardware Serial bus only carries DXL traffic now (host is on the adapter), 1 Mbps is technically usable — but motors and sketches currently default to 115 200 and there's no benefit at this scale. Stick with 115 200 unless you have a reason.

**You need the shield powered and wired first** — do section 4, then come back here before uploading `leg_controller.ino`.

> Joint limits, motor IDs, baud rates, and zero offsets all live in `configs/robot.yaml` and are codegen'd into `firmware/leg_controller/robot_config.h` (which the sketch `#include`s). To change any of them, edit the YAML and run `python scripts/codegen.py` — do not hand-edit the `.h` file.

Use the repo-local helper sketch `firmware/configure_motor/configure_motor.ino`. It probes the motor across 57 600 / 115 200 / 1 000 000, sets it to your target ID, then changes its baud to 115 200 — all in one upload. Output prints to the adapter's terminal at **57 600** (host link baud — see §5).

Procedure — repeat once per motor (up to twelve times):

1. **Disconnect all motors except the one you're configuring.** Plug only that motor into the shield's TTL jack — they're all identical at this point.
2. Open `firmware/configure_motor/configure_motor.ino`.
3. Edit `#define SOURCE_ID` (match the motor's current ID, or 1 for factory-fresh) and `#define TARGET_ID` — the value from the ID table above.
4. Set UART switch to **Upload**, click **Upload** in the IDE.
5. Flip UART switch to **DYNAMIXEL**, press **RESET**, wait ~3 seconds.
6. Open a terminal on the adapter's port at **57600** baud (`python scripts/usb_console.py` does this, or any terminal of your choice). You should see `OK: motor at ID <n>, baud 115200` — the 115200 here is the *motor's* baud, which is independent of the host link.
7. Unplug that motor, plug in the next one, change `TARGET_ID`, repeat from step 4.

If you see `FAIL: no motor at ID <n> (tried baud 57600, 115200, 1000000)`, the motor is at an unexpected baud (run `find_motor.ino`), or power/cable/switch isn't right. If you see `FAIL: setID` or `FAIL: setBaudrate`, power-cycle the motor and try again — partial configuration is normal and the sketch handles it.

---

## 4. Physical assembly (power OFF)

1. Stack the **Dynamixel Shield** onto the UNO R3. Press until fully seated.
2. **Remove the VIN jumper cap** from the shield ("Disconnect VIN" mode) so Dynamixel power comes only from the shield's screw terminals, not the Arduino VIN path.
3. Wire bench supply **+ / −** to the shield's Dynamixel power screw terminals, matching the silkscreen exactly. Wrong polarity destroys hardware. Leave the supply OFF.
4. Daisy-chain the servos: **shield's 3-pin TTL JST socket → servo 1 (in) → servo 1 (out) → servo 2 (in) → servo 2 (out) → servo 3 (in)**. Mechanical order does not set IDs — that's done in section 3.
   - Use a **3-pin TTL** jack on the shield. The 4-pin RS-485 jack will not work with XL430-T.
5. Wire the **USB-to-TTL adapter** to the shield's exposed pin headers using three jumper wires:
   - adapter **TX**  → Arduino **D7**
   - adapter **RX**  → Arduino **D8**
   - adapter **GND** → Arduino **GND**
   - adapter **VCC** → **leave disconnected** (Arduino is powered via its own USB).
6. Plug both USB cables into your Mac:
   - **Arduino USB-B** → flashing.
   - **Adapter USB**   → host commands + sketch output.

```
   [ USB-to-TTL adapter ]                      [ Dynamixel Shield ]
     TX RX GND                                 ← 3-pin TTL JST to servos
      |  |  |
      |  |  └─── Arduino GND
      |  └─────── Arduino D8  (sketch TX)
      └────────── Arduino D7  (sketch RX)
                          ↑
                  [ Arduino UNO R3 ]
                          ↑
                       USB-B to Mac (flashing only)
```

---

## 5. How the dual-serial setup works (read before uploading)

UNO R3 has one hardware UART (`Serial` on D0/D1). The Dynamixel Shield uses those same pins for the DXL bus. **At runtime, the hardware Serial carries DXL bus traffic only** — the shield's transceiver actively drives D0/D1, and any other source on those pins gets electrically overridden.

That's why host comms go through `SoftwareSerial(7, 8)` and the USB-to-TTL adapter on D7/D8. The adapter exposes its own `/dev/cu.usbserial*` (or `/dev/cu.SLAB_USBtoUART`) port on your Mac, and that's the port your terminal, `send_angles.py`, and `jog.py` talk to. The Arduino's onboard USB-B (`/dev/cu.usbmodem*`) is used only for flashing.

**Host link runs at 57 600 baud.** All sketches and Python scripts default to this. `SoftwareSerial` on a 16 MHz AVR is too unreliable at 115 200 in practice — bytes get dropped or corrupted, especially on the Mac→Arduino direction. 57 600 is the documented fallback and works cleanly. The DXL bus on hardware Serial stays at 115 200, independent of the host link.

What this means in practice:
- The shield's **UART switch sits on DYNAMIXEL** during normal operation. Only flip to Upload for flashing.
- **Do not try to talk to the Arduino via its onboard USB monitor at runtime.** It's wired into the same UART the shield uses, so anything you type there gets blocked by the shield's transceiver. The adapter is the only host-comm path.
- `SoftwareSerial` occasionally still injects phantom noise bytes before a real line begins; `leg_controller.ino`'s loop discards anything before the `q,` prefix to defend against this.
- `sscanf("%f", ...)` is **not linked** in the default Arduino AVR build — `leg_controller.ino` parses floats with `strtod` instead. Don't switch back to `sscanf %f`; it'll silently fail.

---

## 6. Upload firmware

1. Open `firmware/leg_controller/leg_controller.ino` in Arduino IDE.
2. **Tools → Board → Arduino AVR Boards → Arduino UNO**.
3. **Tools → Port** → the **Arduino's** USB device (`/dev/cu.usbmodem...`). This is for flashing only; not the adapter's port.
4. Set UART switch on shield to **Upload**.
5. Click **Upload**. Wait for it to finish.
6. Set UART switch to **DYNAMIXEL**.
7. **Support the leg by hand** — when the first command lands, motors will snap to whatever position you send. Don't let it whip into something.
8. Turn the bench supply **ON** (correct voltage), then turn the shield's **Dynamixel power switch ON**.
9. Press **RESET** on the Arduino once.

---

## 7. Verify the controller is alive

Open a terminal on the **adapter's** port at **57600** baud. The easiest way:

```bash
python scripts/usb_console.py --port /dev/cu.usbserial-XXXX
```

(or run `python scripts/usb_console.py --list` to see what's plugged in — pick the adapter's port, not the Arduino's `/dev/cu.usbmodem*`).

Expected output after reset:

```
the controller is ready
ID 1 pos ticks=2048 deg=180.00
... (one line per motor found)
dynamixels ready (12 joints)
mode: usb command (q,q0,...,qN-1)
```

Then send twelve zeros + Enter (line ending Newline):

```
q,0,0,0,0,0,0,0,0,0,0,0,0
```

Reply should be `ok` followed by the twelve clamped values, and the legs snap to
the zero pose. Sending fewer than twelve values is rejected with
`? bad format at index <i>` or `? missing comma at index <i>`.

**If the terminal is blank or shows nothing:** baud rate or wrong port. Confirm 57 600 and that you picked the adapter's port. If still blank, double-check the three jumper wires (TX↔D7, RX↔D8, GND↔GND).

**If you see `? no servo ID <n>`** for any motor: IDs / baud / protocol don't match the firmware, or UART switch is still on Upload, or shield power is off, or the cable is in the RS-485 jack. Re-check sections 3, 4, and 6.

---

## 8. Drive the leg from Python

One-shot:

```bash
python scripts/send_angles.py 0.0 0.0 0.0 --port /dev/cu.usbserial-XXXX
```

Or set the port once per terminal session:

```bash
export SERIAL_PORT=/dev/cu.usbserial-XXXX
python scripts/send_angles.py 0.1 -0.2 0.5
```

Interactive (no per-command lag — port stays open), picking the leg:

```bash
python scripts/jog.py --leg FL --backend hw --port /dev/cu.usbserial-XXXX
```

```
> 0 0 0
  ok 0.000 0.000 0.000 ...
> 0.5 -0.3 0.2
  ok 0.500 -0.300 0.200 ...
```

You type three angles; they go to the chosen leg's shoulder/wing/knee, and the
other nine joints hold their cached pose. The reply echoes all twelve.

> `send_angles.py` and `send_foot.py` take the same `--leg` flag, defaulting to
> `FL`.

For a full step cycle rather than static poses, use `scripts/swing_hw.py
--backend hw --rate 33 --loop`.

`scripts/usb_console.py` is a terminal replacement for Arduino Serial Monitor — useful for one-off inspection. All of these default to baud **57 600** to match the sketch.

---

## 9. Protocol

- One line per command: `q,<q0>,<q1>,...,<q11>` — **twelve** radian values,
  newline-terminated, in canonical order (FL, FR, BL, BR × shoulder, wing, knee).
- Reply: `ok` followed by the twelve clamped values. Errors:
  `? bad format; use q,q0,q1,...`, `? bad format at index <i>`,
  `? missing comma at index <i>`, `? no servo ID <n>`, `? syncWrite failed err=<code>`.
- All twelve goals go out in **one `syncWrite` packet**, which is the point of
  the multi-leg refactor — twelve separate `setGoalPosition()` calls would not
  fit the timing budget.
- Joint limits (radians, enforced by the Arduino, identical for every leg):

  | Joint | Min | Max |
  |-------|-----|-----|
  | Shoulder | −π/2 | +π/2 |
  | Wing | −0.873 (≈ −50°) | +0.873 (≈ +50°) |
  | Knee | −2.007 (≈ −115°) | +π/2 |

Limits, IDs, offsets, and directions all come from `robot_config.h`, generated
from `configs/robot.yaml`. They are asserted against the MJCF at model-load
time, so sim and firmware can't silently disagree.

**This link is fire-and-forget by design.** `SoftwareSerial` drops roughly 15%
of host bytes while the DXL bus is busy, so `ArduinoBackend` does not wait for
the `ok` echo — a dropped command is simply superseded by the next one at the
streaming rate. Measured round-trip is ~19 ms, giving a practical ceiling of
**40–50 Hz**; use `--rate 33`. If you need reliable feedback, use the U2D2 path.

Motion speed is firmware-set, not commanded: `leg_controller.ino` writes
`PROFILE_VELOCITY = 0` to each motor in `setup()`. Zero means "no profile" —
the servo goes to each goal as fast as it can, which is what you want when the
host is streaming a smooth trajectory at 33 Hz and the profile would otherwise
fight it. Raise it only for slow, discrete moves.

---

## 10. Calibration

> **Do this with `scripts/calibrate.py` if you have a U2D2.** The guided wizard
> walks a whole leg — offset *and* direction sign — and writes the results
> straight into `configs/robot.yaml`:
>
> ```bash
> python scripts/view.py --model quad &     # visual reference for "zero pose"
> python scripts/calibrate.py --leg FR
> python scripts/codegen.py                 # regenerate robot_config.h, then reflash
> ```
>
> The manual procedure below is the Arduino-path equivalent, and is what FL was
> originally calibrated with on 2026-05-14.

Offsets and direction signs are **per joint**, generated into
`robot_config.h` as `OFFSETS_DEG[N_JOINTS]` and `DIRS[N_JOINTS]` from the
`offset_deg` / `direction` fields in `configs/robot.yaml`. They all currently
ship at `180.0` / `+1`. That maps `q = 0 rad` to motor position 180° — the
geometric middle of `OP_POSITION`'s 0–360° range. Around that neutral, each
joint's full radian limit lands inside the motor's reachable range:

| Joint | Range from q=0 | Motor degrees |
|---|---|---|
| Shoulder | ±π/2 (±90°) | 90°–270° |
| Wing | ±0.873 (±50°) | 130°–230° |
| Knee | −2.007 to +π/2 (−115° to +90°) | 65°–270° |

**Physical assembly assumption:** when you mount the leg links to the horns, do it with the motor sitting at 180° (its electrical midpoint). Then "q=0 across the board" produces the sim's zero pose mechanically.

If you remount a motor or the sim's zero diverges from the physical neutral,
recalibrate that joint:

1. Power the bus, leave the motors **torque off** (or just observe present positions before commands move them).
2. By hand, move the leg to the sim's zero pose.
3. Read the boot-time `ID <n> pos deg=...` line printed by `setup()` — that's the motor's current degree value.
4. Put that value in the joint's `offset_deg` in **`configs/robot.yaml`** (not in the sketch, and not in `robot_config.h` — that file is generated).
5. `python scripts/codegen.py`, then reflash.

If the joint moves the *opposite* way from the model, flip its `direction` from
`1` to `-1` in the same YAML line and regenerate. The firmware applies it as
`deg = DIRS[i] * q[i] * 180/π + OFFSETS_DEG[i]`.

---

## 11. ~~Known issue — knee can't reach negative angles yet~~ Resolved

Previously, the knee's `−2.007 rad (≈ −115°)` minimum was unreachable because `OP_POSITION` is single-turn 0–360° and negative degrees clamp to 0. The 180° offset in §10 fixes this: `−115°` from neutral lands at motor `65°`, well inside the range. No `OP_EXTENDED_POSITION` switch needed.

---

## 12. Safety

- **Do not exceed 12.0 V** on the Dynamixel bus.
- **Do not hot-plug Dynamixel cables** while shield power is on.
- **Polarity** on the shield's power terminals must match the silkscreen.
- Restrain the leg before the first command — motors snap to goal positions immediately.
