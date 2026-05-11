# Single-leg hardware setup

Start-to-finish bring-up for the 3-DoF leg: three daisy-chained Dynamixels driven by an Arduino UNO R3, commanded over USB from your Mac. Joint order everywhere is **shoulder → wing → knee**, same as `meshes/single_leg.xml`.

The Arduino's onboard USB is **flash-only** for this project. Host comms (commands and replies) go through a small USB-to-TTL adapter wired to the Arduino's D7/D8 pins. The reason: on UNO R3, the Dynamixel Shield and the onboard USB-UART chip both live on the same hardware UART (D0/D1) and electrically fight each other for D0 — you cannot reliably read host bytes through the Arduino's USB while the shield is driving the DXL bus. See §5 for the full explanation. The adapter sidesteps the problem cleanly by putting host comms on a second, dedicated serial path that ROBOTIS designed the shield for.

---

## 1. Parts

| Part | Notes |
|------|-------|
| **Arduino UNO R3** | ATmega328P. Onboard USB used for flashing only. |
| **ROBOTIS DYNAMIXEL Shield** | TTL half-duplex driver + screw terminals + UART switch + Dynamixel power switch. |
| **3 × XL430-W250-T** servos | Daisy-chained. IDs **1 / 2 / 3** (shoulder / wing / knee), **Protocol 2.0**, **baud 115 200**. |
| **3-pin TTL JST cables** | XL430 uses TTL only — **never** the 4-pin RS-485 port on the shield. |
| **USB-to-TTL adapter** | CP2102, CH340, or FTDI. Any cheap module from Amazon. Carries host commands and replies. |
| **3× M/F jumper wires** | "Dupont" cables to connect adapter ↔ Arduino. |
| **Bench supply** | **6.5–12.0 V** (XL430's spec — the shield itself accepts 5–24 V, but the motors don't), ~5 A capable for three XL-class motors under load. **Do not exceed 12.0 V.** |
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
4. From the repo root (the directory containing `requirements.txt`, one level above `hardware/`): `pip install -r requirements.txt` — needed for the helper scripts in `scripts/`.
5. macOS bundles drivers for CP2102 and FTDI out of the box (macOS 11+). For CH340 modules, install [WCH's signed driver](https://www.wch-ic.com/downloads/CH34XSER_MAC_ZIP.html) if your adapter doesn't enumerate.

---

## 3. Configure the servos (IDs, baud, protocol)

Each motor must end up at **ID 1 / 2 / 3**, **baud 115 200**, **Protocol 2.0**. Factory defaults are **ID 1, baud 57 600** — three fresh XL430s on the same bus will all answer to ID 1 and collide, which is why you configure one motor at a time below.

> Why 115 200 and not 1 Mbps (the XL430's nominal max)? Because the hardware Serial bus only carries DXL traffic now (host is on the adapter), 1 Mbps is technically usable — but motors and sketches currently default to 115 200 and there's no benefit at this scale. Stick with 115 200 unless you have a reason.

**You need the shield powered and wired first** — do section 4, then come back here before uploading `leg_controller.ino`.

Use the repo-local helper sketch `hardware/tools/configure_motor/configure_motor.ino`. It probes the motor across 57 600 / 115 200 / 1 000 000, sets it to your target ID, then changes its baud to 115 200 — all in one upload. Output prints to the adapter's terminal at 115 200.

Procedure — repeat three times:

1. **Disconnect all motors except the one you're configuring.** Plug only motor #1 (or #2 or #3 — they're all identical at this point) into the shield's TTL jack.
2. Open `hardware/tools/configure_motor/configure_motor.ino`.
3. Edit `#define SOURCE_ID` (match the motor's current ID, or 1 for factory-fresh) and `#define TARGET_ID` — **1** for shoulder, **2** for wing, **3** for knee.
4. Set UART switch to **Upload**, click **Upload** in the IDE.
5. Flip UART switch to **DYNAMIXEL**, press **RESET**, wait ~3 seconds.
6. Open a terminal on the adapter's port at **115200** baud (`python scripts/usb_console.py` does this, or any terminal of your choice). You should see `OK: motor at ID <n>, baud 115200`.
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

What this means in practice:
- The shield's **UART switch sits on DYNAMIXEL** during normal operation. Only flip to Upload for flashing.
- **Do not try to talk to the Arduino via its onboard USB monitor at runtime.** It's wired into the same UART the shield uses, so anything you type there gets blocked by the shield's transceiver. The adapter is the only host-comm path.
- `SoftwareSerial` is bit-banged at 115 200, which is at the upper end of its reliability range on a 16 MHz AVR. For sustained high-rate streaming (e.g. closed-loop control at >100 Hz), this would be marginal — but for interactive jogging and one-shot Python commands it's fine.

---

## 6. Upload firmware

1. Open `hardware/leg_controller/leg_controller.ino` in Arduino IDE.
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

Open a terminal on the **adapter's** port at **115200** baud. The easiest way:

```bash
python scripts/usb_console.py --port /dev/cu.usbserial-XXXX
```

(or run `python scripts/usb_console.py --list` to see what's plugged in — pick the adapter's port, not the Arduino's `/dev/cu.usbmodem*`).

Expected output after reset:

```
the controller is ready
dynamixels ready
mode: usb command (q,q1,q2,q3)
```

Then send `q,0.0,0.0,0.0` + Enter (line ending Newline). Reply should be `ok 0.00 0.00 0.00` and the leg snaps to the zero pose.

**If the terminal is blank or shows nothing:** baud rate or wrong port. Confirm 115 200 and that you picked the adapter's port. If still blank, double-check the three jumper wires (TX↔D7, RX↔D8, GND↔GND).

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

Interactive (no per-command lag — port stays open):

```bash
python scripts/jog.py --port /dev/cu.usbserial-XXXX
```

```
> 0 0 0
  ok 0.00 0.00 0.00
> 0.5 -0.3 0.2
  ok 0.50 -0.30 0.20
```

`scripts/usb_console.py` is a terminal replacement for Arduino Serial Monitor — useful for one-off inspection. All three scripts default to baud **115 200** to match the sketch.

---

## 9. Protocol

- One line per command: `q,<q1>,<q2>,<q3>`, radians, newline-terminated.
- Reply: `ok <q1> <q2> <q3>` (echo of clamped values), or `? bad format` / `? no servo ID <n>`.
- Joint limits (radians, enforced by the Arduino):

  | Joint | Min | Max |
  |-------|-----|-----|
  | Shoulder | −π/2 | +π/2 |
  | Wing | −0.873 (≈ −50°) | +0.873 (≈ +50°) |
  | Knee | −2.007 (≈ −115°) | +π/2 |

Limits match `meshes/single_leg.xml`.

Motion speed is firmware-set, not commanded: `leg_controller.ino` writes `PROFILE_VELOCITY = 40` to each motor in `setup()`. Edit that constant in the sketch if you want faster/slower trajectories — the Python side has no speed parameter.

---

## 10. Calibration

If the leg's mechanical zero doesn't match the sim's zero, apply a constant per-joint offset. Procedure:

1. Move the physical leg into the **sim's zero pose** by hand (motors torque-off, or send the angle that physically produces that pose).
2. Read each motor's present position by adding a debug print to the sketch (`dxl.readControlTableItem(PRESENT_POSITION, id)` returns ticks; convert with `* 360.0 / 4096.0` for degrees).
3. The difference between the read position and 0 rad **is** that joint's offset.
4. Apply the offset in one place — easiest in `leg_controller.ino` right before `setGoalPosition`, so the sim, IK scripts, and Python all stay aligned without knowing about the offset.

---

## 11. Known issue — knee can't reach negative angles yet

`leg_controller.ino` uses `OP_POSITION` (single-turn, 0–360°). The knee's limit goes to **−2.007 rad (≈ −115°)**, which `OP_POSITION` cannot represent — negative degrees get clamped to 0 by the library. Until this is fixed, the knee's full negative range is unreachable.

Two ways to fix:
- Switch knee (and any joint that needs negative angles) to `OP_EXTENDED_POSITION` mode.
- Or, add a +180° offset to the knee in the sketch so the sim's range maps into 65°–270° on the motor — works only if mechanically the wiring/horn allows it.

Pick one before relying on the full sim range on hardware.

---

## 12. Safety

- **Do not exceed 12.0 V** on the Dynamixel bus.
- **Do not hot-plug Dynamixel cables** while shield power is on.
- **Polarity** on the shield's power terminals must match the silkscreen.
- Restrain the leg before the first command — motors snap to goal positions immediately.
