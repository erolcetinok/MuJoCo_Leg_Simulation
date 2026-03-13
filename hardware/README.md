# Single-leg hardware controller

Send three joint angles (radians) from your Mac to an Arduino that drives three daisy-chained Dynamixels. Order: **shoulder, wing, knee** (same as the MuJoCo single-leg model).

## Hardware

- **Arduino:** Uno or **Mega (recommended)**. Mega has two serial ports: Serial = USB (Mac), Serial1 = Dynamixel shield.
- **Shield:** Dynamixel Shield 2.0 (or compatible) for Arduino.
- **Servos:** Three Dynamixels, daisy-chained. Set IDs to **1** (shoulder), **2** (wing), **3** (knee). Protocol 2.0, baud **1000000** (1 Mbps) in the sketch.
- **Leg:** Same 3-DoF layout as in `meshes/single_leg.xml`.

## Arduino (leg_controller)

1. Install **Dynamixel2Arduino** in Arduino IDE: **Sketch → Include Library → Manage Libraries** → search “Dynamixel2Arduino” → Install.
2. Open `hardware/leg_controller/leg_controller.ino`.
3. **Board:** Arduino Mega 2560 (or Uno; see note below).
4. **Port:** Select the USB port (e.g. `/dev/cu.usbmodem…` on Mac).
5. Upload. Open **Serial Monitor** at **115200**; you should see “leg_controller ready…” and “Dynamixels ready.” if all three IDs are found.

**Uno note:** On Uno the shield uses the same Serial as USB, so you can’t easily have Mac commands and Dynamixel on the same board. Use Mega, or use Uno only with the Serial Monitor (no Python script) to test the leg.

## Finding the port on Mac

- Arduino IDE: **Tools → Port** lists the device (e.g. `/dev/cu.usbmodem14101`).
- Terminal: `ls /dev/cu.usb*` when the board is plugged in.

## Python (send angles from Mac)

1. Install dependency: `pip install pyserial`
2. Run:
   ```bash
   python scripts/send_angles.py 0.0 0.0 0.0 --port /dev/cu.usbmodem14101
   ```
   Replace the port with your actual port. Angles are in **radians** (shoulder, wing, knee).

You can set the port once and use it for the session:
```bash
export SERIAL_PORT=/dev/cu.usbmodem14101
python scripts/send_angles.py 0.1 -0.2 0.5
```

## Protocol

- One line per command: `q,<q1>,<q2>,<q3>` with angles in radians.
- Arduino replies with a line like `ok 0.1 -0.2 0.5` (echo of clamped angles).
- Joint limits match `meshes/single_leg.xml` and are enforced on the Arduino.

## Calibration

If “zero” on the physical leg doesn’t match the sim, add an offset per joint (in the Arduino sketch or in Python before sending). Keep offsets in one place so sim and hardware stay aligned.
