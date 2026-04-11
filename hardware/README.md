# Single-leg hardware controller

Send three joint angles (radians) from your Mac to an Arduino that drives three daisy-chained Dynamixels. Order: **shoulder, wing, knee** (same as the MuJoCo single-leg model).

## Hardware

- **Arduino:** **Arduino UNO Q** (recommended) or **Mega 2560**. Both use two UARTs: `Serial` = USB to your Mac for angle commands, `Serial1` = Dynamixel shield on **D0/D1**. The UNO Q MCU runs at **3.3 V** logic on the headers; confirm your servo model and shield are compatible (many Dynamixels use 5 V TTL—level shifting or a 3.3 V–tolerant bus may be required).
- **Shield:** Dynamixel Shield 2.0 (or compatible) for Arduino.
- **Servos:** Three Dynamixels, daisy-chained. Set IDs to **1** (shoulder), **2** (wing), **3** (knee). Protocol 2.0, baud **1000000** (1 Mbps) in the sketch.
- **Leg:** Same 3-DoF layout as in `meshes/single_leg.xml`.

## Arduino (leg_controller)

1. Install **Dynamixel2Arduino** in Arduino IDE: **Sketch → Include Library → Manage Libraries** → search “Dynamixel2Arduino” → Install.
2. Open `hardware/leg_controller/leg_controller.ino`.
3. **Board:** **Arduino UNO Q** (install the **Arduino UNO Q** / Zephyr board package in the Boards Manager if prompted), or **Arduino Mega 2560** for the same UART layout.
4. **Port:** Select the USB port (e.g. `/dev/cu.usbmodem…` on Mac).
5. Upload. Open **Serial Monitor** at **115200**; you should see “the controller is ready” and “dynamixels ready” if all three IDs are found.

**Other boards:** Classic **Arduino Uno (AVR)** shares one hardware UART between USB and the shield pins, so you cannot run this sketch’s Python-over-USB workflow and the Dynamixel bus at the same time without extra wiring or code changes. Prefer UNO Q or Mega.

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
