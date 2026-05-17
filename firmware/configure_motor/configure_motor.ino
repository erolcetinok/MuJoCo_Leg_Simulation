// One-shot motor configuration for UNO R3 + ROBOTIS Dynamixel Shield.
// Takes a motor currently at SOURCE_ID (auto-detects current baud across
// 57600 / 115200 / 1000000) and sets it to TARGET_ID at TARGET_BAUD.
//
// Connect ONE XL430 at a time. Edit SOURCE_ID and TARGET_ID below, upload,
// flip UART switch to DYNAMIXEL, press RESET, wait a few seconds. Open a
// terminal on the USB-to-TTL adapter's port (/dev/cu.usbserial* or
// /dev/cu.SLAB_USBtoUART) at 115200 baud to see the final status line.
//
// If you don't know the motor's current ID/baud, run
// hardware/tools/find_motor/find_motor.ino first.

#define SOURCE_ID 3   // <<< what the motor is at right now (per find_motor) >>>
#define TARGET_ID 1   // <<< what you want the motor to become >>>

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#define DXL_DIR_PIN 2
#define DXL_PROTOCOL_VERSION 2.0f
#define TARGET_BAUD  115200
#define HOST_BAUD    57600

// Bauds to try when probing for the motor's current state. Covers factory-
// fresh (57600), the post-migration value (115200), and the old project
// default (1000000) so motors mid-migration are still detectable.
const uint32_t TRIAL_BAUDS[] = {57600, 115200, 1000000};
const int NUM_TRIAL_BAUDS = sizeof(TRIAL_BAUDS) / sizeof(TRIAL_BAUDS[0]);

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX
Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);

void setup() {
  cmd.begin(HOST_BAUD);
  uint32_t source_baud = 0;

  for (int i = 0; i < NUM_TRIAL_BAUDS; i++) {
    dxl.begin(TRIAL_BAUDS[i]);
    dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
    delay(200);
    if (dxl.ping(SOURCE_ID)) {
      source_baud = TRIAL_BAUDS[i];
      break;
    }
  }

  if (source_baud == 0) {
    dxl.begin(TARGET_BAUD);
    cmd.print("FAIL: no motor at ID ");
    cmd.print(SOURCE_ID);
    cmd.println(" (tried baud 57600, 115200, 1000000)");
    cmd.println("Run find_motor.ino to discover the motor's current settings.");
    return;
  }

  // Change ID if needed.
  if (TARGET_ID != SOURCE_ID) {
    dxl.torqueOff(SOURCE_ID);
    if (!dxl.setID(SOURCE_ID, TARGET_ID)) {
      dxl.begin(TARGET_BAUD);
      cmd.println("FAIL: setID");
      return;
    }
  }

  // Change baud if needed.
  if (source_baud != TARGET_BAUD) {
    dxl.torqueOff(TARGET_ID);
    if (!dxl.setBaudrate(TARGET_ID, TARGET_BAUD)) {
      cmd.println("FAIL: setBaudrate (motor at new ID, still at old baud)");
      return;
    }
  }

  // Verify at final settings.
  dxl.begin(TARGET_BAUD);
  delay(200);
  if (dxl.ping(TARGET_ID)) {
    cmd.print("OK: motor at ID ");
    cmd.print(TARGET_ID);
    cmd.print(", baud ");
    cmd.println(TARGET_BAUD);
  } else {
    cmd.println("WARN: configured but final ping failed — power-cycle and rescan");
  }
}

void loop() {}
