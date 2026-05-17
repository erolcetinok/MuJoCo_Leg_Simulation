// Reads joint angles over the USB-to-TTL adapter and drives the 3 dynamixels.
// Protocol: q,<q1>,<q2>,<q3>  (radians, order: shoulder, wing, knee)
//
// Hardware:
//   - Arduino UNO R3 + ROBOTIS Dynamixel Shield + 3× XL430-W250-T motors
//   - CP2102 / CH340 / FTDI USB-to-TTL adapter wired:
//       adapter TX → Arduino D7   (Arduino's SoftwareSerial RX)
//       adapter RX → Arduino D8   (Arduino's SoftwareSerial TX)
//       adapter GND → Arduino GND
//   - Adapter's VCC stays disconnected (Arduino is powered via its own USB).
//
// Two USB connections at runtime:
//   - Arduino USB-B (/dev/cu.usbmodem*)  → flash only; do not use for comms.
//   - Adapter USB    (/dev/cu.usbserial* or /dev/cu.SLAB_USBtoUART)  → host
//                                                                     I/O at
//                                                                     HOST_BAUD
//                                                                     (see
//                                                                     robot_config.h).
//
// UART switch on the shield:
//   - Upload position → flashing the sketch.
//   - DYNAMIXEL position → runtime (shield uses the hardware UART for the
//     DXL bus).
//
// Joint limits, motor IDs, offsets, and bauds all come from robot_config.h,
// generated from ../../configs/robot.yaml by scripts/codegen.py — do not
// hand-edit the .h.

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#include "robot_config.h"

#define MAX_LINE 64
#define DXL_PROTOCOL_VERSION 2.0f
#define DXL_DIR_PIN 2

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX

Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);
using namespace ControlTableItem;

void processLine(char* line);

void setup() {
  cmd.begin(HOST_BAUD);
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  cmd.println("the controller is ready");

  for (int id = ID_SHOULDER; id <= ID_KNEE; id++) {
    if (!dxl.ping(id)) {
      cmd.print("? no servo ID ");
      cmd.println(id);
      continue;
    }
    dxl.reboot(id);
    delay(300);
    int32_t ticks = dxl.getPresentPosition(id);
    cmd.print("ID ");
    cmd.print(id);
    cmd.print(" pos ticks=");
    cmd.print(ticks);
    cmd.print(" deg=");
    cmd.println(ticks * 360.0f / 4096.0f, 2);
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_POSITION);
    dxl.torqueOn(id);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 40);
  }
  cmd.println("dynamixels ready");
  cmd.println("mode: usb command (q,q1,q2,q3)");
}

void loop() {
  static char line[MAX_LINE];
  static int len = 0;

  while (cmd.available() && len < MAX_LINE - 1) {
    char c = cmd.read();
    if (c == '\n' || c == '\r') {
      line[len] = '\0';
      if (len > 0) {
        processLine(line);
      }
      len = 0;
      break;
    }
    // Discard SoftwareSerial noise before the 'q,' command prefix.
    if (len == 0 && c != 'q') continue;
    line[len++] = c;
  }
  if (len >= MAX_LINE - 1) {
    len = 0;
  }
}

void processLine(char* line) {
  // Arduino AVR's default sscanf has no %f support, so parse with strtod.
  if (line[0] != 'q' || line[1] != ',') {
    cmd.println("? bad format; use q,q1,q2,q3");
    return;
  }
  char* p = line + 2;
  char* end;
  float q1 = strtod(p, &end);
  if (end == p || *end != ',') {
    cmd.println("? bad format; use q,q1,q2,q3");
    return;
  }
  p = end + 1;
  float q2 = strtod(p, &end);
  if (end == p || *end != ',') {
    cmd.println("? bad format; use q,q1,q2,q3");
    return;
  }
  p = end + 1;
  float q3 = strtod(p, &end);
  if (end == p) {
    cmd.println("? bad format; use q,q1,q2,q3");
    return;
  }

  q1 = constrain(q1, LIMIT_SHOULDER[0], LIMIT_SHOULDER[1]);
  q2 = constrain(q2, LIMIT_WING[0], LIMIT_WING[1]);
  q3 = constrain(q3, LIMIT_KNEE[0], LIMIT_KNEE[1]);

  float d1 = q1 * 180.0f / PI + OFFSET_SHOULDER_DEG;
  float d2 = q2 * 180.0f / PI + OFFSET_WING_DEG;
  float d3 = q3 * 180.0f / PI + OFFSET_KNEE_DEG;

  dxl.setGoalPosition(ID_SHOULDER, d1, UNIT_DEGREE);
  dxl.setGoalPosition(ID_WING, d2, UNIT_DEGREE);
  dxl.setGoalPosition(ID_KNEE, d3, UNIT_DEGREE);

  cmd.print("ok ");
  cmd.print(q1);
  cmd.print(" ");
  cmd.print(q2);
  cmd.print(" ");
  cmd.println(q3);
}
