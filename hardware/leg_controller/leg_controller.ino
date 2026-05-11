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
//                                                                     115200.
//
// UART switch on the shield:
//   - Upload position → flashing the sketch.
//   - DYNAMIXEL position → runtime (shield uses the hardware UART for the
//     DXL bus).

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#define DXL_BAUD 115200
#define HOST_BAUD 115200
#define MAX_LINE 64
#define DXL_PROTOCOL_VERSION 2.0f
#define DXL_DIR_PIN 2

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX

Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);
using namespace ControlTableItem;

#define ID_SHOULDER 1
#define ID_WING     2
#define ID_KNEE     3

const float LIMIT_SHOULDER[2] = { -1.57079632679f, 1.57079632679f };
const float LIMIT_WING[2]     = { -0.872664625997f, 0.872664625997f };
const float LIMIT_KNEE[2]     = { -2.00712863979f, 1.57079632679f };

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
    line[len++] = c;
  }
  if (len >= MAX_LINE - 1) {
    len = 0;
  }
}

void processLine(char* line) {
  float q1, q2, q3;
  if (sscanf(line, "q,%f,%f,%f", &q1, &q2, &q3) != 3) {
    cmd.println("? bad format; use q,q1,q2,q3");
    return;
  }

  q1 = constrain(q1, LIMIT_SHOULDER[0], LIMIT_SHOULDER[1]);
  q2 = constrain(q2, LIMIT_WING[0],     LIMIT_WING[1]);
  q3 = constrain(q3, LIMIT_KNEE[0],     LIMIT_KNEE[1]);

  float d1 = q1 * 180.0f / PI;
  float d2 = q2 * 180.0f / PI;
  float d3 = q3 * 180.0f / PI;

  dxl.setGoalPosition(ID_SHOULDER, d1, UNIT_DEGREE);
  dxl.setGoalPosition(ID_WING,     d2, UNIT_DEGREE);
  dxl.setGoalPosition(ID_KNEE,    d3, UNIT_DEGREE);

  cmd.print("ok ");
  cmd.print(q1);
  cmd.print(" ");
  cmd.print(q2);
  cmd.print(" ");
  cmd.println(q3);
}
