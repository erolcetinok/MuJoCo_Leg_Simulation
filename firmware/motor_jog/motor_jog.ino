// Direct-degree motor jogger for UNO R3 + Dynamixel Shield + USB-to-TTL
// adapter on D7/D8. Bypasses clamping / radian conversion / joint limits —
// sends raw degrees straight to the chosen motor. Use this to confirm motors
// physically move and to find each motor's usable range before tuning
// leg_controller.ino's limits.
//
// Setup is identical to leg_controller.ino — see that file's header for the
// adapter wiring and switch positions. Connect a terminal to the adapter's
// port (/dev/cu.usbserial* or /dev/cu.SLAB_USBtoUART) at 115200 baud, line
// ending Newline.
//
// Commands (one per line):
//   m,<id>,<degrees>   absolute position. Examples:
//                        m,1,180   shoulder to 180°
//                        m,2,90    wing to 90°
//                        m,3,270   knee to 270°
//   +<id>              move motor <id> by +STEP_DEG degrees from current goal
//   -<id>              move motor <id> by -STEP_DEG degrees from current goal
//   ?                  print each motor's current goal position
//
// XL430 in OP_POSITION mode accepts 0–360° (single turn). Negative angles
// or > 360° get clamped by the library to 0 / 360. Power off and free-
// spin a motor by hand to find its mechanical safe range before driving
// it under torque.

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#define DXL_BAUD 115200
#define HOST_BAUD 57600
#define DXL_PROTOCOL_VERSION 2.0f
#define DXL_DIR_PIN 2
#define STEP_DEG 15.0f

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX

Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);
using namespace ControlTableItem;

const uint8_t IDS[] = {1, 2, 3};
const int NUM_IDS = sizeof(IDS) / sizeof(IDS[0]);

float goal[4] = {0, 180, 180, 180};  // index by id; id 0 unused. start at center.

void setup() {
  cmd.begin(HOST_BAUD);
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  cmd.println("motor_jog ready");

  for (int i = 0; i < NUM_IDS; i++) {
    uint8_t id = IDS[i];
    if (!dxl.ping(id)) {
      cmd.print("? no servo ID ");
      cmd.println(id);
      continue;
    }
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_POSITION);
    dxl.torqueOn(id);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 40);
    dxl.setGoalPosition(id, goal[id], UNIT_DEGREE);
  }
  cmd.println("commands: m,<id>,<deg>   +<id>   -<id>   ?");
}

void loop() {
  static char line[64];
  static int len = 0;

  while (cmd.available() && len < 63) {
    char c = cmd.read();
    if (c == '\n' || c == '\r') {
      line[len] = '\0';
      if (len > 0) processLine(line);
      len = 0;
      break;
    }
    line[len++] = c;
  }
  if (len >= 63) len = 0;
}

void moveMotor(uint8_t id, float deg) {
  if (id < 1 || id > 3) {
    cmd.println("? id must be 1, 2, or 3");
    return;
  }
  goal[id] = deg;
  dxl.setGoalPosition(id, deg, UNIT_DEGREE);
  cmd.print("ok id ");
  cmd.print(id);
  cmd.print(" -> ");
  cmd.print(deg);
  cmd.println(" deg");
}

void processLine(char* line) {
  int id;
  float deg;
  if (sscanf(line, "m,%d,%f", &id, &deg) == 2) {
    moveMotor((uint8_t)id, deg);
    return;
  }
  if (line[0] == '+' && sscanf(line + 1, "%d", &id) == 1) {
    moveMotor((uint8_t)id, goal[id] + STEP_DEG);
    return;
  }
  if (line[0] == '-' && sscanf(line + 1, "%d", &id) == 1) {
    moveMotor((uint8_t)id, goal[id] - STEP_DEG);
    return;
  }
  if (line[0] == '?') {
    for (int i = 0; i < NUM_IDS; i++) {
      uint8_t id = IDS[i];
      cmd.print("id ");
      cmd.print(id);
      cmd.print(" goal ");
      cmd.print(goal[id]);
      cmd.println(" deg");
    }
    return;
  }
  cmd.println("? bad format; use m,<id>,<deg> or +<id> or -<id> or ?");
}
