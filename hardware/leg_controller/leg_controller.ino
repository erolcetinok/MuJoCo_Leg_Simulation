// reads the joint angles over usb serial and drives the 3 dynamixels
// protocol for using the program q,<q1>,<q2>,<q3>:
// angles in radians. order shoulder (0), wing (1), knee (2)
// install Dynamixel2Arduino library. mega: Serial1 = shield, Serial = usb (mac)

#define SERIAL_BAUD 115200
#define MAX_LINE 64

#if defined(ARDUINO_AVR_MEGA2560)
  #define DXL_SERIAL Serial1
  #define CMD_SERIAL Serial
  #define DXL_DIR_PIN 2
#else
  #define DXL_SERIAL Serial
  #define CMD_SERIAL Serial
  #define DXL_DIR_PIN 2
#endif

#define DXL_BAUD 1000000
#define DXL_PROTOCOL_VERSION 2.0f

#include <Dynamixel2Arduino.h>
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

#define ID_SHOULDER 1
#define ID_WING     2
#define ID_KNEE     3

// joint order: shoulder, wing, knee
// the limits in radians are the same as in the mujoco model
const float LIMIT_SHOULDER[2] = { -1.57079632679f, 1.57079632679f };
const float LIMIT_WING[2]     = { -0.872664625997f, 0.872664625997f };
const float LIMIT_KNEE[2]     = { -2.00712863979f, 1.57079632679f };

void processLine(char* line);

void setup() {
  CMD_SERIAL.begin(SERIAL_BAUD);
#if defined(ARDUINO_AVR_MEGA2560)
  while (!CMD_SERIAL) {;}
#endif
  CMD_SERIAL.println("the controller is ready");

  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  for (int id = ID_SHOULDER; id <= ID_KNEE; id++) {
    if (!dxl.ping(id)) {
      CMD_SERIAL.print("? no servo ID ");
      CMD_SERIAL.println(id);
      continue;
    }
    dxl.torqueOff(id);
    dxl.setOperatingMode(id, OP_POSITION);
    dxl.torqueOn(id);
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 40);
  }
  CMD_SERIAL.println("dynamixels ready");
  CMD_SERIAL.println("mode: usb command (q,q1,q2,q3)");
}

void loop() {
  static char line[MAX_LINE];
  static int len = 0;

  // read one line from the serial port
  while (CMD_SERIAL.available() && len < MAX_LINE - 1) {
    char c = CMD_SERIAL.read();
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

// parse the three angles from the line then write to dynamixels in degrees
void processLine(char* line) {
  float q1, q2, q3;
  if (sscanf(line, "q,%f,%f,%f", &q1, &q2, &q3) != 3) {
    CMD_SERIAL.println("? bad format; use q,q1,q2,q3");
    return;
  }

  // clamp sim limits
  q1 = constrain(q1, LIMIT_SHOULDER[0], LIMIT_SHOULDER[1]);
  q2 = constrain(q2, LIMIT_WING[0],     LIMIT_WING[1]);
  q3 = constrain(q3, LIMIT_KNEE[0],     LIMIT_KNEE[1]);

  float d1 = q1 * 180.0f / PI;
  float d2 = q2 * 180.0f / PI;
  float d3 = q3 * 180.0f / PI;

  dxl.setGoalPosition(ID_SHOULDER, d1, UNIT_DEGREE);
  dxl.setGoalPosition(ID_WING,     d2, UNIT_DEGREE);
  dxl.setGoalPosition(ID_KNEE,    d3, UNIT_DEGREE);

  // echo back to see what the arduino received
  CMD_SERIAL.print("ok ");
  CMD_SERIAL.print(q1);
  CMD_SERIAL.print(" ");
  CMD_SERIAL.print(q2);
  CMD_SERIAL.print(" ");
  CMD_SERIAL.println(q3);
}
