// Sweep every XL-series baud + low IDs to find a single motor's current settings.
// Run with ONE motor connected — multiple motors at the same ID will collide
// on the bus and the scan will see nothing.
//
// Upload (UART switch on Upload), flip switch to DYNAMIXEL, press RESET, open
// a terminal on the USB-to-TTL adapter's port at 115200 baud. Scan takes
// ~5 seconds. Results print at the end.

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#define DXL_DIR_PIN 2
#define DXL_PROTOCOL_VERSION 2.0f
#define HOST_BAUD 57600

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX
Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);

const uint32_t BAUDS[] = {9600, 57600, 115200, 1000000};
const int NUM_BAUDS = sizeof(BAUDS) / sizeof(BAUDS[0]);
const uint8_t MAX_ID = 10;  // factory default is 1; most reconfigs stay low

struct Finding {
  uint32_t baud;
  uint8_t  id;
  uint16_t model;
};
Finding findings[16];
int num_findings = 0;

void setup() {
  cmd.begin(HOST_BAUD);

  for (int b = 0; b < NUM_BAUDS; b++) {
    dxl.begin(BAUDS[b]);
    dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
    for (uint8_t id = 0; id <= MAX_ID; id++) {
      if (dxl.ping(id)) {
        if (num_findings < 16) {
          findings[num_findings].baud  = BAUDS[b];
          findings[num_findings].id    = id;
          findings[num_findings].model = dxl.getModelNumber(id);
          num_findings++;
        }
      }
    }
  }

  delay(200);

  cmd.print("=== scan complete: ");
  cmd.print(num_findings);
  cmd.println(" motor(s) found ===");
  for (int i = 0; i < num_findings; i++) {
    cmd.print("  ID ");
    cmd.print(findings[i].id);
    cmd.print("  baud ");
    cmd.print(findings[i].baud);
    cmd.print("  model ");
    cmd.println(findings[i].model);
  }
  if (num_findings == 0) {
    cmd.println("Motor is either unpowered, on wrong jack (must be 3-pin TTL),");
    cmd.println("at an ID > 10, or at a baud not in the scan list.");
  }
}

void loop() {}
