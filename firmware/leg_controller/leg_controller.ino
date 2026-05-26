// Reads joint angles over the USB-to-TTL adapter and drives N_JOINTS dynamixels.
// Protocol: q,<q0>,<q1>,...,<q[N_JOINTS-1]>  (radians, joint order = YAML order
// = FL,FR,BL,BR × shoulder,wing,knee; see robot_config.h for the index map).
//
// Hardware:
//   - Arduino UNO R3 + ROBOTIS Dynamixel Shield + 12× XL430-W250-T motors
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
//
// All N_JOINTS goal positions are written to the DXL bus in a single
// syncWrite packet per host line. This is the Stanford Pupper convention
// and cuts bus time vs N_JOINTS separate setGoalPosition() calls, which is
// what frees up the SoftwareSerial RX ISR enough to receive host bytes
// reliably (~15% drop rate seen on the single-leg path was caused by per-motor
// writes blocking the bus).

#include <SoftwareSerial.h>
#include <Dynamixel2Arduino.h>

#include "robot_config.h"

#define MAX_LINE 256              // 12 floats × ~16 chars + commas + 'q,' fits
#define DXL_PROTOCOL_VERSION 2.0f
#define DXL_DIR_PIN 2

// XL430 goal-position control table entry: addr 116, 4 bytes (int32_t ticks).
#define GOAL_POSITION_ADDR 116
#define GOAL_POSITION_LEN  4

SoftwareSerial cmd(7, 8);  // RX=D7 ← adapter TX, TX=D8 → adapter RX

Dynamixel2Arduino dxl(Serial, DXL_DIR_PIN);
using namespace ControlTableItem;

// Pre-built syncWrite descriptor. Reused every line — the param struct is
// large (~200 B), so allocate once at file scope rather than per-call.
typedef struct __attribute__((packed)) {
  int32_t goal_position;
} sw_data_t;

static ParamForSyncWriteInst_t sw_info;
static sw_data_t sw_data[N_JOINTS];

void processLine(char* line);

void setup() {
  cmd.begin(HOST_BAUD);
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  cmd.println("the controller is ready");

  for (int i = 0; i < N_JOINTS; i++) {
    uint8_t id = IDS[i];
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
    // profile_velocity = 0 → no trapezoidal cap; motor tracks each
    // streamed goal as fast as it can. Required for smooth streaming
    // control (Stanford Pupper convention).
    dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
  }

  // Build the syncWrite descriptor once. Per-tick we only refresh the
  // goal_position payload, not the ID list.
  sw_info.packet.p_buf = nullptr;
  sw_info.packet.is_completed = false;
  sw_info.addr = GOAL_POSITION_ADDR;
  sw_info.addr_length = GOAL_POSITION_LEN;
  sw_info.id_count = N_JOINTS;
  for (int i = 0; i < N_JOINTS; i++) {
    sw_info.xel[i].id = IDS[i];
    sw_info.xel[i].p_data = (uint8_t*)&sw_data[i].goal_position;
  }

  cmd.print("dynamixels ready (");
  cmd.print(N_JOINTS);
  cmd.println(" joints)");
  cmd.println("mode: usb command (q,q0,...,qN-1)");
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
    cmd.println("? bad format; use q,q0,q1,...");
    return;
  }
  float q[N_JOINTS];
  char* p = line + 2;
  char* end;
  for (int i = 0; i < N_JOINTS; i++) {
    q[i] = strtod(p, &end);
    if (end == p) {
      cmd.print("? bad format at index ");
      cmd.println(i);
      return;
    }
    p = end;
    if (i < N_JOINTS - 1) {
      if (*p != ',') {
        cmd.print("? missing comma at index ");
        cmd.println(i);
        return;
      }
      p++;
    }
  }

  // Clamp, apply sign + offset, convert to ticks, stage into the syncWrite
  // payload. Ticks: 0..4095 over 360°, so deg → ticks = deg * 4096 / 360.
  for (int i = 0; i < N_JOINTS; i++) {
    float qi = constrain(q[i], LIMITS[i][0], LIMITS[i][1]);
    float deg = DIRS[i] * qi * 180.0f / PI + OFFSETS_DEG[i];
    sw_data[i].goal_position = (int32_t)(deg * 4096.0f / 360.0f + 0.5f);
  }

  // One packet for all N_JOINTS — the whole point of the multi-leg refactor.
  bool ok = dxl.syncWrite(&sw_info);

  if (ok) {
    cmd.print("ok");
    for (int i = 0; i < N_JOINTS; i++) {
      cmd.print(' ');
      cmd.print(q[i], 3);
    }
    cmd.println();
  } else {
    cmd.print("? syncWrite failed err=");
    cmd.println(dxl.getLastLibErrCode());
  }
}
