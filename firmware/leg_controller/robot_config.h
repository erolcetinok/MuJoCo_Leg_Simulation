// AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.
// Do not edit by hand — re-run codegen after editing the YAML.
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#define HOST_BAUD 57600
#define DXL_BAUD 115200

#define ID_SHOULDER 1
#define ID_WING 2
#define ID_KNEE 3

static const float OFFSET_SHOULDER_DEG = 180.000000f;
static const float OFFSET_WING_DEG = 180.000000f;
static const float OFFSET_KNEE_DEG = 180.000000f;

static const float LIMIT_SHOULDER[2] = { -1.57079632679f, 1.57079632679f };
static const float LIMIT_WING[2] = { -0.87266462600f, 0.87266462600f };
static const float LIMIT_KNEE[2] = { -2.00712863979f, 1.57079632679f };
static const float DIR_SHOULDER = 1.0f;
static const float DIR_WING = 1.0f;
static const float DIR_KNEE = 1.0f;

#endif  // ROBOT_CONFIG_H
