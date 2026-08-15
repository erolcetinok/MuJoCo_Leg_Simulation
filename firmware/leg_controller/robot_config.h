// AUTO-GENERATED from configs/robot.yaml by scripts/codegen.py.
// Do not edit by hand — re-run codegen after editing the YAML.
#ifndef ROBOT_CONFIG_H
#define ROBOT_CONFIG_H

#define HOST_BAUD 57600
#define DXL_BAUD 115200

#define N_JOINTS 12
#define N_LEGS 4

// Joint order is the canonical YAML order: legs FL, FR, BL, BR x joints
// shoulder, wing, knee. The host wire format and the syncWrite packet both
// follow this index order.
static const uint8_t IDS[N_JOINTS] = {
  1,  // [0] shoulder_FL
  2,  // [1] wing_FL
  3,  // [2] knee_FL
  4,  // [3] shoulder_FR
  5,  // [4] wing_FR
  6,  // [5] knee_FR
  7,  // [6] shoulder_BL
  8,  // [7] wing_BL
  9,  // [8] knee_BL
  10,  // [9] shoulder_BR
  11,  // [10] wing_BR
  12  // [11] knee_BR
};

static const float OFFSETS_DEG[N_JOINTS] = {
  180.000000f,  // [0] shoulder_FL
  180.000000f,  // [1] wing_FL
  180.000000f,  // [2] knee_FL
  180.000000f,  // [3] shoulder_FR
  180.000000f,  // [4] wing_FR
  180.000000f,  // [5] knee_FR
  180.000000f,  // [6] shoulder_BL
  180.000000f,  // [7] wing_BL
  180.000000f,  // [8] knee_BL
  180.000000f,  // [9] shoulder_BR
  180.000000f,  // [10] wing_BR
  180.000000f  // [11] knee_BR
};

static const float DIRS[N_JOINTS] = {
  1.0f,  // [0] shoulder_FL
  1.0f,  // [1] wing_FL
  1.0f,  // [2] knee_FL
  1.0f,  // [3] shoulder_FR
  1.0f,  // [4] wing_FR
  1.0f,  // [5] knee_FR
  1.0f,  // [6] shoulder_BL
  1.0f,  // [7] wing_BL
  1.0f,  // [8] knee_BL
  1.0f,  // [9] shoulder_BR
  1.0f,  // [10] wing_BR
  1.0f  // [11] knee_BR
};

static const float LIMITS[N_JOINTS][2] = {
  { -1.57079632679f, 1.57079632679f },  // [0] shoulder_FL
  { -0.87266462600f, 0.87266462600f },  // [1] wing_FL
  { -2.00712863979f, 1.57079632679f },  // [2] knee_FL
  { -1.57079632679f, 1.57079632679f },  // [3] shoulder_FR
  { -0.87266462600f, 0.87266462600f },  // [4] wing_FR
  { -2.00712863979f, 1.57079632679f },  // [5] knee_FR
  { -1.57079632679f, 1.57079632679f },  // [6] shoulder_BL
  { -0.87266462600f, 0.87266462600f },  // [7] wing_BL
  { -2.00712863979f, 1.57079632679f },  // [8] knee_BL
  { -1.57079632679f, 1.57079632679f },  // [9] shoulder_BR
  { -0.87266462600f, 0.87266462600f },  // [10] wing_BR
  { -2.00712863979f, 1.57079632679f }  // [11] knee_BR
};

#endif  // ROBOT_CONFIG_H
