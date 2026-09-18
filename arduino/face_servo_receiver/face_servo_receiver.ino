/*
 * Step 5 receiver for calibrated 16-channel servo-angle packets.
 *
 * Accepted wire format at 115200 baud:
 *   ANGLES,<CH0 degrees>,<CH1 degrees>,...,<CH15 degrees>\n
 *
 * The default build intentionally does NOT initialize or command a PCA9685.
 * A manually enabled CH0-only calibration mode is available below, but it is
 * compiled out until ENABLE_CH0_SERVO_TEST is changed to 1 and re-uploaded.
 * ``ANGLES`` packets always only validate and store values; they never drive
 * physical outputs in this Step 5 sketch.
 */

#ifndef ENABLE_CH0_SERVO_TEST
#define ENABLE_CH0_SERVO_TEST 1
#endif

#if ENABLE_CH0_SERVO_TEST
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

const unsigned long BAUD_RATE = 115200;
const size_t SERVO_COUNT = 16;
const size_t MAX_PACKET_LENGTH = 256;

#if ENABLE_CH0_SERVO_TEST
// This narrow initial range is for manual, one-step-at-a-time observation.
// It is not a measured calibration and must not be widened without checking
// the mechanically safe range of the one servo connected to CH0.
const uint8_t CH0_TEST_CHANNEL = 0;
const uint16_t CH0_TEST_MIN_PULSE = 295;
const uint16_t CH0_TEST_MAX_PULSE = 320;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
bool ch0TestEnabled = false;
#endif

// Python's fixed packet position -> PCA9685 channel mapping.
const uint8_t PCA9685_CHANNELS[SERVO_COUNT] = {
  0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15
};

// This named order matches FaceExpression and RealHead.SERVO_CHANNELS.
const char *const ACTUATOR_NAMES[SERVO_COUNT] = {
  "eye_left_lr", "eye_left_ud", "eye_right_lr", "eye_right_ud",
  "eyelid_left_upper", "eyelid_left_lower",
  "eyelid_right_upper", "eyelid_right_lower",
  "eyebrow_left", "eyebrow_right", "cheek_left", "cheek_right",
  "forehead_left", "forehead_right", "upper_lip", "jaw"
};

float lastAcceptedAngles[SERVO_COUNT];
char packetBuffer[MAX_PACKET_LENGTH];
size_t packetLength = 0;
bool packetOverflow = false;

bool parseAnglesPacket(char *packet, float *candidateAngles) {
  char *savePointer = NULL;
  char *token = strtok_r(packet, ",", &savePointer);
  if (token == NULL || strcmp(token, "ANGLES") != 0) {
    return false;
  }

  for (size_t position = 0; position < SERVO_COUNT; ++position) {
    token = strtok_r(NULL, ",", &savePointer);
    if (token == NULL) {
      return false;
    }

    char *endPointer = NULL;
    double angle = strtod(token, &endPointer);
    // 0~180 is transport sanity validation only, not per-servo safety policy.
    if (endPointer == token || *endPointer != '\0' || !isfinite(angle) ||
        angle < 0.0f || angle > 180.0f) {
      return false;
    }
    candidateAngles[position] = static_cast<float>(angle);
  }

  return strtok_r(NULL, ",", &savePointer) == NULL;
}

#if ENABLE_CH0_SERVO_TEST
void releaseCh0Output() {
  // A zero-width pulse releases CH0; no other PCA9685 channel is touched.
  pwm.setPWM(CH0_TEST_CHANNEL, 0, 0);
}

bool handleCh0TestCommand(char *packet) {
  if (strcmp(packet, "CH0_TEST_ENABLE") == 0) {
    ch0TestEnabled = true;
    Serial.println("OK CH0 test enabled; send CH0_SET,<295..320>");
    return true;
  }

  if (strcmp(packet, "CH0_TEST_STOP") == 0) {
    releaseCh0Output();
    ch0TestEnabled = false;
    Serial.println("OK CH0 output released; test disabled");
    return true;
  }

  if (strncmp(packet, "CH0_", 4) != 0) {
    return false;
  }

  char *savePointer = NULL;
  char *command = strtok_r(packet, ",", &savePointer);
  if (command == NULL || strcmp(command, "CH0_SET") != 0) {
    Serial.println("ERR malformed CH0 test command");
    return true;
  }

  char *pulseText = strtok_r(NULL, ",", &savePointer);
  if (pulseText == NULL || strtok_r(NULL, ",", &savePointer) != NULL ||
      pulseText[0] == '-') {
    Serial.println("ERR malformed CH0_SET pulse");
    return true;
  }

  char *endPointer = NULL;
  unsigned long pulse = strtoul(pulseText, &endPointer, 10);
  if (endPointer == pulseText || *endPointer != '\0' ||
      pulse < CH0_TEST_MIN_PULSE || pulse > CH0_TEST_MAX_PULSE) {
    Serial.println("ERR CH0 pulse outside conservative 295..320 range");
    return true;
  }

  if (!ch0TestEnabled) {
    Serial.println("ERR send CH0_TEST_ENABLE before CH0_SET");
    return true;
  }

  pwm.setPWM(CH0_TEST_CHANNEL, 0, static_cast<uint16_t>(pulse));
  Serial.print("OK CH0 pulse set to ");
  Serial.println(pulse);
  return true;
}
#endif

void handlePacket() {
#if ENABLE_CH0_SERVO_TEST
  if (handleCh0TestCommand(packetBuffer)) {
    return;
  }
#endif

  float candidateAngles[SERVO_COUNT];
  if (!parseAnglesPacket(packetBuffer, candidateAngles)) {
    Serial.println("ERR malformed ANGLES packet");
    return;  // Do not change retained values or command any motor.
  }

  for (size_t position = 0; position < SERVO_COUNT; ++position) {
    uint8_t channel = PCA9685_CHANNELS[position];
    lastAcceptedAngles[channel] = candidateAngles[position];
  }

  // ANGLES packets never call pwm.setPWM(...), even in CH0 test mode.
  Serial.println("OK ANGLES accepted; physical output remains disabled");
}

void setup() {
  Serial.begin(BAUD_RATE);

#if ENABLE_CH0_SERVO_TEST
  pwm.begin();
  pwm.setPWMFreq(50);
  releaseCh0Output();  // Ensure CH0 is released at boot, before any command.
  Serial.println("CH0 manual test compiled; output disabled until CH0_SET");
#endif

  Serial.println("Step 5 receiver READY; PCA9685 output disabled");
}

void loop() {
  while (Serial.available() > 0) {
    char next = static_cast<char>(Serial.read());

    if (next == '\n') {
      if (packetOverflow) {
        Serial.println("ERR packet too long");
      } else {
        packetBuffer[packetLength] = '\0';
        handlePacket();
      }
      packetLength = 0;
      packetOverflow = false;
    } else if (next != '\r') {
      if (packetLength >= MAX_PACKET_LENGTH - 1) {
        packetOverflow = true;
      } else if (!packetOverflow) {
        packetBuffer[packetLength++] = next;
      }
    }
  }
}
