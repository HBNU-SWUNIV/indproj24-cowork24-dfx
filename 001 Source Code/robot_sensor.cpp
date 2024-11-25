#include "../../include/Robot/Robot_sensor.h"

// KalmanFilter Class
class KalmanFilter
{
public:
  KalmanFilter()
  {
    R = 0.01;   // Measurement error variance
    Q = 0.1;    // Process error variance
    xhat = 0.0; // Initial estimate
    P = 1.0;    // Initial estimate error
    K = 0.1;    // Initial gain
  }

  float applyFilter(float Z)
  {
    // Update Time
    xhatminus = xhat;
    Pminus = P + Q;

    // Update Measurement
    K = Pminus / (Pminus + R);
    xhat = xhatminus + K * (Z - xhatminus);
    P = (1 - K) * Pminus;

    return xhat; // Return filtered values
  }

private:
  float R, Q;                 // Filter parameters
  float xhat, P;              // Status and Errors
  float xhatminus, Pminus, K; // Time and Measurement Update Variables
};

// Creating a KalmanFilter Object
KalmanFilter kalmanFilterGyroX, kalmanFilterGyroY, kalmanFilterGyroZ;
KalmanFilter kalmanFilterAccX, kalmanFilterAccY, kalmanFilterAccZ;
KalmanFilter kalmanFilterQuatW, kalmanFilterQuatX, kalmanFilterQuatY, kalmanFilterQuatZ;

RobotSensor::RobotSensor()
{
}

RobotSensor::~RobotSensor()
{
  DEBUG_SERIAL.end();
}

bool RobotSensor::init(void)
{
  DEBUG_SERIAL.begin(57600);

  initBumper();
  initIR();
  initSonar();
  initLED();

  uint8_t get_error_code = 0x00;

#if defined NOETIC_SUPPORT
  battery_state_msg_.temperature = NAN;
#endif

  battery_state_msg_.current = NAN;
  battery_state_msg_.charge = NAN;
  battery_state_msg_.capacity = NAN;
  battery_state_msg_.design_capacity = NAN;
  battery_state_msg_.percentage = NAN;

  get_error_code = imu_.begin();

  if (get_error_code != 0x00)
    DEBUG_SERIAL.println("Failed to init Sensor");
  else
    DEBUG_SERIAL.println("Success to init Sensor");

  return get_error_code;
}

void RobotSensor::initIMU(void)
{
  imu_.begin();
}

void RobotSensor::updateIMU(void)
{
  imu_.update();
}

void RobotSensor::calibrationGyro()
{
  uint32_t pre_time;
  uint32_t t_time;

  const uint8_t led_ros_connect = 3;

  imu_.SEN.gyro_cali_start();

  t_time = millis();
  pre_time = millis();

  while (!imu_.SEN.gyro_cali_get_done())
  {
    imu_.update();

    if (millis() - pre_time > 5000)
    {
      break;
    }
    if (millis() - t_time > 100)
    {
      t_time = millis();
      setLedToggle(led_ros_connect);
    }
  }
}

sensor_msgs::Imu RobotSensor::getIMU(void)
{
  // Filtering Gyro data
  imu_msg_.angular_velocity.x = kalmanFilterGyroX.applyFilter(imu_.SEN.gyroADC[0] * GYRO_FACTOR);
  imu_msg_.angular_velocity.y = kalmanFilterGyroY.applyFilter(imu_.SEN.gyroADC[1] * GYRO_FACTOR);
  imu_msg_.angular_velocity.z = kalmanFilterGyroZ.applyFilter(imu_.SEN.gyroADC[2] * GYRO_FACTOR);

  imu_msg_.angular_velocity_covariance[0] = 0.02;
  imu_msg_.angular_velocity_covariance[1] = 0;
  imu_msg_.angular_velocity_covariance[2] = 0;
  imu_msg_.angular_velocity_covariance[3] = 0;
  imu_msg_.angular_velocity_covariance[4] = 0.02;
  imu_msg_.angular_velocity_covariance[5] = 0;
  imu_msg_.angular_velocity_covariance[6] = 0;
  imu_msg_.angular_velocity_covariance[7] = 0;
  imu_msg_.angular_velocity_covariance[8] = 0.02;

  // Filtering acceleration data
  imu_msg_.linear_acceleration.x = kalmanFilterAccX.applyFilter(imu_.SEN.accADC[0] * ACCEL_FACTOR);
  imu_msg_.linear_acceleration.y = kalmanFilterAccY.applyFilter(imu_.SEN.accADC[1] * ACCEL_FACTOR);
  imu_msg_.linear_acceleration.z = kalmanFilterAccZ.applyFilter(imu_.SEN.accADC[2] * ACCEL_FACTOR);

  imu_msg_.linear_acceleration_covariance[0] = 0.04;
  imu_msg_.linear_acceleration_covariance[1] = 0;
  imu_msg_.linear_acceleration_covariance[2] = 0;
  imu_msg_.linear_acceleration_covariance[3] = 0;
  imu_msg_.linear_acceleration_covariance[4] = 0.04;
  imu_msg_.linear_acceleration_covariance[5] = 0;
  imu_msg_.linear_acceleration_covariance[6] = 0;
  imu_msg_.linear_acceleration_covariance[7] = 0;
  imu_msg_.linear_acceleration_covariance[8] = 0.04;

  // Filtering Direction value (quaternion)
  imu_msg_.orientation.w = kalmanFilterQuatW.applyFilter(imu_.quat[0]);
  imu_msg_.orientation.x = kalmanFilterQuatX.applyFilter(imu_.quat[1]);
  imu_msg_.orientation.y = kalmanFilterQuatY.applyFilter(imu_.quat[2]);
  imu_msg_.orientation.z = kalmanFilterQuatZ.applyFilter(imu_.quat[3]);

  imu_msg_.orientation_covariance[0] = 0.0025;
  imu_msg_.orientation_covariance[1] = 0;
  imu_msg_.orientation_covariance[2] = 0;
  imu_msg_.orientation_covariance[3] = 0;
  imu_msg_.orientation_covariance[4] = 0.0025;
  imu_msg_.orientation_covariance[5] = 0;
  imu_msg_.orientation_covariance[6] = 0;
  imu_msg_.orientation_covariance[7] = 0;
  imu_msg_.orientation_covariance[8] = 0.0025;

  return imu_msg_;
}

float *RobotSensor::getOrientation(void)
{
  static float orientation[4];

  orientation[0] = imu_.quat[0];
  orientation[1] = imu_.quat[1];
  orientation[2] = imu_.quat[2];
  orientation[3] = imu_.quat[3];

  return orientation;
}

sensor_msgs::MagneticField RobotSensor::getMag(void)
{
  mag_msg_.magnetic_field.x = imu_.SEN.magADC[0] * MAG_FACTOR;
  mag_msg_.magnetic_field.y = imu_.SEN.magADC[1] * MAG_FACTOR;
  mag_msg_.magnetic_field.z = imu_.SEN.magADC[2] * MAG_FACTOR;

  mag_msg_.magnetic_field_covariance[0] = 0.0048;
  mag_msg_.magnetic_field_covariance[1] = 0;
  mag_msg_.magnetic_field_covariance[2] = 0;
  mag_msg_.magnetic_field_covariance[3] = 0;
  mag_msg_.magnetic_field_covariance[4] = 0.0048;
  mag_msg_.magnetic_field_covariance[5] = 0;
  mag_msg_.magnetic_field_covariance[6] = 0;
  mag_msg_.magnetic_field_covariance[7] = 0;
  mag_msg_.magnetic_field_covariance[8] = 0.0048;

  return mag_msg_;
}

float RobotSensor::checkVoltage(void)
{
  float vol_value;

  vol_value = getPowerInVoltage();

  return vol_value;
}

uint8_t RobotSensor::checkPushButton(void)
{
  return getPushButton();
}

void RobotSensor::melody(uint16_t *note, uint8_t note_num, uint8_t *durations)
{
  for (int thisNote = 0; thisNote < note_num; thisNote++)
  {
    int noteDuration = 1000 / durations[thisNote];
    tone(BDPIN_BUZZER, note[thisNote], noteDuration);

    int pauseBetweenNotes = noteDuration * 1.30;
    delay(pauseBetweenNotes);
    noTone(BDPIN_BUZZER);
  }
}

void RobotSensor::makeSound(uint8_t index)
{
  const uint16_t NOTE_C4 = 262;
  const uint16_t NOTE_D4 = 294;
  const uint16_t NOTE_E4 = 330;
  const uint16_t NOTE_F4 = 349;
  const uint16_t NOTE_G4 = 392;
  const uint16_t NOTE_A4 = 440;
  const uint16_t NOTE_B4 = 494;
  const uint16_t NOTE_C5 = 523;

  uint16_t note[8] = {0, 0};
  uint8_t duration[8] = {0, 0};

  switch (index)
  {
  case 1:
    note[0] = NOTE_C4;
    duration[0] = 4;
    note[1] = NOTE_D4;
    duration[1] = 4;
    note[2] = NOTE_E4;
    duration[2] = 4;
    note[3] = NOTE_F4;
    duration[3] = 4;
    note[4] = NOTE_G4;
    duration[4] = 4;
    note[5] = NOTE_A4;
    duration[5] = 4;
    note[6] = NOTE_B4;
    duration[6] = 4;
    note[7] = NOTE_C5;
    duration[7] = 4;
    break;

  default:
    note[0] = NOTE_C4;
    duration[0] = 4;
    note[1] = NOTE_D4;
    duration[1] = 4;
    note[2] = NOTE_E4;
    duration[2] = 4;
    note[3] = NOTE_F4;
    duration[3] = 4;
    note[4] = NOTE_G4;
    duration[4] = 4;
    note[5] = NOTE_A4;
    duration[5] = 4;
    note[6] = NOTE_B4;
    duration[6] = 4;
    note[7] = NOTE_C5;
    duration[7] = 4;
    break;
  }

  melody(note, 8, duration);
}

void RobotSensor::initBumper(void)
{
  ollo_.begin(3, TOUCH_SENSOR);
  ollo_.begin(4, TOUCH_SENSOR);
}

uint8_t RobotSensor::checkPushBumper(void)
{
  uint8_t push_state = 0;

  if (ollo_.read(3, TOUCH_SENSOR) == HIGH)
    push_state = 2;
  else if (ollo_.read(4, TOUCH_SENSOR) == HIGH)
    push_state = 1;
  else
    push_state = 0;

  return push_state;
}

void RobotSensor::initIR(void)
{
  ollo_.begin(2, IR_SENSOR);
}

float RobotSensor::getIRsensorData(void)
{
  float ir_data = ollo_.read(2, IR_SENSOR);

  return ir_data;
}

void RobotSensor::initSonar(void)
{
  sonar_pin_.trig = BDPIN_GPIO_1;
  sonar_pin_.echo = BDPIN_GPIO_2;

  pinMode(sonar_pin_.trig, OUTPUT);
  pinMode(sonar_pin_.echo, INPUT);
}

void RobotSensor::updateSonar(uint32_t t)
{
  static uint32_t t_time = 0;
  static bool make_pulse = TRUE;
  static bool get_duration = FALSE;

  float distance = 0.0, duration = 0.0;

  if (make_pulse == TRUE)
  {
    digitalWrite(sonar_pin_.trig, HIGH);

    if (t - t_time >= 10)
    {
      digitalWrite(sonar_pin_.trig, LOW);

      get_duration = TRUE;
      make_pulse = FALSE;

      t_time = t;
    }
  }

  if (get_duration == TRUE)
  {
    duration = pulseIn(sonar_pin_.echo, HIGH);
    distance = ((float)(340 * duration) / 10000) / 2;

    make_pulse = TRUE;
    get_duration = FALSE;
  }

  sonar_data_ = distance;
}

float RobotSensor::getSonarData(void)
{
  float distance = 0.0;

  distance = sonar_data_;

  return distance;
}

float RobotSensor::getIlluminationData(void)
{
  uint16_t light;

  light = analogRead(A1);

  return light;
}

void RobotSensor::initLED(void)
{
  led_pin_array_.front_left = BDPIN_GPIO_4;
  led_pin_array_.front_right = BDPIN_GPIO_6;
  led_pin_array_.back_left = BDPIN_GPIO_8;
  led_pin_array_.back_right = BDPIN_GPIO_10;

  pinMode(led_pin_array_.front_left, OUTPUT);
  pinMode(led_pin_array_.front_right, OUTPUT);
  pinMode(led_pin_array_.back_left, OUTPUT);
  pinMode(led_pin_array_.back_right, OUTPUT);
}

void RobotSensor::setLedPattern(double linear_vel, double angular_vel)
{
  if (linear_vel > 0.0 && angular_vel == 0.0) // front
  {
    digitalWrite(led_pin_array_.front_left, HIGH);
    digitalWrite(led_pin_array_.front_right, HIGH);
    digitalWrite(led_pin_array_.back_left, LOW);
    digitalWrite(led_pin_array_.back_right, LOW);
  }
  else if (linear_vel >= 0.0 && angular_vel > 0.0) // front left
  {
    digitalWrite(led_pin_array_.front_left, HIGH);
    digitalWrite(led_pin_array_.front_right, LOW);
    digitalWrite(led_pin_array_.back_left, LOW);
    digitalWrite(led_pin_array_.back_right, LOW);
  }
  else if (linear_vel >= 0.0 && angular_vel < 0.0) // front right
  {
    digitalWrite(led_pin_array_.front_left, LOW);
    digitalWrite(led_pin_array_.front_right, HIGH);
    digitalWrite(led_pin_array_.back_left, LOW);
    digitalWrite(led_pin_array_.back_right, LOW);
  }
  else if (linear_vel < 0.0 && angular_vel == 0.0) // back
  {
    digitalWrite(led_pin_array_.front_left, LOW);
    digitalWrite(led_pin_array_.front_right, LOW);
    digitalWrite(led_pin_array_.back_left, HIGH);
    digitalWrite(led_pin_array_.back_right, HIGH);
  }
  else if (linear_vel <= 0.0 && angular_vel > 0.0) // back right
  {
    digitalWrite(led_pin_array_.front_left, LOW);
    digitalWrite(led_pin_array_.front_right, LOW);
    digitalWrite(led_pin_array_.back_left, LOW);
    digitalWrite(led_pin_array_.back_right, HIGH);
  }
  else if (linear_vel <= 0.0 && angular_vel < 0.0) // back left
  {
    digitalWrite(led_pin_array_.front_left, LOW);
    digitalWrite(led_pin_array_.front_right, LOW);
    digitalWrite(led_pin_array_.back_left, HIGH);
    digitalWrite(led_pin_array_.back_right, LOW);
  }
  else
  {
    digitalWrite(led_pin_array_.front_left, LOW);
    digitalWrite(led_pin_array_.front_right, LOW);
    digitalWrite(led_pin_array_.back_left, LOW);
    digitalWrite(led_pin_array_.back_right, LOW);
  }
}
