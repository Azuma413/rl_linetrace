#include <Arduino.h>
#include <Servo.h>
Servo servo;
int refrecter_pin = A0;
int volume_pin = A1;
int servo_pin = 9;
int angle = 0;
const int buff_size = 10;
int ref_val_list[buff_size];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(volume_pin, INPUT);
  delay(10);
  pinMode(refrecter_pin, INPUT);
  delay(10);
  servo.attach(servo_pin);
}

void loop() {
  int volume_val = analogRead(volume_pin);
  int ref_val = analogRead(refrecter_pin);
  Serial.print(volume_val);
  Serial.print(",");
  for (int i = 0; i < buff_size - 1; i++) {
    ref_val_list[i] = ref_val_list[i + 1];
  }
  ref_val_list[buff_size - 1] = ref_val;
  int ref_val_sum = 0;
  for (int i = 0; i < buff_size; i++) {
    ref_val_sum += ref_val_list[i];
  }
  ref_val = ref_val_sum / buff_size;
  Serial.println(ref_val);
  // 平均値を目標角度に変換
  angle = ref_val / 1024.0 * 180;
  servo.write(angle);
  delay(100);
}
