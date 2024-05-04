#include <Arduino.h>
#include <Servo.h>

// グローバル変数の宣言
Servo servo;
int refrecter_pin = A0;
int volume_pin = A1;
int servo_pin = 9;
const int buff_size = 10;
int ref_val_list[buff_size];

void setup() {
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
  Serial.print(ref_val);
  Serial.print(",");
  // バッファーサイズに応じて配列を更新
  for (int i = 0; i < buff_size - 1; i++) {
    ref_val_list[i] = ref_val_list[i + 1];
  }
  ref_val_list[buff_size - 1] = ref_val;
  // バッファーサイズに応じて平均値を計算
  int ref_val_sum = 0;
  for (int i = 0; i < buff_size; i++) {
    ref_val_sum += ref_val_list[i];
  }
  // 平均値を出力
  ref_val = ref_val_sum / buff_size;
  Serial.println(ref_val);
}
