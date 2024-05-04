#include <Arduino.h>
#include <Servo.h>
#include <ArxContainer.h>

// グローバル変数の宣言
Servo servo;
int refrecter_pin = A0;
int volume_pin = A1;
int servo_pin = 9;
int min_buff_size = 1;
int max_buff_size = 100;
int buff_size = 0;
arx::stdx::deque<int, 100> ref_buffer;

void setup() {
  Serial.begin(57600);
  pinMode(volume_pin, INPUT);
  delay(10);
  pinMode(refrecter_pin, INPUT);
  delay(10);
  servo.attach(servo_pin);
}

void loop() {
  // バッファーサイズを可変抵抗の値に応じて変更
  buff_size = analogRead(volume_pin)/1024*100;
  // バッファーサイズの最小値と最大値を適用
  if(buff_size < min_buff_size) buff_size = min_buff_size;
  if(buff_size > max_buff_size) buff_size = max_buff_size;
  // バッファーに値を追加
  ref_buffer.push_back(analogRead(refrecter_pin));
  // バッファーサイズを超えた場合は先頭の値を削除
  if(ref_buffer.size() > buff_size) ref_buffer.pop_front();
  // 平均値を計算
  int sum = 0;
  for(int i = 0; i < ref_buffer.size(); i++){
    sum += ref_buffer[i];
  }
  int avg = sum / ref_buffer.size();
  // 平均値を角度に変換
  int angle = map(avg, 0, 1023, 0, 180);
  servo.write(angle);
}
