#include <Arduino.h>
#include <Servo.h>

// グローバル変数の宣言
Servo servo;
int refrecter_pin = A0;
int volume_pin = A1;
int servo_pin = 9;
int angle = 0;

void setup() {
  Serial.begin(9600);
  // 可変抵抗のピンを初期化
  pinMode(volume_pin, INPUT);
  delay(10);
  // フォトリフレクタのピンを初期化
  pinMode(refrecter_pin, INPUT);
  // サーボモータの構造体を初期化
  servo.attach(servo_pin);
}

void loop() {
  // 値を取得
  int volume_val = analogRead(volume_pin);
  int ref_val = analogRead(refrecter_pin);
  // シリアル通信で値を出力
  Serial.print(volume_val);
  Serial.print(",");
  Serial.println(ref_val);
  // サーボモータの角度を設定
  angle = ref_val / 1024.0 * 180;
  servo.write(angle);
  delay(100);
}
