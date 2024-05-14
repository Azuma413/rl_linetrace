#include <Arduino.h>
#include <Servo.h>

// Arduino課題用のクラス
class ToolKit {
  private:
    Servo servo;
    int reflector_pin;
    int volume_pin;
    int servo_pin;
    const static int buff_size = 10;
    int ref_val_list[buff_size];
    int count = 0;
  
  public:
    ToolKit(int reflector_pin=A0, int volume_pin=A1, int servo_pin=9) {
      this->reflector_pin = reflector_pin;
      this->volume_pin = volume_pin;
      this->servo_pin = servo_pin;
      Serial.begin(57600);
      pinMode(volume_pin, INPUT); // 可変抵抗のピンを初期化
      pinMode(reflector_pin, INPUT); // フォトリフレクタのピンを初期化
      this->servo.attach(servo_pin); // サーボモータのピンを初期化
    }
    // フォトリフレクタのピン番号のゲッター
    int get_reflector_pin() {
      return reflector_pin;
    }
    // 可変抵抗のピン番号のゲッター
    int get_volume_pin() {
      return volume_pin;
    }
    // 配列を更新する関数
    void update_ref_val_list(int ref_val) {
      if(count < this->buff_size) {
        ref_val_list[count] = ref_val;
        count++;
      }else {
        count = 0;
      }
    }
    // 平均値を計算する関数
    int get_average_ref_val() {
      int ref_val_sum = 0;
      for (int i = 0; i < this->buff_size; i++) {
        ref_val_sum += ref_val_list[i];
      }
      return ref_val_sum / this->buff_size;
    }
    // 値をシリアル出力する関数
    void print_val(int val1, int val2) {
      Serial.print(val1);
      Serial.print(",");
      Serial.println(val2);
    }
    //10bitでサーボの角度を設定する関数
    void set10bit_angle(int val) {
      this->servo.write(val / 1024.0 * 180);
    }
};

ToolKit toolkit;

void setup() {
  toolkit = ToolKit();
}

void loop() {
  // 値を取得
  int vol_val = analogRead(toolkit.get_volume_pin());
  int ref_val = analogRead(toolkit.get_reflector_pin());
  // 値を出力
  toolkit.print_val(vol_val, ref_val);
  // サーボモータの角度を設定
  toolkit.set10bit_angle(ref_val);
  delay(100); // 10Hzでループ
}