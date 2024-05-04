#include <Arduino.h>
#include <Servo.h>

ToolKit toolkit;

void setup() {
  toolkit = ToolKit(buff_size=10);
}

void loop() {
  // 値を取得
  int vol_val = analogRead(toolkit.get_volume_pin());
  int ref_val = analogRead(toolkit.get_reflector_pin());
  // バッファーサイズに応じて配列を更新
  toolkit.update_ref_val_list(ref_val);
  // バッファーサイズに応じて平均値を計算
  int average_ref_val = toolkit.get_average_ref_val();
  // 値を出力
  toolkit.print_val(vol_val, average_ref_val);
  delay(100); // 10Hzでループ
}

// Arduino課題用のクラス
class ToolKit {
  private:
    Servo servo;
    int reflector_pin;
    int volume_pin;
    int servo_pin;
    int ref_val_list[];
    int count = 0;
  
  public:
    void ToolKit(int reflector_pin=A0, int volume_pin=A1, int servo_pin=9, int buff_size=10) {
      this->reflector_pin = reflector_pin;
      this->volume_pin = volume_pin;
      this->servo_pin = servo_pin;
      ref_val_list = new int[buff_size];
      Serial.begin(57600);
      pinMode(volume_pin, INPUT); // 可変抵抗のピンを初期化
      pinMode(reflector_pin, INPUT); // フォトリフレクタのピンを初期化
      servo.attach(servo_pin); // サーボモータのピンを初期化
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
      if(count < buff_size) {
        ref_val_list[count] = ref_val;
        count++;
      }else {
        count = 0;
      }
    }
    // 平均値を計算する関数
    int get_average_ref_val() {
      int ref_val_sum = 0;
      for (int i = 0; i < buff_size; i++) {
        ref_val_sum += ref_val_list[i];
      }
      return ref_val_sum / buff_size;
    }
    // 値をシリアル出力する関数
    void print_val(int val1, int val2) {
      Serial.print(val1);
      Serial.print(",");
      Serial.println(val2);
    }
    //10bitでサーボの角度を設定する関数
    void set10bit_angle(int val) {
      servo.write(val / 1024.0 * 180);
    }
};