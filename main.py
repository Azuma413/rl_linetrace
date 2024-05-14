# これを起動すればシステムが動き出す 
# source rl_env/bin/activate
# pip install gpiozero
# pip install lgpio
# pip install RPi.GPIO
# pip install pigpio
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

ain1 = DigitalOutputDevice(1)
ain2 = DigitalOutputDevice(2)
pwma = PWMOutputDevice(12)

def control(speed):
    """
    モーターを制御する関数
    speed: -1~1の範囲でモーターの速度を指定
    """
    if speed > 0:
        ain1.on()
        ain2.off()
        pwma.value = speed
    elif speed < 0:
        ain1.off()
        ain2.on()
        pwma.value = -speed
    else:
        ain1.off()
        ain2.off()
        pwma.value = 0
        
if __name__ == "__main__":
    count = 0
    flag = True
    while True:
        if count == 10:
            flag = False
        elif count == -10:
            flag = True
        control(count*0.1)
        sleep(0.5)
        if flag:
            count += 1
        else:
            count -= 1