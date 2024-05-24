from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep
import cv2 # pip install opencv-python
import gym
from gym import spaces
import numpy as np
from dm_env import specs

# 定数の宣言
LINE_THREASHOLD = 128 # 画像の2値化の閾値 0-255の範囲で指定

class MyController(gym.Env):
    def __init__(self, env_config=None):
        super(MyController, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.obs_size = 64 # 観測画像のサイズ
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 1), dtype=np.float32)
        self.reward_range = (-1, 1) # rewardの範囲
        self.simulator = None # シミュレータのインスタンスを保持する変数
        self.ain1 = DigitalOutputDevice(24) # モーター1の制御ピン1
        self.ain2 = DigitalOutputDevice(23) # モーター1の制御ピン2
        self.pwma = PWMOutputDevice(18) # モーター1のPWM制御ピン
        self.bin1 = DigitalOutputDevice(7) # モーター2の制御ピン1
        self.bin2 = DigitalOutputDevice(1) # モーター2の制御ピン2
        self.pwmb = PWMOutputDevice(13) # モーター2のPWM制御ピン
        self.cap = cv2.VideoCapture(0) # カメラのキャプチャ
        self.action = None
        self.image = None
        
    def reset(self):
        """
        環境のリセット関数
        """
        # モーターを停止
        self.ain1.off()
        self.ain2.off()
        self.pwma.value = 0
        self.bin1.off()
        self.bin2.off()
        self.pwmb.value = 0
        # 初期画像を取得
        obs = self.make_obs()
        print("reset controller")
        return { 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}

    def step(self, action):
        """
        環境を1ステップ進める関数
        """
        print("action: ", action)
        self.control(action[0], 0) # モーター1を制御
        self.control(action[1], 1) # モーター2を制御
        obs = self.make_obs() # 観測を取得
        reward = 0 # 学習はしないので報酬は0
        done = False
        self.action = action
        return { 'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done , 'action': action.astype(np.float32)}
        
    def observation_spec(self):
        return specs.Array(shape=(1, self.obs_size, self.obs_size), dtype=np.float32, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1, maximum=1)

    def render(self, mode='rgb_array'):
        """
        記録用の画像を返す関数
        """
        frame = None
        # action方向に矢印を描画
        if self.action is not None:
            frame = self.image.copy()
            h, w = frame.shape[:2]
            cv2.arrowedLine(frame, (w//2, h//2), (w//2+int(self.action[0]*w//4), h//2+int(self.action[1]*w//4)), (255, 0, 0), 2)
        else:
            frame = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        return frame

    def control(self, speed, motor):
        """
        モーターを制御する関数
        speed: -1~1の範囲でモーターの速度を指定
        """
        if motor == 0:
            if speed > 0:
                self.ain1.on()
                self.ain2.off()
                self.pwma.value = speed
            elif speed < 0:
                self.ain1.off()
                self.ain2.on()
                self.pwma.value = -speed
            else:
                self.ain1.off()
                self.ain2.off()
                self.pwma.value = 0
        elif motor == 1:
            if speed > 0:
                self.bin1.on()
                self.bin2.off()
                self.pwmb.value = speed
            elif speed < 0:
                self.bin1.off()
                self.bin2.on()
                self.pwmb.value = -speed
            else:
                self.bin1.off()
                self.bin2.off()
                self.pwmb.value = 0
        else:
            raise ValueError("motor must be 0 or 1")

    def make_obs(self):
        """
        画像を取得して観測を作成する関数
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.image = frame
            # frameを正方形にクリップ
            h, w = frame.shape[:2]
            if h > w:
                frame = frame[(h-w)//2:(h+w)//2, :]
            else:
                frame = frame[:, (w-h)//2:(w+h)//2]
            # frameを64x64にリサイズ
            frame = cv2.resize(frame, (64, 64))
            # frameをモノクロに変換
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frameを2値化
            _, frame = cv2.threshold(frame, LINE_THREASHOLD, 255, cv2.THRESH_BINARY)
            frame = frame.astype(np.float32) # 観測をfloat32に変換
            # frameを正規化
            frame /= 255
            frame = 1 - frame # 白黒反転
            frame = frame[None,:,:]
            return frame