from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep
import cv2 # pip install opencv-python
import gym
from gym import spaces
import numpy as np
from dm_env import specs
import socket
import time

# 定数の宣言
CHANGE_MOTOR = True # モータの順番を入れ替えるか。Trueの場合、モータ0とモータ1の制御が入れ替わる
MAX_UDP_PACKET_SIZE = 5000
MAX_SPEED = 60 # 最大速度[mm/s]
NOMINAL_SPEED = 30 # ノミナル速度[mm/step]

class MyController(gym.Env):
    def __init__(self, env_config=None):
        super(MyController, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.obs_size = 64 # 観測画像のサイズ
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 3), dtype=np.float32)
        self.reward_range = (-1, 1) # rewardの範囲
        self.simulator = None # シミュレータのインスタンスを保持する変数
        self.ain1 = DigitalOutputDevice(24) # モーター1の制御ピン1
        self.ain2 = DigitalOutputDevice(23) # モーター1の制御ピン2
        self.pwma = PWMOutputDevice(18) # モーター1のPWM制御ピン
        self.bin1 = DigitalOutputDevice(25) # モーター2の制御ピン1
        self.bin2 = DigitalOutputDevice(1) # モーター2の制御ピン2
        self.pwmb = PWMOutputDevice(13) # モーター2のPWM制御ピン
        camera_idx = 0
        while True: # キャプチャに成功するまで繰り返す
            self.cap = cv2.VideoCapture(camera_idx)
            if self.cap.isOpened():
                print(f"camera{camera_idx} opened")
                break
            if camera_idx > 30:
                raise ValueError("camera not found")
        self.duty = 0.7 # 10mm/stepとなるように調整
        self.action_discount = 0.6
        self.action_average = 0
        self.action = 0
        self.obs = None
        self.prior_action = 0
        self.action_limit = 0.2
        self.time = None
        self.theta = 0
        self.freq = 1.0
        self.thresh = 0.5 # 2値化の閾値（平均値の何倍か）
        # udp通信の設定
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.address = ('192.168.0.255', 12345)
        self.udp_socket.bind(self.address)

    # デコンストラクタ
    def __del__(self):
        self.cap.release()
        self.control(0, move=False)
        self.udp_socket.close()
        
    def reset(self):
        """
        環境のリセット関数
        """
        # モーターを停止
        self.control(0, move=False)
        # 初期画像を取得
        self.obs = self.make_obs()
        obs = np.transpose(self.obs, (2, 0, 1)).astype(np.float32)
        print("reset controller")
        return { 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}

    def step(self, action):
        """
        環境を1ステップ進める関数
        action[0]: 進行方向に対する角度
        """
        self.prior_action = self.action
        self.action = action
        self.action_average = self.action_average + action*self.action_discount*self.action_limit
        if self.action_average > 1:
            self.action_average -= 2
        elif self.action_average < -1:
            self.action_average += 2
        self.theta = (self.action_average + self.action*self.action_limit)*np.pi
        self.control(self.theta) # モーターを制御
        self.obs = self.make_obs() # 観測を取得
        obs = np.transpose(self.obs, (2, 0, 1)).astype(np.float32)
        
        return { 'observation': obs, 'reward': np.array([0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': action.astype(np.float32)}
        
    def observation_spec(self):
        return specs.Array(shape=(3, self.obs_size, self.obs_size), dtype=np.float32, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=(1,), dtype=np.float32, name='action', minimum=-1, maximum=1)

    def render(self, mode='rgb_array'):
        """
        記録用の画像を返す関数
        obsも見れるようにしたい。
        """
        # 実行周期を書き込む
        now = time.time()
        t = 1
        if self.time is not None:
            t = now - self.time
        self.time = now
        self.freq = 1/t
        print(f"freq: {self.freq}")
        image = None
        if self.obs is not None:
            image = self.obs.copy()*255
            image[:,:,1] = image[:,:,0]
            image[:,:,2] = image[:,:,0]
            # 4倍に拡大
            image = cv2.resize(image, (256, 256))
            image = image.astype(np.uint8)
        else:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        h, w = image.shape[:2]
        # theta方向に矢印を描画
        image = cv2.arrowedLine(image, (w//2, h//2), (w//2+int(np.cos(self.theta)*h//4), h//2+int(np.sin(self.theta)*h//4)), (255, 0, 0), 5)
        # 周波数を描画
        image = cv2.putText(image, f"{self.freq:.2f}Hz", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.send_udp(image)
        return image

    def control(self, theta, move=True):
        """
        モーターを制御する関数
        """
        if not move:
            self.ain1.off()
            self.ain2.off()
            self.bin1.off()
            self.bin2.off()
            self.pwma.value = 0
            self.pwmb.value = 0
            return
        self.duty = NOMINAL_SPEED*self.freq/MAX_SPEED # 周波数に応じてdutyを変更
        if self.duty > 1:
            self.duty = 1
        elif self.duty < 0:
            self.duty = 0
        speed = [np.cos(theta + np.pi/4)*self.duty, np.sin(theta + np.pi/4)*self.duty]
        if CHANGE_MOTOR:
            speed = speed[::-1]
        # motor0 control
        if speed[0] > 0:
            self.ain1.on()
            self.ain2.off()
            self.pwma.value = speed[0]
        elif speed[0] < 0:
            self.ain1.off()
            self.ain2.on()
            self.pwma.value = -speed[0]
        else:
            self.ain1.off()
            self.ain2.off()
            self.pwma.value = 0
        # motor1 control
        if speed[1] > 0:
            self.bin1.off()
            self.bin2.on()
            self.pwmb.value = speed[1]
        elif speed[1] < 0:
            self.bin1.on()
            self.bin2.off()
            self.pwmb.value = -speed[1]
        else:
            self.bin1.off()
            self.bin2.off()
            self.pwmb.value = 0

    def make_obs(self):
        """
        画像を取得して観測を作成する関数
        """
        while True:
            for _ in range(10): # 最新の画像を取得するために10回読み込む
                ret, frame = self.cap.read()
            if not ret:
                continue
            # frameを正方形にクリップ
            h, w = frame.shape[:2]
            if h > w:
                frame = frame[(h-w)//2:(h+w)//2, :]
            else:
                frame = frame[:, (w-h)//2:(w+h)//2]
            frame = cv2.resize(frame, (64, 64)) # frameを64x64にリサイズ
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # frameをモノクロに変換
            mean = np.mean(frame) # frameの平均値を取得
            print(f"mean: {mean}")
            var = np.var(frame) # frameの分散を取得
            print(f"var: {var}")
            if var < 0.01: # 分散が小さい場合は線を検出していないと判断。観測を白で埋める
                frame = np.ones_like(frame)
            else:
                thereshold = mean*self.thresh # 平均値に応じて2値化の閾値を変更
                _, frame = cv2.threshold(frame, thereshold, 255, cv2.THRESH_BINARY) # frameを2値化
                frame = frame.astype(np.float32)/255 # 観測をfloat32に変換して正規化
            frame = np.dstack([frame, np.zeros_like(frame), np.zeros_like(frame)])
            frame[:,:,1] = (self.action_average + 1)/2
            frame[:,:,2] = (self.prior_action + 1)/2
            return frame
        
    def send_udp(self, image):
        """
        画像をUDPで送信する関数
        """
        byte = cv2.imencode('.png', image)[1].tobytes()
        chunks = [byte[i:i+MAX_UDP_PACKET_SIZE] for i in range(0, len(byte), MAX_UDP_PACKET_SIZE)]
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            header = i.to_bytes(4, 'big') + total_chunks.to_bytes(4, 'big')
            is_last_chunk = (1 if i == total_chunks - 1 else 0).to_bytes(1, 'big')
            udp_packet = header + is_last_chunk + chunk
            self.udp_socket.sendto(udp_packet, self.address)
            # print(f"send udp: {len(udp_packet)}")
