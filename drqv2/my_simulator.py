# px:mm = 1:1
# ロボットの位置やカーブの中心はmmで管理する。
# 画像の大きさは560*720とする。
# 観測として与える画像はロボットを中心とする64*64の画像とする。
# 円弧は以下のように描画する
# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_8, shift=0)
# center: 中心座標, axes: 長軸と短軸の長さ, angle: 回転角度(基本的には0), startAngle: 描画する円弧の開始角度, endAngle: 描画する円弧の終了角度

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dm_env import specs
import pandas as pd

class MyEnv(gym.Env):
    def __init__(self, env_config=None):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.obs_size = 64 # 観測画像のサイズ
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 1), dtype=np.float32)
        self.reward_range = (-1, 1) # rewardの範囲
        self.simulator = None # シミュレータのインスタンスを保持する変数
        self.episode_length = 300 # エピソードの最大ステップ数
        self.count = 0 # 現在のステップ数
        
    def reset(self):
        # self.simulator = MySimulator(init_pos=[640, 200], obs_size=[self.obs_size, self.obs_size]) # 初期座標を変更する場合 
        self.simulator = MySimulator(obs_size=[self.obs_size, self.obs_size]) # シミュレータのインスタンスを生成
        obs, _, _ = self.simulator.simulate(np.array([0, 0])) # 初期観測を取得
        self.count = 0 # ステップ数をリセット
        obs = obs.astype(np.float32) # 観測をfloat32に変換
        obs = obs[None,:,:] # (64, 64) -> (1, 64, 64)
        return { 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}
    
    def step(self, action):
        self.count += 1 # ステップ数をカウント
        obs, reward, done = self.simulator.simulate(action) # シミュレータを1ステップ進める
        obs = obs.astype(np.float32) # 観測をfloat32に変換
        obs = obs[None,:,:] # (64, 64) -> (1, 64, 64)
        if self.count >= self.episode_length: # 最大ステップ数に達したらdoneをTrueにする
            done = True
        return { 'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done , 'action': action.astype(np.float32)}
    
    def observation_spec(self):
        return specs.Array(shape=(1, self.obs_size, self.obs_size), dtype=np.float32, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1, maximum=1)
    
    def render(self, mode='rgb_array'):
        return self.simulator.render() # シミュレータのrender関数を呼び出す

class MySimulator:
    """
    機能追加リスト
    ・マップの種類を複数用意（ランダム生成できたらなお良い）
    ・観測のyaw方向のノイズを追加(回転させる)
    ・観測のpitch方向のノイズを追加(観測範囲を拡張してからリサイズ)
    """
    def __init__(self, init_pos=[360, 80], obs_size=[64, 64]):
        self.image_size = [560, 720] # 画像のサイズ
        self.obs_size = obs_size # 観測画像のサイズ
        self.robot_pos = np.array(init_pos) # ロボットの初期位置
        update_rate = 20 # Hz 1ステップの更新頻度
        max_speed = 300 # mm/s ロボットの最大速度
        self.max_length = max_speed / update_rate # mm 1ステップで進む最大距離
        self.centers = np.array([ # 円弧の中心座標
            [520, 200],
            [520, 400],
            [360, 400],
            [200, 400],
            [200, 200]
        ])
        self.ranges = np.array([ # 円弧の半径
            120,
            80,
            80,
            80,
            120
        ])
        self.areas = np.array([ # 円弧や直線のエリア
            [[520, 720], [0, 400]],
            [[360, 520], [200, 400]],
            [[200, 520], [400, 560]],
            [[200, 360], [200, 400]],
            [[0, 200], [0, 400]],
            [[200, 520], [0, 200]]
        ])
        self.obs_map = self.image_generator() # 観測画像を生成
        self.reward = 0
        self.is_in_area = False
        self.action = np.array([0, 0])
    
    def image_generator(self, line_thickness=20):
        """
        mapの画像を生成する関数
        """
        image = np.ones(self.image_size)
        angles = np.array([
            [-90, 90],
            [180, 270],
            [0, 180],
            [270, 360],
            [90, 270]
        ])
        for center, r, angle in zip(self.centers, self.ranges, angles):
            cv2.ellipse(image, tuple(center), (r, r), 0, angle[0], angle[1], 0, line_thickness) # 円弧を描画
        cv2.line(image, (200, 80), (520, 80), 0, line_thickness) # 直線を描画
        return image
    
    def simulate(self, action):
        """
        action:正規化された[x, y]の速度ベクトル
        """
        prior_pos = self.robot_pos.copy() # 1ステップ前のロボットの位置
        self.robot_pos += (self.max_length * action).astype(int) # ロボットの位置を更新
        self.is_in_area = False
        self.reward = 0
        for i, area in enumerate(self.areas): # エリアごとに報酬を計算
            if area[0][0] <= self.robot_pos[0] < area[0][1] and area[1][0] <= self.robot_pos[1] < area[1][1]:
                # robotがareaに入った時の処理
                length = 0
                error = 0
                if i == 5: # 直線のエリア
                    length = self.robot_pos[0] - prior_pos[0]
                    error = np.abs(self.robot_pos[1] - 80)
                else: # 円弧のエリア
                    theta = []
                    theta.append(np.arctan2(self.robot_pos[1] - self.centers[i][1], self.robot_pos[0] - self.centers[i][0]))
                    theta.append(np.arctan2(prior_pos[1] - self.centers[i][1], prior_pos[0] - self.centers[i][0]))
                    for j in range(2):
                        if theta[j] < 0:
                            theta[j] += 2*np.pi
                    length = self.ranges[i] * (theta[0] - theta[1])
                    error = np.abs(np.sqrt((self.robot_pos[0] - self.centers[i][0])**2 + (self.robot_pos[1] - self.centers[i][1])**2) - self.ranges[i])
                    if i == 1 or i == 3: # 時計回りの場合は負の値にする
                        length = -length
                self.reward = np.clip(length/self.max_length, -1, 1) - np.tanh(error/20)*2 # 報酬を計算
                self.is_in_area = True
                break
        obs = self.obs_map[int(self.robot_pos[1]-self.obs_size[1]//2):int(self.robot_pos[1]+self.obs_size[1]//2), int(self.robot_pos[0]-self.obs_size[0]//2):int(self.robot_pos[0]+self.obs_size[0]//2)]
        if not self.is_in_area: # エリア外に出た場合は報酬を-1にする
            self.reward = -1
        if obs.shape != (self.obs_size[1], self.obs_size[0]):
            self.is_in_area = False
            self.reward = -1
            # obsのサイズが合わない（=マップからはみ出している）場合は白で埋める
            obs = np.ones(self.obs_size)
        # rewardを-1,1にclipする
        self.reward = np.clip(self.reward, -1, 1)
        self.action = action
        return obs, self.reward, not self.is_in_area
    
    def render(self):
        """
        学習の進捗確認用の画像を生成する
        """
        image = np.zeros([self.image_size[0], self.image_size[1], 3])
        image[:,:,:] = self.obs_map[:,:,None] * 255
        cv2.drawMarker(image, (360, 80), (0, 0, 0), cv2.MARKER_CROSS, 40, 10)
        cv2.circle(image, tuple(self.robot_pos), 10, (0, 255, 0), -1)
        # 観測範囲を表示
        cv2.rectangle(image, (self.robot_pos[0]-self.obs_size[0]//2, self.robot_pos[1]-self.obs_size[1]//2), (self.robot_pos[0]+self.obs_size[0]//2, self.robot_pos[1]+self.obs_size[1]//2), (0, 0, 255), 2)
        # 進行方向を表示
        cv2.arrowedLine(image, tuple(self.robot_pos), tuple(self.robot_pos + (self.max_length*self.action).astype(int)), (255, 0, 0), 2)
        # 上下反転
        image = cv2.flip(image, 0)
        # rewardを表示
        cv2.putText(image, 'reward: {:.2f}'.format(self.reward), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 座標を表示
        cv2.putText(image, 'x: {}, y: {}'.format(self.robot_pos[0], self.robot_pos[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 衝突していたら文字を表示
        if not self.is_in_area:
            cv2.putText(image, 'out of area', (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return image