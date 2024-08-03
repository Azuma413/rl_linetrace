import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dm_env import specs
import pandas as pd
from collections import deque

class MyEnv2(gym.Env):
    def __init__(self, env_config=None):
        super(MyEnv2, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.obs_size = 64 # 観測画像のサイズ
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 3), dtype=np.float32)
        self.reward_range = (-1, 1) # rewardの範囲
        self.simulator = None # シミュレータのインスタンスを保持する変数
        self.episode_length = 300 # エピソードの最大ステップ数
        self.count = 0 # 現在のステップ数

    def reset(self):
        self.simulator = MySimulator2(obs_size=[self.obs_size, self.obs_size]) # 初期座標を変更する場合 
        obs, _ = self.simulator.simulate(0) # 初期観測を取得
        self.count = 0 # ステップ数をリセット
        obs = obs.astype(np.float32) # 観測をfloat32に変換
        obs = np.transpose(obs, (2, 0, 1)) # [64,64,2]->[2,64,64]
        return { 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0]).astype(np.float32)}

    def step(self, action):
        done = False
        self.count += 1 # ステップ数をカウント
        obs, reward = self.simulator.simulate(action[0]) # シミュレータを1ステップ進める
        obs = obs.astype(np.float32) # 観測をfloat32に変換
        obs = np.transpose(obs, (2, 0, 1))
        if self.count >= self.episode_length: # 最大ステップ数に達したらdoneをTrueにする
            done = True
        return { 'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done , 'action': action}

    def observation_spec(self):
        return specs.Array(shape=(3, self.obs_size, self.obs_size), dtype=np.float32, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=(1,), dtype=np.float32, name='action', minimum=-1, maximum=1)

    def render(self, mode='rgb_array'):
        return self.simulator.render() # シミュレータのrender関数を呼び出す

class AreaInfo:
    def __init__(self, area_idx, start_edge, end_edge, start_pos, end_pos, image_size = [1000,1000]):
        self.area_idx = area_idx
        self.start_pos = [image_size[0]/4*area_idx[0], image_size[1]/4*(3-area_idx[1])]
        self.end_pos = [image_size[0]/4*area_idx[0], image_size[1]/4*(3-area_idx[1])]
        if start_edge == 0:
            self.start_pos[0] += image_size[0]/20*(1+start_pos*3)
            self.start_pos[1] += image_size[1]/4
        elif start_edge == 1:
            self.start_pos[0] += image_size[0]/4
            self.start_pos[1] += image_size[1]/20*(1+start_pos*3)
        elif start_edge == 2:
            self.start_pos[0] += image_size[0]/20*(1+start_pos*3)
        elif start_edge == 3:
            self.start_pos[1] += image_size[1]/20*(1+start_pos*3)

        if end_edge == 0:
            self.end_pos[0] += image_size[0]/20*(1+end_pos*3)
            self.end_pos[1] += image_size[1]/4
        elif end_edge == 1:
            self.end_pos[0] += image_size[0]/4
            self.end_pos[1] += image_size[1]/20*(1+end_pos*3)
        elif end_edge == 2:
            self.end_pos[0] += image_size[0]/20*(1+end_pos*3)
        elif end_edge == 3:
            self.end_pos[1] += image_size[1]/20*(1+end_pos*3)

        self.is_line = False
        if start_edge == 0 and end_edge == 1:
            self.start_angle, self.end_angle = self.set_angle(1)
            corner = [image_size[0]/4*(area_idx[0]+1), image_size[1]/4*(3-area_idx[1]+1)] # startとendの間の角の座標
            if abs(corner[0] - self.start_pos[0]) > abs(corner[1] - self.end_pos[1]):
                self.circle_start = True
                self.radius = abs(corner[1] - self.end_pos[1])
                self.center = [self.start_pos[0] + self.radius, corner[1]]
            else:
                self.circle_start = False
                self.radius = abs(corner[0] - self.start_pos[0])
                self.center = [corner[0], self.end_pos[1] + self.radius]
        elif start_edge == 0 and end_edge == 3:
            self.start_angle, self.end_angle = self.set_angle(2)
            corner = [image_size[0]/4*(area_idx[0]), image_size[1]/4*(3-area_idx[1]+1)]
            if abs(corner[0] - self.start_pos[0]) > abs(corner[1] - self.end_pos[1]):
                self.circle_start = True
                self.radius = abs(corner[1] - self.end_pos[1])
                self.center = [self.start_pos[0] - self.radius, corner[1]]
            else:
                self.circle_start = False
                self.radius = abs(corner[0] - self.start_pos[0])
                self.center = [corner[0], self.end_pos[1] + self.radius]
        elif start_edge == 1 and end_edge == 0:
            self.start_angle, self.end_angle = self.set_angle(1)
            corner = [image_size[0]/4*(area_idx[0]+1), image_size[1]/4*(3-area_idx[1]+1)]
            if abs(corner[1] - self.start_pos[1]) > abs(corner[0] - self.end_pos[0]):
                self.circle_start = True
                self.radius = abs(corner[0] - self.end_pos[0])
                self.center = [corner[0], self.start_pos[1] + self.radius]
            else:
                self.circle_start = False
                self.radius = abs(corner[1] - self.start_pos[1])
                self.center = [self.end_pos[0] + self.radius, corner[1]]
        elif start_edge == 1 and end_edge == 2:
            self.start_angle, self.end_angle = self.set_angle(4)
            corner = [image_size[0]/4*(area_idx[0]+1), image_size[1]/4*(3-area_idx[1])]
            if abs(corner[1] - self.start_pos[1]) > abs(corner[0] - self.end_pos[0]):
                self.circle_start = True
                self.radius = abs(corner[0] - self.end_pos[0])
                self.center = [corner[0], self.start_pos[1] - self.radius]
            else:
                self.circle_start = False
                self.radius = abs(corner[1] - self.start_pos[1])
                self.center = [self.end_pos[0] + self.radius, corner[1]]
        elif start_edge == 2 and end_edge == 1:
            self.start_angle, self.end_angle = self.set_angle(4)
            corner = [image_size[0]/4*(area_idx[0]+1), image_size[1]/4*(3-area_idx[1])]
            if abs(corner[0] - self.start_pos[0]) > abs(corner[1] - self.end_pos[1]):
                self.circle_start = True
                self.radius = abs(corner[1] - self.end_pos[1])
                self.center = [self.start_pos[0] + self.radius, corner[1]]
            else:
                self.circle_start = False
                self.radius = abs(corner[0] - self.start_pos[0])
                self.center = [corner[0], self.end_pos[1] - self.radius]
        elif start_edge == 2 and end_edge == 3:
            self.start_angle, self.end_angle = self.set_angle(3)
            corner = [image_size[0]/4*(area_idx[0]), image_size[1]/4*(3-area_idx[1])]
            if abs(corner[0] - self.start_pos[0]) > abs(corner[1] - self.end_pos[1]):
                self.circle_start = True
                self.radius = abs(corner[1] - self.end_pos[1])
                self.center = [self.start_pos[0] - self.radius, corner[1]]
            else:
                self.circle_start = False
                self.radius = abs(corner[0] - self.start_pos[0])
                self.center = [corner[0], self.end_pos[1] - self.radius]
        elif start_edge == 3 and end_edge == 0:
            self.start_angle, self.end_angle = self.set_angle(2)
            corner = [image_size[0]/4*(area_idx[0]), image_size[1]/4*(3-area_idx[1]+1)]
            if abs(corner[1] - self.start_pos[1]) > abs(corner[0] - self.end_pos[0]):
                self.circle_start = True
                self.radius = abs(corner[0] - self.end_pos[0])
                self.center = [corner[0], self.start_pos[1] + self.radius]
            else:
                self.circle_start = False
                self.radius = abs(corner[1] - self.start_pos[1])
                self.center = [self.end_pos[0] - self.radius, corner[1]]
        elif start_edge == 3 and end_edge == 2:
            self.start_angle, self.end_angle = self.set_angle(3)
            corner = [image_size[0]/4*(area_idx[0]), image_size[1]/4*(3-area_idx[1])]
            if abs(corner[1] - self.start_pos[1]) > abs(corner[0] - self.end_pos[0]):
                self.circle_start = True
                self.radius = abs(corner[0] - self.end_pos[0])
                self.center = [corner[0], self.start_pos[1] - self.radius]
            else:
                self.circle_start = False
                self.radius = abs(corner[1] - self.start_pos[1])
                self.center = [self.end_pos[0] - self.radius, corner[1]]
        else:
            self.circle_start = False
            self.start_angle = 0
            self.end_angle = 0
            self.is_line = True
            self.center = [0, 0]
            self.radius = 0
        self.line_edge = self.set_line_edge(self.start_pos, self.end_pos, self.center, self.circle_start, self.start_angle, start_edge, end_edge)
    def set_angle(self, quadrant):
        if quadrant == 1:
            s = 180
            e = 270
        elif quadrant == 2:
            s = 270
            e = 360
        elif quadrant == 3:
            s = 0
            e = 90
        elif quadrant == 4:
            s = 90
            e = 180
        return s, e
    def set_line_edge(self, start_pos, end_pos, center, circle_start, start_angle, start_edge, end_edge):
        reverse = False
        if (start_edge - end_edge + 4)%4 == 3:
            reverse = False
        elif (start_edge - end_edge + 4)%4 == 1:
            reverse = True
        if circle_start:
            if reverse:
                if start_angle == 270 or start_angle == 90:
                    return [center[0], end_pos[1]]
                elif start_angle == 0 or start_angle == 180:
                    return [end_pos[0], center[1]]
            else:
                if start_angle == 0 or start_angle == 180:
                    return [center[0], end_pos[1]]
                elif start_angle == 270 or start_angle == 90:
                    return [end_pos[0], center[1]]
        else:
            if reverse:
                if start_angle == 0 or start_angle == 180:
                    return [center[0], start_pos[1]]
                elif start_angle == 90 or start_angle == 270:
                    return [start_pos[0], center[1]]
            else:
                if start_angle == 0 or start_angle == 180:
                    return [start_pos[0], center[1]]
                elif start_angle == 90 or start_angle == 270:
                    return [center[0], start_pos[1]]

class MySimulator2:
    """
    ・観測のyaw方向のノイズを追加(回転させる)
    ・観測のpitch方向のノイズを追加(観測範囲を拡張してからリサイズ)
    """
    def __init__(self, obs_size=[64, 64]):
        self.image_size = [1000, 1000] # mapのサイズ
        self.obs_size = obs_size # 観測画像のサイズ defalt 64*64
        update_rate = 2.5 #5 Hz ステップの更新頻度
        max_speed = 50 # mm/s ロボットの最大速度
        self.max_length = max_speed / update_rate # mm 1ステップで進む最大距離
        self.areas = []
        self.obs_map = self.image_generator() # 観測画像を生成
        self.reward = 0
        self.is_in_area = False
        # actionの更新割引率
        self.action_discount = 0.6 # 0.2
        self.action_average = 0 # ロボットのx軸方向を0度とした時の，進行方向の重み付け平均
        self.action = 0 # self.action_averageの方向を0度とした時の進行方向の角度
        init_pos_list = []
        for area in self.areas:
            if area.is_line:
                init_pos_list.append([(area.start_pos[0]+area.end_pos[0])//2, (area.start_pos[1]+area.end_pos[1])//2])
        self.robot_pos = np.array(init_pos_list[np.random.randint(len(init_pos_list))], dtype=int)
        self.obs = None # 観測画像を保持する変数
        self.prior_pos = self.robot_pos
        self.prior_action = 0
        self.use_action_diff_reward = True # 進行方向との差ではなく，前回actioとの差を報酬として用いる。
        self.action_limit = 0.2 #0.25 # 進行方向を制限する 0.1なら+-18度。0.5以下に設定すること。

    def check_edge(self, area_idx, edge):
        # self.areasの情報と照らし合わせつつ，エッジが利用可能かどうか調べる。
        grid = np.zeros([4,4]) # 4*4のグリッドを作成
        # すでに通過済みのエリアを1にする
        grid[area_idx[0], area_idx[1]] = 1
        for area in self.areas:
            grid[area.area_idx[0], area.area_idx[1]] = 1
        # edgeの方向のエリアをスタートに設定する
        if edge == 0:
            start = [area_idx[0], area_idx[1]-1]
        elif edge == 1:
            start = [area_idx[0]+1, area_idx[1]]
        elif edge == 2:
            start = [area_idx[0], area_idx[1]+1]
        elif edge == 3:
            start = [area_idx[0]-1, area_idx[1]]
        goal = [0,3] # ゴールは固定
        rows, cols = len(grid), len(grid[0]) # gridの行数と列数
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右の移動方向
        def in_bounds(r, c):
            """
            gridの範囲内かどうかを判定する関数
            """
            return 0 <= r < rows and 0 <= c < cols
        def bfs(start, goal):
            """
            bfsでゴールに到達できるかどうかを判定する関数
            """
            queue = deque([start]) # キューを作成
            visited = set([tuple(start)]) # 訪問済みのセット
            while queue: # キューが空になるまで繰り返す
                r, c = queue.popleft() # キューの先頭を取り出す
                if [r, c] == goal: # ゴールに到達した場合はTrueを返す
                    return True
                for dr, dc in directions: # 上下左右に移動する
                    nr, nc = r + dr, c + dc # 移動先の座標
                    if in_bounds(nr, nc) and ((nr, nc) not in visited) and grid[nr][nc] == 0: # gridの範囲内で未訪問かつ通過済みでない場合
                        queue.append((nr, nc)) # キューに追加
                        visited.add((nr, nc)) # 訪問済みに追加
            return False # ゴールに到達できなかった場合はFalseを返す
        # この時点でスタートがgrid外ならFalseを返す
        if in_bounds(start[0], start[1]) and grid[start[0]][start[1]] == 0:
            return bfs(start, goal)
        else:
            return False
    
    def image_generator(self, line_thickness=20):
        """
        mapの画像を生成する関数
        """
        # スタートのエリアを設定する
        self.areas.append(AreaInfo([1,3], 3, 1, 0.5, 0.5, self.image_size))
        # スタートは固定
        start_edge = 3
        start_pos = 0.5
        area_idx = [2,3]
        while True:
            # 選んだ方向のエッジが利用可能かどうか調べる
            while True:
                end_edge = np.random.randint(4)
                if self.check_edge(area_idx, end_edge):
                    break
            end_pos = np.random.rand()
            self.areas.append(AreaInfo(area_idx, start_edge, end_edge, start_pos, end_pos, self.image_size))
            start_edge = (end_edge + 2)%4
            start_pos = end_pos
            if end_edge == 0:
                area_idx = [area_idx[0], area_idx[1]-1]
            elif end_edge == 1:
                area_idx = [area_idx[0]+1, area_idx[1]]
            elif end_edge == 2:
                area_idx = [area_idx[0], area_idx[1]+1]
            elif end_edge == 3:
                area_idx = [area_idx[0]-1, area_idx[1]]
            if area_idx[0]==0 and area_idx[1]==3:
                self.areas.append(AreaInfo(area_idx, start_edge, 1, start_pos, 0.5, self.image_size))
                # print('finish to generate area data')
                break
        image = np.ones(self.image_size)
        for area in self.areas:
            if area.is_line:
                cv2.line(image, (int(area.start_pos[0]),int(area.start_pos[1])), (int(area.end_pos[0]),int(area.end_pos[1])), 0, line_thickness)
            else:
                cv2.ellipse(image, (int(area.center[0]),int(area.center[1])), (int(area.radius),int(area.radius)), 0, area.start_angle, area.end_angle, 0, line_thickness)
                if area.circle_start:
                    cv2.line(image, (int(area.line_edge[0]),int(area.line_edge[1])), (int(area.end_pos[0]),int(area.end_pos[1])), 0, line_thickness)
                else:
                    cv2.line(image, (int(area.start_pos[0]),int(area.start_pos[1])), (int(area.line_edge[0]),int(area.line_edge[1])), 0, line_thickness)
        return image

    def calc_reward(self):
        """
        ３つ合わせて-1から1の範囲に収まるように調整すること。
        """
        reward = 0
        if self.use_action_diff_reward:
            diff = np.abs(self.prior_action - self.action)
            reward += 0.5 - diff/2
        else:
            b = 2
            reward += (0.5 - np.abs(self.action)**b) # -0.5~0.5の範囲
        # 2. 観測の黒ピクセルの平均座標と画像の中心座標の距離を一定値で割った値を報酬（罰則）とする
        if np.sum(self.obs[:,:,0] == 0) == 0:
            return -1
        else:
            black_pixels = np.where(self.obs[:,:,0] == 0) # 黒ピクセルの座標を取得
            black_center = np.array([np.mean(black_pixels[1]), np.mean(black_pixels[0])]) # 黒ピクセルの中心座標
            image_center = np.array([self.obs_size[0]//2, self.obs_size[1]//2]) # 画像の中心座標
            # 黒ピクセルの中心座標と画像の中心座標の距離を罰則とする
            reward += (0.5 - np.linalg.norm(black_center - image_center)/np.linalg.norm(image_center)) # -1~1の範囲
        # rewardを-1から1の範囲に収める
        reward = np.clip(reward, -1, 1)
        return reward

    def simulate(self, action):
        """
        シミュレーションを1ステップ進める関数
        """
        self.prior_action = self.action
        self.prior_pos = self.robot_pos
        self.action = action # actionを更新 0.1を掛けると進行方向が制限される。
        self.action_average = self.action_average + action*self.action_discount*self.action_limit # action_averageを更新
        if self.action_average > 1:
            self.action_average -= 2
        elif self.action_average < -1:
            self.action_average += 2

        # ロボットの位置を更新
        theta = (self.action_average + self.action*self.action_limit)*np.pi
        new_pos = self.robot_pos + (self.max_length * np.array([np.cos(theta), np.sin(theta)])).astype(int)
        self.is_in_area = True # 画像の範囲内にいるかどうかを示すフラグ
        if 0 <= new_pos[0] < self.image_size[0] and 0 <= new_pos[1] < self.image_size[1]:
            self.robot_pos = new_pos # new_posが画像の範囲内ならロボットの位置を更新
        else: # 画像の範囲外に出た場合はロボットの位置を更新しない
            self.is_in_area = False # 画像の範囲外に出たことを示すフラグを立てる

        # 観測を更新
        obs = self.obs_map[int(self.robot_pos[1]-self.obs_size[1]//2):int(self.robot_pos[1]+self.obs_size[1]//2), int(self.robot_pos[0]-self.obs_size[0]//2):int(self.robot_pos[0]+self.obs_size[0]//2)]
        if obs.shape != (self.obs_size[1], self.obs_size[0]):
            # obsのサイズが合わない（=マップからはみ出している）場合は白で埋める
            obs = np.ones(self.obs_size)

        # 進行方向を示す観測を追加
        obs = np.dstack([obs, np.zeros_like(obs), np.zeros_like(obs)])
        obs[:,:,1] = (self.action_average + 1)/2 # action_averageに合わせてグレーに塗りつぶす
        # 前回actionを示す観測を追加
        obs[:,:,2] = (self.prior_action + 1)/2
        self.obs = obs

        # rewardを計算
        self.reward = 0
        if self.is_in_area:
            self.reward = self.calc_reward()
        else:
            self.reward = -1
        return obs, self.reward

    def render(self):
        """
        学習の進捗確認用の画像を生成する
        """
        image = np.zeros([self.image_size[0], self.image_size[1], 3])
        image[:,:,:] = self.obs_map[:,:,None] * 255
        cv2.drawMarker(image, (375, 125), (0, 0, 0), cv2.MARKER_CROSS, 40, 10)
        cv2.circle(image, tuple(self.robot_pos), 10, (0, 255, 0), -1)
        # 観測範囲を表示
        cv2.rectangle(image, (self.robot_pos[0]-self.obs_size[0]//2, self.robot_pos[1]-self.obs_size[1]//2), (self.robot_pos[0]+self.obs_size[0]//2, self.robot_pos[1]+self.obs_size[1]//2), (0, 0, 255), 2)
        # 進行方向を表示
        cv2.arrowedLine(image, tuple(self.prior_pos), tuple(self.robot_pos), (255, 0, 0), 2)
        # 上下反転
        image = cv2.flip(image, 0)
        obs = self.obs[:,:,0]
        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_NEAREST)
        obs = np.dstack([np.zeros_like(obs), obs, obs])
        # y軸を反転
        obs = cv2.flip(obs, 0)
        image[:128, self.image_size[1]-128:self.image_size[1]] = obs*255
        # rewardを表示
        cv2.putText(image, 'reward: {:.2f}'.format(self.reward), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 座標を表示
        cv2.putText(image, 'x: {}, y: {}'.format(self.robot_pos[0], self.robot_pos[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # actionを表示
        cv2.putText(image, 'action: {:.2f}'.format(self.action), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'prior action: {:.2f}'.format(self.prior_action), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, 'motion direction: {:.2f}'.format(self.action_average), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 衝突していたら文字を表示
        if not self.is_in_area:
            cv2.putText(image, 'out of area', (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return image