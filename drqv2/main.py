import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import torch

# import dmc
import utils
from video import VideoRecorder
from my_controller import MyController # 実機制御用のGym環境をインポート

torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = make_agent(self.eval_env.observation_spec(),
                                self.eval_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0

    def setup(self):
        self.eval_env = MyController() # 実機制御用のGym環境をインスタンス化
        self.video_recorder = VideoRecorder(Path(__file__).parent)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        time_step = self.eval_env.reset()
        self.video_recorder.init(self.eval_env, enabled=(episode == 0))
        while not time_step["done"]:
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step["observation"],
                                        self.global_step,
                                        eval_mode=True)
                print("action: ", action)
            time_step = self.eval_env.step(action)
            self.video_recorder.record(self.eval_env)
            total_reward += time_step["reward"]
            step += 1

        episode += 1
        self.video_recorder.save(f'{self.global_frame}.mp4')

    def load_snapshot(self, snapshot: Path):
        with snapshot.open('rb') as f:
            payload = torch.load(f, map_location=self.device)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workspace = Workspace(cfg)
    snapshot = Path(Path(__file__).parent, 'snapshot.pt') # このファイルと同じディレクトリにsnapshot.ptがあると仮定
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot) # 学習済みモデルをロード
    workspace.eval()

if __name__ == '__main__':
    main()