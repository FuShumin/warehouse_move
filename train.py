from move_env import MoveGoodsEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def make_env():
    def _init():
        env = MoveGoodsEnv()
        return env

    return _init


# 使用GPU进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
torch.backends.cudnn.benchmark = True

env = make_vec_env(MoveGoodsEnv, n_envs=4, vec_env_cls=DummyVecEnv, monitor_dir="logs/", seed=42)


# env = SubprocVecEnv([make_env() for _ in range(n_envs)])


def learning_rate_f(progress_remaining):
    if progress_remaining > 0.5:
        return 0.0003
    else:
        return 0.0002


def learning_rate_decay(progress_remaining):
    initial_lr = 0.0003
    final_lr = 0.00008
    return final_lr + (initial_lr - final_lr) * progress_remaining


class PlottingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self):
        reward = np.mean(self.locals.get('rewards', []))
        self.rewards.append(reward)
        return True

    def plot(self):
        plt.plot(self.rewards)
        plt.xlabel('Number of Steps')
        plt.ylabel('Mean Reward')
        plt.title('Training Progress')
        plt.savefig('training_progress.png')


policy_kwargs = dict(
    net_arch=[512, 512, 512]  # 2层MLP
)

checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='movegoods_model')
plotting_callback = PlottingCallback()
model = PPO("MlpPolicy", env, verbose=1, device=device,
            learning_rate=learning_rate_decay,
            tensorboard_log="./move_goods_tensorboard/",
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=32,
            )

model.learn(total_timesteps=1000000, callback=[checkpoint_callback, plotting_callback])
model.save("move_goods_model-v4")

# # 绘制奖励曲线
# log_dir = "logs/"
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Move Goods PPO Reward")
# plt.savefig("reward_curve.png")
