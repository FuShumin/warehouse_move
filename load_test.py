from move_env import MoveGoodsEnv
from stable_baselines3 import PPO

env = MoveGoodsEnv()
obs = env.reset()
model = PPO.load("move_goods_model-v1.zip")

for i in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)

    if reward == -1:
        # 重新进行动作
        continue

    if done:
        obs = env.reset()  # 如果环境结束了，重置环境
        break  # 或者继续下一次循环
