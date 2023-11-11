import gymnasium as gym
import numpy as np


env = gym.make('CliffWalking-v0', render_mode="human")
observation, info = env.reset()

# 0: 위, 1: 오른쪽, 2: 아래, 3: 왼쪽으로 이동
theta_0 = np.array([
    [np.nan, 1, 1, np.nan], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, 1, 1, 1], [np.nan, np.nan, 1, 1],
    [1, 1, 1, np.nan], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, np.nan, 1, 1],
    [1, 1, 1, np.nan], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, 1, np.nan, 1], [1, np.nan, 1, 1],
    [1, np.nan, np.nan, np.nan]
])

for _ in range(0):
    action = env.action_space.sample()  # 행동 랜덤
    # observation: 위치
    observation, reward, terminated, truncated, info = env.step(0) # 역학 1단계 실행

    if terminated or truncated:
        observation, info = env.reset()


env.close()
