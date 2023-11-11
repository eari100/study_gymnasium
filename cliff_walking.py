import gymnasium as gym


env = gym.make('CliffWalking-v0', render_mode="human")
observation, info = env.reset()

for _ in range(30):
    action = env.action_space.sample()  # 행동 랜덤
    observation, reward, terminated, truncated, info = env.step(action) # 역학 1단계 실행

    if terminated or truncated:
        observation, info = env.reset()


env.close()
