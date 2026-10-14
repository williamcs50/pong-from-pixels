import gymnasium as gym
import ale_py

env = gym.make("ALE/Pong-v5", render_mode="human", repeat_action_probability=0.0)

obs, info = env.reset()

while True:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()