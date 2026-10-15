# scripts/random_agent.py
# Play Pong with random actions in a game window, the baseline before any training.

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human", repeat_action_probability=0.0)

env.reset()

while True:
    action = env.action_space.sample()

    _, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        env.reset()