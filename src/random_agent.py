import gymnasium as gym
import ale_py

# Create the Pong environment
env = gym.make('ALE/Pong-v5', render_mode="human")

# Reset the environment
obs, info = env.reset() 

while True:
    # Pick a random action
    action = env.action_space.sample()

    # Step the environment with that action
    obs, reward, terminated, truncated, info = env.step(action)

    # If game is over, reset the environment
    if terminated or truncated:
        obs, info = env.reset()