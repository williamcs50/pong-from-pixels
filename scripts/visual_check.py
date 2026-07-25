# scripts/visual_check.py
"""Save a preprocessed Pong frame for eyeball verification.
Run, open the PNG, confirm ball + paddles are visible."""

import os
import sys

import ale_py
import cv2
import gymnasium
import numpy as np

gymnasium.register_envs(ale_py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import Preprocessor

env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")
obs, _ = env.reset(seed=42)

# Step forward so the ball is in play
for _ in range(60):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()

env.close()

# Save raw frame for comparison
script_dir = os.path.dirname(os.path.abspath(__file__))
cv2.imwrite(os.path.join(script_dir, "raw_frame.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

# Preprocess and save
prep = Preprocessor()
processed = prep.preprocess(obs)
save_img = (processed * 255).astype(np.uint8)
cv2.imwrite(os.path.join(script_dir, "preprocessed_frame.png"), save_img)

# 6x magnification with nearest-neighbor so pixels stay crisp
magnified = cv2.resize(save_img, (504, 504), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(script_dir, "preprocessed_magnified.png"), magnified)

print(f"Raw: {obs.shape}")
print(f"Processed: {processed.shape}, dtype={processed.dtype}")
print(f"Ball-candidate pixels (bright, in field):")
field = processed[15:, :]
for y, x in np.argwhere(field > 180):
    print(f"  ({y+15}, {x}) = {processed[y+15, x]}")
print("Open the PNGs and confirm ball + paddles are visible.")