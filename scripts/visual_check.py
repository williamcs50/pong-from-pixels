# scripts/visual_check.py
# Save preprocessed Pong frames for eyeball verification.
# Run, open the PNGs, confirm ball + paddles are visible and the stack is in order.

import os
import sys

import ale_py
import cv2
import gymnasium
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import Preprocessor
from src.replay_buffer import ReplayBuffer
from src.transform import transform

gymnasium.register_envs(ale_py)

script_dir = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")
obs, _ = env.reset(seed=42)

# Step forward so the ball is in play
for _ in range(60):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()

# Save raw frame for comparison
cv2.imwrite(os.path.join(script_dir, "raw_frame.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

# Preprocess a single frame and save. processed is already uint8 in [0, 255],
# no rescale needed, that's the bug this replaces.
prep = Preprocessor()
processed = prep.preprocess(obs)
cv2.imwrite(os.path.join(script_dir, "preprocessed_frame.png"), processed)

# 6x magnification with nearest-neighbor so pixels stay crisp
magnified = cv2.resize(processed, (504, 504), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(script_dir, "preprocessed_magnified.png"), magnified)

# Build a real stack with the ball actually moving, and push transitions into
# a buffer, so the stacked-frame render below is checked through the same
# push/sample path training will actually use, not just reset()/step() directly.
initial_stacked = prep.reset(obs)
stacked = initial_stacked
buffer = ReplayBuffer(capacity=20)
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        # A fresh episode started. env.reset() and prep.reset() must always be
        # called together, never one without the other, or the next 3 slots
        # keep frames from the episode that just ended, stitched to the new
        # episode's first frame as if they were one continuous clip.
        obs, _ = env.reset()
        next_stacked = prep.reset(obs)
    else:
        next_stacked = prep.step(obs)
    # next_stacked here is 4 copies of the new episode's first frame, not a
    # true successor of stacked, they're from different episodes. That's
    # fine: done=True is set below, and the Bellman target masks the
    # bootstrap term by (1 - done), so a terminal transition's next_state is
    # never actually read during training. If done were wrong here, it would
    # matter a great deal.
    buffer.push(stacked, 0, float(reward), next_stacked, terminated or truncated)
    stacked = next_stacked

env.close()


def save_stack_strip(state: np.ndarray, filename: str) -> None:
    # state is (84, 84, 4) uint8. Render all 4 frames side by side, in order,
    # so duplicated or reversed slots in the stack are visible at a glance.
    strip = np.concatenate([state[..., i] for i in range(state.shape[-1])], axis=1)
    magnified_strip = cv2.resize(strip, (strip.shape[1] * 3, strip.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(script_dir, filename), magnified_strip)


save_stack_strip(stacked, "stacked_frames.png")

sampled_state, _, _, _, _ = buffer.sample(1)
save_stack_strip(sampled_state[0], "stacked_frames_from_buffer.png")

# Render what the network actually sees: transform's output, scaled back up
# for viewing. Uses initial_stacked, not the final loop-mutated stacked, so
# every channel is the same frame as processed (reset() repeats the first
# frame 4 times). That's what makes this comparable to preprocessed_frame.png,
# if they don't match, the bug is in transform, not in the preprocessor.
network_input = transform(initial_stacked, DEVICE)
first_channel = (network_input[0, 0].cpu().numpy() * 255.0).round().astype(np.uint8)
cv2.imwrite(os.path.join(script_dir, "transform_output.png"), first_channel)

# The strip that actually matters: a state pulled from the buffer, run
# through the shared transform (the exact call the training loop will make),
# then undone just enough to view it, back to HWC, times 255, uint8. If this
# doesn't march the ball in a consistent direction, oldest on the left, the
# bug is in transform or in the sampling, not just in the raw stack.
sampled_input = transform(sampled_state, DEVICE)
sampled_hwc = (sampled_input[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
save_stack_strip(sampled_hwc, "stacked_frames_through_transform.png")

print(f"Raw: {obs.shape}")
print(f"Processed: {processed.shape}, dtype={processed.dtype}, min={processed.min()}, max={processed.max()}")
print(f"Stacked: {stacked.shape}, dtype={stacked.dtype}")
print(f"Network input (transform output): {tuple(network_input.shape)}, dtype={network_input.dtype}, device={network_input.device}")
print(f"Ball-candidate pixels (bright, in field):")
# Crop the score strip (top) and the bottom wall (bottom), both are bright
# and both dominate this threshold if left in, burying the one real signal.
field = processed[15:78, :]
for y, x in np.argwhere(field > 180):
    print(f"  ({y+15}, {x}) = {processed[y+15, x]}")
print("Open the PNGs: confirm ball + paddles are visible, the stacked strip shows the ball moving across 4 distinct frames in order, and transform_output matches preprocessed_frame.")
