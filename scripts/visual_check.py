# scripts/visual_check.py
# Check the preprocessing pipeline numerically and save frames alongside it.
# Every check prints expected next to actual and asserts, so a failure raises.

import math
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
SEED = 42

# Scaled playfield. The score strip and the bottom wall are as bright as the ball.
FIELD_TOP = 15
FIELD_BOTTOM = 78
# Raw paddle columns, scaled by 84/160 = 0.525 to give PADDLE_COLUMNS below.
RAW_PADDLE_COLUMNS = {"left (opponent)": (16, 19), "right (agent)": (140, 143)}
VERTICAL_SCALE = 0.4  # 84 / 210
# Raw playfield. Occlusion is judged here, since the scaled border blurs into paddles.
RAW_FIELD_TOP = 34
RAW_FIELD_BOTTOM = 193
# Slack on the window the raw frame predicts, for rows an over scaling resize adds.
WINDOW_MARGIN = 2
# Raw rows 24 to 33 scale to 9.6 through 13.6, so a run reaching 13 is under it.
TOP_BORDER_LAST_ROW = 13
# Grayscale values: background 87, paddles 147, ball and walls 236.
BALL_THRESHOLD = 180
PADDLE_BAND = (100, 200)
# Paddles only move vertically. Searching by shape finds the score digits instead.
PADDLE_COLUMNS = {"left (opponent)": (8, 10), "right (agent)": (73, 75)}
# Where the bottom wall's antialiasing starts, inside the paddle band.
WALL_GRADIENT_TOP = 75


def find_balls(frame: np.ndarray) -> tuple[list[tuple[float, float, int]], int]:
    # Returns ([(row, col, pixels)], edge blends excluded), in full frame
    # coordinates. A component on the crop's last row is a paddle edge, not a ball.
    crop = frame[FIELD_TOP:FIELD_BOTTOM, :]
    mask = (crop > BALL_THRESHOLD).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(mask, connectivity=8)
    found = []
    excluded = 0
    for label in range(1, component_count):
        ys, xs = np.where(labels == label)
        if ys.max() == crop.shape[0] - 1:
            excluded += 1
            continue
        found.append((ys.mean() + FIELD_TOP, xs.mean(), len(ys)))
    return found, excluded


env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")

# np.random covers ReplayBuffer.sample, the action space covers the rollout.
np.random.seed(SEED)
obs, _ = env.reset(seed=SEED)
env.action_space.seed(SEED)

# Step forward so the ball is in play
for _ in range(60):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()

cv2.imwrite(os.path.join(script_dir, "raw_frame.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

# processed is already uint8 in [0, 255], so no rescale.
prep = Preprocessor()
processed = prep.preprocess(obs)
# Same frame as processed, since the loop below reassigns obs.
raw_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
cv2.imwrite(os.path.join(script_dir, "preprocessed_frame.png"), processed)

magnified = cv2.resize(processed, (504, 504), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(script_dir, "preprocessed_magnified.png"), magnified)

# Push into a buffer so the renders below use the same path training does. At SEED
# 42 the episode terminates at step 1100 (959 at seed 7), so 1,200 clears it.
BUILD_STEPS = 1_200

initial_stacked = prep.reset(obs)
stacked = initial_stacked
buffer = ReplayBuffer(capacity=20)
boundary_stacks = []
motion_stack = None
motion_stack_step = None
for step in range(BUILD_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    if episode_over:
        # env.reset() and prep.reset() always together, or slots span two episodes.
        obs, _ = env.reset()
        next_stacked = prep.reset(obs)
        # The pushed next_state spans two episodes, masked by (1 - dones) in train.py.
        for i in range(1, next_stacked.shape[-1]):
            assert np.array_equal(next_stacked[..., 0], next_stacked[..., i]), (
                f"step {step}: reset left slot {i} holding a frame from the previous episode"
            )
        boundary_stacks.append((step, next_stacked))
    else:
        next_stacked = prep.step(obs)
        # Shift register: the 3 oldest slots must be the 3 newest of the old stack.
        assert np.array_equal(next_stacked[..., :3], stacked[..., 1:]), (
            f"step {step}: stack did not shift, slots are dropped, duplicated, or reversed"
        )
        # Motion needs 4 distinct frames with a ball, which the last stack rarely has.
        slots = [next_stacked[..., i] for i in range(next_stacked.shape[-1])]
        all_distinct = all(not np.array_equal(a, b) for a, b in zip(slots, slots[1:]))
        if motion_stack is None and all_distinct and all(find_balls(slot)[0] for slot in slots):
            motion_stack = next_stacked
            motion_stack_step = step
    buffer.push(stacked, action, float(reward), next_stacked, terminated or truncated)
    stacked = next_stacked

env.close()


def save_stack_strip(state: np.ndarray, filename: str) -> None:
    # All 4 frames side by side in order, so bad slot ordering is visible.
    strip = np.concatenate([state[..., i] for i in range(state.shape[-1])], axis=1)
    magnified_strip = cv2.resize(strip, (strip.shape[1] * 3, strip.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(script_dir, filename), magnified_strip)


save_stack_strip(stacked, "stacked_frames.png")

sampled_state, _, _, _, _ = buffer.sample(1)
save_stack_strip(sampled_state[0], "stacked_frames_from_buffer.png")

# initial_stacked, so every channel is the frame processed came from.
network_input = transform(initial_stacked, DEVICE)
first_channel = (network_input[0, 0].cpu().numpy() * 255.0).round().astype(np.uint8)
cv2.imwrite(os.path.join(script_dir, "transform_output.png"), first_channel)

# A buffer state through the shared transform, the call the training loop makes.
sampled_input = transform(sampled_state, DEVICE)
sampled_hwc = (sampled_input[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
save_stack_strip(sampled_hwc, "stacked_frames_through_transform.png")

print(f"Raw: {obs.shape}")
print(f"Processed: {processed.shape}, dtype={processed.dtype}, min={processed.min()}, max={processed.max()}")
print(f"Stacked: {stacked.shape}, dtype={stacked.dtype}")
print(f"Network input (transform output): {tuple(network_input.shape)}, dtype={network_input.dtype}, device={network_input.device}")
# Check 1: transform only divides by 255, so any nonzero diff is a real bug.
transform_diff = np.abs(first_channel.astype(int) - processed.astype(int)).max()
print(f"transform_output vs preprocessed_frame: max abs diff = {transform_diff}, expected 0")
assert transform_diff == 0, "transform() did not round trip preprocessed_frame exactly"

# Check 2: components, not loose coordinates, so a paddle edge blend is named as one.
print("Ball candidates (bright components in field):")
candidates, edge_blends = find_balls(processed)
for row, col, pixels in candidates:
    print(f"  ball: row {row:.1f}, col {col:.1f}, {pixels}px")
if edge_blends:
    print(f"  excluded {edge_blends} paddle/wall edge blend(s) touching the crop's last row")
print(f"Ball components after exclusions: {len(candidates)}, expected 1 (0 only if occluded or mid-reset)")

# Check 3: paddle row ranges, on the full frame since FIELD_TOP crops the top extreme.
print("Paddles (longest in-band run in each paddle column):")


def longest_run(column: np.ndarray) -> tuple[int, int]:
    # Returns (start, length) of the longest contiguous True run, (0, 0) if none.
    best_start = best_length = current_start = current_length = 0
    for i, value in enumerate(column):
        if value:
            if current_length == 0:
                current_start = i
            current_length += 1
            if current_length > best_length:
                best_start, best_length = current_start, current_length
        else:
            current_length = 0
    return best_start, best_length


paddle_band = (processed > PADDLE_BAND[0]) & (processed < PADDLE_BAND[1])
raw_paddle_band = (raw_gray > PADDLE_BAND[0]) & (raw_gray < PADDLE_BAND[1])
for name, (first_col, last_col) in PADDLE_COLUMNS.items():
    # The raw paddle is the resize input, so it predicts height and position.
    raw_first, raw_last = RAW_PADDLE_COLUMNS[name]
    raw_start, raw_length = longest_run(raw_paddle_band[:, raw_first:raw_last + 1].any(axis=1))
    if raw_length == 0:
        print(f"  {name}: NOT FOUND in raw cols {raw_first}-{raw_last}, nothing to predict from")
        continue

    # Windowed. A global longest run relocates to a border blend and skips the check.
    window_top = max(0, math.floor(raw_start * VERTICAL_SCALE) - WINDOW_MARGIN)
    window_bottom = min(processed.shape[0], math.ceil((raw_start + raw_length) * VERTICAL_SCALE) + WINDOW_MARGIN)
    window_start, length = longest_run(paddle_band[window_top:window_bottom, first_col:last_col + 1].any(axis=1))
    if length == 0:
        print(f"  {name}: MISSING, raw paddle is {raw_length} rows at {raw_start}-{raw_start + raw_length - 1} but scaled rows {window_top}-{window_bottom - 1} are empty  <-- MISMATCH")
        continue
    top = window_top + window_start
    bottom = top + length - 1
    scaled = raw_length * VERTICAL_SCALE
    # floor or ceil of n * scale, two values rather than a band.
    expected = {max(1, math.floor(scaled)), math.ceil(scaled)}

    raw_top = raw_start
    raw_bottom = raw_start + raw_length - 1
    if raw_top <= RAW_FIELD_TOP:
        position = "TOP extreme, under the border"
    elif raw_bottom >= RAW_FIELD_BOTTOM:
        position = "BOTTOM extreme, under the border"
    else:
        position = "fully visible"

    raw_extent = f"{raw_length} raw rows at {raw_top}-{raw_bottom}"
    if top <= TOP_BORDER_LAST_ROW or bottom >= WALL_GRADIENT_TOP:
        # Paddle plus blend, so asserting the length would test the contamination.
        print(f"  {name}: rows {top}-{bottom}, height {length}, {position}, height not asserted, scaled run merged with a border blend ({raw_extent})")
        continue
    verdict = "" if length in expected else "  <-- MISMATCH"
    print(f"  {name}: rows {top}-{bottom}, height {length} (expected {sorted(expected)} from {raw_extent}), {position}{verdict}")


# Check 4: ball position per slot, so motion is coordinates and not an impression.
def report_stack_motion(state: np.ndarray, label: str) -> None:
    print(f"Ball per slot, {label}:")
    positions = []
    for i in range(state.shape[-1]):
        found, _ = find_balls(state[..., i])
        if not found:
            positions.append(None)
            print(f"  slot {i}: no ball (occluded or mid-reset)")
            continue
        row, col, _ = max(found, key=lambda candidate: candidate[2])
        positions.append((row, col))
        print(f"  slot {i}: row {row:.1f}, col {col:.1f}")

    # Adjacent pairs only. Dropping the empties compares slot 0 against slot 3.
    deltas = [
        float(round(later[1] - earlier[1], 1))
        for earlier, later in zip(positions, positions[1:])
        if earlier is not None and later is not None
    ]
    gaps = sum(1 for p in positions if p is None)
    if not deltas:
        print(f"  cannot judge motion, no two adjacent slots both have a ball ({gaps} slot(s) empty)")
        return
    # Slot 0 is oldest, so deltas share a sign unless the ball bounced.
    sign_changes = sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0)
    coverage = f", {gaps} slot(s) empty so only {len(deltas)} of 3 pairs measured" if gaps else ""
    print(f"  column deltas oldest to newest: {deltas}, sign changes: {sign_changes} (expect 0, or 1 on a bounce){coverage}")
    if all(d == 0 for d in deltas):
        print("  WARNING: ball did not move across the stack, slots may be duplicates")


if motion_stack is None:
    print("No stack in this run had a ball in all 4 slots, motion is not measurable here.")
else:
    report_stack_motion(motion_stack, f"live stack from step {motion_stack_step}")
report_stack_motion(sampled_state[0], "stack sampled from buffer")
report_stack_motion(sampled_hwc, "sampled stack through transform")

# Check 5: the slot assert ran in the loop, so 0 here means nothing was checked.
print(f"Episode boundaries exercised: {len(boundary_stacks)} at step(s) {[step for step, _ in boundary_stacks]}, all 4 slots identical after each reset")
for step, boundary_stack in boundary_stacks:
    save_stack_strip(boundary_stack, f"stacked_frames_after_reset_step{step}.png")


def save_annotated_strip(state: np.ndarray, filename: str) -> None:
    # Ball circled and slots numbered, so image and coordinates can be cross checked.
    scale = 3
    panels = []
    for i in range(state.shape[-1]):
        panel = cv2.cvtColor(state[..., i], cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (panel.shape[1] * scale, panel.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        found, _ = find_balls(state[..., i])
        if found:
            row, col, _ = max(found, key=lambda candidate: candidate[2])
            cv2.circle(panel, (int(col * scale), int(row * scale)), 10, (0, 0, 255), 1)
        cv2.putText(panel, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel)
    cv2.imwrite(os.path.join(script_dir, filename), np.concatenate(panels, axis=1))


save_annotated_strip(motion_stack if motion_stack is not None else stacked, "stacked_frames_annotated.png")
print("Wrote stacked_frames_annotated.png: slots numbered, detected ball circled.")
