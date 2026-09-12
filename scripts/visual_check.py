# scripts/visual_check.py
# Check the preprocessing pipeline numerically and save frames alongside it.
# Every check prints expected next to actual and asserts, so a failure raises
# rather than waiting to be noticed in a PNG. Open stacked_frames_annotated.png
# to confirm the images agree with the printed coordinates.

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

# Measured constants, all derived from the raw frame and the resize factors, not
# from whatever a given run happens to show. Raw playfield is rows 34 to 194 of
# 210, scaled by 84/210 = 0.4. Raw paddles are 16 rows tall at columns 16 to 19
# and 140 to 143, scaled by 84/160 = 0.525. Background grays to 87, paddles to
# 147, ball and walls to 236.
# Cropping the score strip and the bottom wall matters for ball detection: both
# are bright and both bury the one real signal if left in.
FIELD_TOP = 15
FIELD_BOTTOM = 78
# The expected scaled paddle height is derived per run from the raw paddle on the
# same frame, not from a table. At either extreme the border is drawn over the
# paddle (measured: 236 on the cut side, background on the open side) and it
# slides under progressively, so the visible raw height itself varies and any
# fixed band either fails on legitimate frames or is too wide to fail at all.
# Raw paddle columns, the source of the scaled ones above at 0.525.
RAW_PADDLE_COLUMNS = {"left (CPU)": (16, 19), "right (player)": (140, 143)}
VERTICAL_SCALE = 0.4  # 84 / 210
# Raw playfield bounds: the border is 236 at rows 24 to 33 above and from 194
# below, so a paddle reaching these rows is sliding under it. Occlusion is judged
# here rather than on the scaled frame, where the wall's antialiasing merges with
# the paddle and a fully visible paddle can look like it is under the border.
RAW_FIELD_TOP = 34
RAW_FIELD_BOTTOM = 193
# Slack around the window the raw frame predicts, in both directions. Without it,
# rows added by an over-scaling resize fall outside the window and go unseen.
# Much larger, and a collapsed paddle can reach a border blend and be excused as
# contaminated instead of failing.
WINDOW_MARGIN = 2
# The top border blends through scaled row 13 (raw rows 24 to 33 scale to 9.6
# through 13.6), so a run reaching row 13 means the paddle is under it.
TOP_BORDER_LAST_ROW = 13
BALL_THRESHOLD = 180
PADDLE_BAND = (100, 200)  # above background, below ball and wall
# Paddles only ever move vertically, so their columns are fixed. Searching for
# them by shape instead reports the score digits as paddles, since digits are
# narrow and sit in the same brightness band.
PADDLE_COLUMNS = {"left (CPU)": (8, 10), "right (player)": (73, 75)}
# The bottom wall's antialiasing starts here, in band and indistinguishable from
# a paddle by value alone, so a run confined to it proves nothing either way.
WALL_GRADIENT_TOP = 75


def find_balls(frame: np.ndarray) -> tuple[list[tuple[float, float, int]], int]:
    # Returns ([(row, col, pixels) per real candidate], count of edge blends
    # excluded). Coordinates are in full frame space, not field crop space. A
    # component touching the crop's last row is the paddle edge blending into the
    # wall below it, which otherwise reads as a second ball.
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
# Keep the raw grayscale of the frame `processed` came from. The loop below
# reassigns obs, and the paddle check needs to measure the raw paddle on the
# same frame it measures the scaled one, or the comparison is meaningless.
raw_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
cv2.imwrite(os.path.join(script_dir, "preprocessed_frame.png"), processed)

# 6x magnification with nearest-neighbor so pixels stay crisp
magnified = cv2.resize(processed, (504, 504), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(script_dir, "preprocessed_magnified.png"), magnified)

# Build a real stack with the ball actually moving, and push transitions into
# a buffer, so the stacked-frame render below is checked through the same
# push/sample path training will actually use, not just reset()/step() directly.
#
# A real Pong episode runs well over a thousand steps, so in 20 steps the
# terminal branch below never fires even once. Forcing a boundary mid loop is
# the only way this script exercises it at all, on real frames rather than the
# fake ones the unit test uses.
FORCED_BOUNDARY_STEP = 10
# The ball does not exist on screen for the first ~15 steps after a reset, and it
# vanishes again whenever it is occluded or between points. So the loop has to run
# well past the forced boundary, and the stack used for the motion check has to be
# picked because it has a ball in all 4 slots, not because it happened to be last.
BUILD_STEPS = 60

initial_stacked = prep.reset(obs)
stacked = initial_stacked
buffer = ReplayBuffer(capacity=20)
boundary_stacks = []
motion_stack = None
motion_stack_step = None
for step in range(BUILD_STEPS):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    episode_over = terminated or truncated or step == FORCED_BOUNDARY_STEP
    if episode_over:
        # A fresh episode started. env.reset() and prep.reset() must always be
        # called together, never one without the other, or the next 3 slots
        # keep frames from the episode that just ended, stitched to the new
        # episode's first frame as if they were one continuous clip.
        obs, _ = env.reset()
        next_stacked = prep.reset(obs)
        # The transition pushed below therefore spans two episodes. Safe only
        # because done is True for it and src/train.py masks the bootstrap term
        # by (1 - dones), so that next_state is never read.
        # The catch for the bug that would not look dramatic: after a reset all
        # 4 slots must be the same frame. One stale slot means the stack is
        # spanning two episodes, and on screen that is a single slightly wrong
        # frame at the start of an episode, easy to miss and easy to excuse.
        for i in range(1, next_stacked.shape[-1]):
            assert np.array_equal(next_stacked[..., 0], next_stacked[..., i]), (
                f"step {step}: reset left slot {i} holding a frame from the previous episode"
            )
        boundary_stacks.append((step, next_stacked))
    else:
        next_stacked = prep.step(obs)
        # Stacking is a shift register: the 3 oldest slots of the new stack must
        # be the 3 newest of the old one. Catches dropped, duplicated, and
        # reversed slots without needing to look at anything.
        assert np.array_equal(next_stacked[..., :3], stacked[..., 1:]), (
            f"step {step}: stack did not shift, slots are dropped, duplicated, or reversed"
        )
        # Capture the first stack holding 4 distinct frames that each have a ball.
        # Motion is only measurable there. Taking the loop's last stack instead
        # lands in a serve delay or an occlusion, and taking an early one lands on
        # the reset's duplicated slots, which show no motion for a benign reason.
        slots = [next_stacked[..., i] for i in range(next_stacked.shape[-1])]
        all_distinct = all(not np.array_equal(a, b) for a, b in zip(slots, slots[1:]))
        if motion_stack is None and all_distinct and all(find_balls(slot)[0] for slot in slots):
            motion_stack = next_stacked
            motion_stack_step = step
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
# Check 1: transform() only divides by 255 and multiplies back, so the round
# trip is lossless for uint8. Any nonzero diff is a real bug, not rounding.
transform_diff = np.abs(first_channel.astype(int) - processed.astype(int)).max()
print(f"transform_output vs preprocessed_frame: max abs diff = {transform_diff}, expected 0")
assert transform_diff == 0, "transform() did not round trip preprocessed_frame exactly"

# Check 2: group bright pixels into components instead of printing loose
# coordinates, so a paddle edge blend is reported as what it is.
print("Ball candidates (bright components in field):")
candidates, edge_blends = find_balls(processed)
for row, col, pixels in candidates:
    print(f"  ball: row {row:.1f}, col {col:.1f}, {pixels}px")
if edge_blends:
    print(f"  excluded {edge_blends} paddle/wall edge blend(s) touching the crop's last row")
# Expect exactly 1, except when the ball is genuinely occluded by a paddle or
# absent during a post-score reset, where 0 is correct and not a failure.
print(f"Ball components after exclusions: {len(candidates)}, expected 1 (0 only if occluded or mid-reset)")

# Check 3: report each paddle's row range, so "at the top extreme" is a number
# rather than an impression. A height outside UNOCCLUDED_PADDLE_HEIGHT while the
# paddle is fully visible means the resize is distorting. Measured on the full
# frame, not the field crop: at the top extreme the paddle sits at rows 13 to 15,
# so cropping at FIELD_TOP discards all but one row of exactly the position that
# most needs checking.
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
    # Measure the paddle in the raw frame first. It is the input to the resize, so
    # it says both how tall the scaled paddle should be and where it should be.
    raw_first, raw_last = RAW_PADDLE_COLUMNS[name]
    raw_start, raw_length = longest_run(raw_paddle_band[:, raw_first:raw_last + 1].any(axis=1))
    if raw_length == 0:
        print(f"  {name}: NOT FOUND in raw cols {raw_first}-{raw_last}, nothing to predict from")
        continue

    # Measure the scaled frame only inside the predicted window. Taking the
    # column's global longest run instead lets a collapsed paddle escape: the
    # longest run relocates to a border blend and the height check gets skipped
    # as contaminated, which is how a 1-row paddle passed before.
    window_top = max(0, math.floor(raw_start * VERTICAL_SCALE) - WINDOW_MARGIN)
    window_bottom = min(processed.shape[0], math.ceil((raw_start + raw_length) * VERTICAL_SCALE) + WINDOW_MARGIN)
    window_start, length = longest_run(paddle_band[window_top:window_bottom, first_col:last_col + 1].any(axis=1))
    if length == 0:
        print(f"  {name}: MISSING, raw paddle is {raw_length} rows at {raw_start}-{raw_start + raw_length - 1} but scaled rows {window_top}-{window_bottom - 1} are empty  <-- MISMATCH")
        continue
    top = window_top + window_start
    bottom = top + length - 1
    scaled = raw_length * VERTICAL_SCALE
    # A block of n raw rows lands on floor(n * scale) or ceil(n * scale) scaled
    # rows depending on where it falls on the pixel grid. Two values, not a band:
    # widening this until it passed is what made the previous version untestable.
    expected = {max(1, math.floor(scaled)), math.ceil(scaled)}

    # Occlusion is judged on the raw frame. On the scaled frame the wall's
    # antialiasing merges with the paddle, so a paddle sitting well inside the
    # playfield can look like it is under the border.
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
        # Both borders blend into the background across the full width, and both
        # blends land inside the paddle band: scaled row 13 at the top, rows 75
        # and up at the bottom. A run touching either is paddle plus blend, so
        # asserting on its length would test the contamination, not the resize.
        print(f"  {name}: rows {top}-{bottom}, height {length}, {position}, height not asserted, scaled run merged with a border blend ({raw_extent})")
        continue
    verdict = "" if length in expected else "  <-- MISMATCH"
    print(f"  {name}: rows {top}-{bottom}, height {length} (expected {sorted(expected)} from {raw_extent}), {position}{verdict}")


# Check 4: the motion check that used to be eyeball only. Track the ball through
# all 4 slots of a stack and print its position per slot, so "the ball moves
# across 4 distinct frames, oldest on the left" is a list of coordinates.
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

    # Adjacent pairs only. Dropping the empty slots and differencing what remains
    # would compare slot 0 against slot 3 as though they were one frame apart and
    # print the three frame gap as a single plausible delta.
    deltas = [
        float(round(later[1] - earlier[1], 1))
        for earlier, later in zip(positions, positions[1:])
        if earlier is not None and later is not None
    ]
    gaps = sum(1 for p in positions if p is None)
    if not deltas:
        print(f"  cannot judge motion, no two adjacent slots both have a ball ({gaps} slot(s) empty)")
        return
    # Oldest slot is index 0, so consecutive column deltas should share a sign.
    # Mixed signs mean either the ball bounced inside these 4 frames or the slots
    # are out of order, and a single bounce is the only benign explanation.
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

# Check 5: the forced episode boundary. Every stack returned by prep.reset() was
# already asserted to hold 4 copies of one frame while the loop ran, on real
# frames rather than the unit test's fake ones.
print(f"Episode boundaries exercised: {len(boundary_stacks)} at step(s) {[step for step, _ in boundary_stacks]}, all 4 slots identical after each reset")
for step, boundary_stack in boundary_stacks:
    save_stack_strip(boundary_stack, f"stacked_frames_after_reset_step{step}.png")


def save_annotated_strip(state: np.ndarray, filename: str) -> None:
    # The strip with the detected ball circled and slots numbered, so the image
    # and the numbers above can be checked against each other instead of the
    # image being trusted on its own.
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
