# scripts/record_clip.py
# Record one checkpoint's clip by the episode and clip window rules in
# docs/runs/run-02.md: the first episode at evaluation seed 0, played through
# evaluate(), saved only if two plays agree, cut to its last 30 seconds. With --rally,
# the first rally clip rule instead: the first seed whose episode has a rally, cut
# to start just before it.
#
#   python scripts/record_clip.py ~/pong-runs/run-02/checkpoints/step_05400000.pt
#   python scripts/record_clip.py ~/pong-runs/run-02/checkpoints/step_00600000.pt --rally

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from src.evaluate import EVAL_EPSILON, check_eval_args, evaluate
from src.git_info import git_state, software_versions

FRAMESKIP = 4  # Pong v5 repeats each action for 4 emulator frames
EMULATOR_FPS = 60
FPS = EMULATOR_FPS // FRAMESKIP  # one recorded frame per step, so this plays at real speed
SCALE = 3
CLIP_SECONDS = 30  # the clip window rule in docs/runs/run-02.md
CLIP_FRAMES = CLIP_SECONDS * FPS

# Pong's six actions, and what each does to the paddle as measured by Check 6 in
# scripts/visual_check.py. The page draws its three bars from these groups.
ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")
PADDLE_GROUPS = {"up": (2, 4), "down": (3, 5), "still": (0, 1)}

# From scripts/visual_check.py: the ball is this gray in every channel, the field lies
# between these raw rows, and the agent's paddle starts at this raw column.
BALL_SHADE = 236
RAW_FIELD_TOP, RAW_FIELD_BOTTOM = 34, 193
AGENT_PADDLE_LEFT = 140
RETURN_ZONE = AGENT_PADDLE_LEFT - 10  # a ball turning back past here turned off the paddle

# The first rally clip rule in docs/runs/run-02.md.
RALLY_RETURNS = 3
RALLY_LEAD_FRAMES = 2 * FPS
MAX_RALLY_SEEDS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one checkpoint's evaluation episode as a clip.")
    parser.add_argument("checkpoint", type=os.path.expanduser, help="path to a step_NNNNNNNN.pt file")
    parser.add_argument("--out", type=os.path.expanduser, default=None,
                        help="defaults to <run_dir>/clips/<checkpoint name>.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EVAL_EPSILON)
    parser.add_argument("--rally", action="store_true",
                        help="apply the first rally clip rule: search seeds for a rally and cut around it")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a clip already at --out instead of refusing")
    args = parser.parse_args()
    # Otherwise ffmpeg rejects it only after both plays.
    if args.out is not None and not args.out.lower().endswith(".mp4"):
        parser.error(f"--out must end in .mp4, got {args.out}")
    check_eval_args(parser, args)
    return args


def ball_x(frame: np.ndarray) -> float | None:
    field = frame[RAW_FIELD_TOP:RAW_FIELD_BOTTOM + 1]
    columns = np.where((field == BALL_SHADE).all(axis=2))[1]
    return float(columns.mean()) if len(columns) else None


def return_frames(frames: list[np.ndarray]) -> list[int]:
    # The frames where the ball, moving toward the agent, turns back beside its paddle.
    xs = [ball_x(frame) for frame in frames]
    return [i for i in range(1, len(xs) - 1)
            if None not in xs[i - 1:i + 2] and xs[i - 1] < xs[i] > xs[i + 1] and xs[i] > RETURN_ZONE]


def rally_start(returns: list[int]) -> int | None:
    # The first return that begins RALLY_RETURNS returns within one clip window.
    for i in range(len(returns) - RALLY_RETURNS + 1):
        if returns[i + RALLY_RETURNS - 1] - returns[i] < CLIP_FRAMES:
            return returns[i]
    return None


def play(checkpoint: str, seed: int, epsilon: float, device: torch.device) -> dict:
    return evaluate(checkpoint, episodes=1, epsilon=epsilon, seed=seed,
                    repeat_action_probability=0.0, device=device, record_video=True)


def play_twice(checkpoint: str, seed: int, epsilon: float, device: torch.device) -> tuple[dict, bool]:
    first = play(checkpoint, seed, epsilon, device)
    second = play(checkpoint, seed, epsilon, device)
    agree = (first["rewards"], first["lengths"]) == (second["rewards"], second["lengths"])
    print(f"  {device}: reward {first['rewards'][0]:+.0f} and {second['rewards'][0]:+.0f}, "
          f"{first['lengths'][0]} and {second['lengths'][0]} steps, {'agree' if agree else 'DIFFER'}")
    return first, agree


def write_video(frames: list[np.ndarray], path: str) -> None:
    height, width = frames[0].shape[:2]
    # H.264 in yuv420p plays in every browser. faststart puts the index at the front,
    # so a page can start playing before the whole file arrives.
    writer = imageio_ffmpeg.write_frames(
        path, (width * SCALE, height * SCALE), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    for frame in frames:
        # Nearest keeps the pixels sharp instead of smearing the paddles.
        writer.send(np.ascontiguousarray(
            cv2.resize(frame, (width * SCALE, height * SCALE), interpolation=cv2.INTER_NEAREST)))
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint))
    name = os.path.splitext(os.path.basename(checkpoint))[0]
    out = args.out or os.path.join(run_dir, "clips", f"{name}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    stem = os.path.splitext(out)[0]
    # A rerun with other settings would otherwise replace a clip already checked.
    existing = [path for path in (out, f"{stem}_q.json", f"{stem}.json") if os.path.exists(path)]
    if existing and not args.overwrite:
        sys.exit(f"{', '.join(existing)} already exist(s). Pass --overwrite to replace, or choose another --out.")

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {checkpoint}")

    seed = args.seed
    if args.rally:
        for seed in range(args.seed, args.seed + MAX_RALLY_SEEDS):
            hits = return_frames(play(checkpoint, seed, args.epsilon, device)["frames"])
            print(f"  seed {seed}: {len(hits)} returns, "
                  f"{'a rally' if rally_start(hits) is not None else 'no rally'}")
            if rally_start(hits) is not None:
                break
        else:
            sys.exit(f"no episode in {MAX_RALLY_SEEDS} seeds has {RALLY_RETURNS} returns within "
                     f"{CLIP_SECONDS} seconds")
    print(f"Playing the seed {seed} first episode twice at epsilon {args.epsilon}")

    result, agree = play_twice(checkpoint, seed, args.epsilon, device)
    if not agree and device.type != "cpu":
        print("  the two plays differ, so trying CPU as the episode rule says")
        device = torch.device("cpu")
        result, agree = play_twice(checkpoint, seed, args.epsilon, device)
    if not agree:
        sys.exit("the two plays still differ, so nothing is recorded until the cause is found")

    frames, q_values = result["frames"], result["q_values"]
    assert len(frames) == result["lengths"][0] + 1, (
        f"{len(frames)} frames for {result['lengths'][0]} steps, expected one per step plus the reset"
    )
    assert len(q_values) == len(frames) - 1, f"{len(q_values)} Q value rows for {len(frames)} frames"
    if len(frames) < CLIP_FRAMES:
        sys.exit(f"the episode has {len(frames)} frames, fewer than the {CLIP_FRAMES} the clip window needs")

    hits = return_frames(frames)
    if args.rally:
        start = rally_start(hits)
        assert start is not None, "the replayed episode lost the rally the search found"
        first = min(max(0, start - RALLY_LEAD_FRAMES), len(frames) - CLIP_FRAMES)
    else:
        first = len(frames) - CLIP_FRAMES
    last = first + CLIP_FRAMES
    # The final frame has no Q values, since no action follows it, so it repeats the last row.
    clip_q = (q_values + [q_values[-1]])[first:last]
    assert len(clip_q) == CLIP_FRAMES, f"{len(clip_q)} Q value rows for a {CLIP_FRAMES} frame clip"
    clip_returns = [h - first for h in hits if first <= h < last]

    write_video(frames[first:last], out)
    with open(f"{stem}_q.json", "w") as handle:
        json.dump({
            "fps": FPS,
            "frames": CLIP_FRAMES,
            "first_episode_frame": first,
            "last_row_repeats_previous": True,
            "action_names": ACTION_NAMES,
            "paddle_groups": PADDLE_GROUPS,
            "q_values": [[round(v, 4) for v in row] for row in clip_q],
        }, handle)
    elapsed = time.perf_counter() - started

    record = {
        "checkpoint": checkpoint,
        "env_step": result["env_step"],
        "seed": seed,
        "window": "rally" if args.rally else "last",
        "epsilon": args.epsilon,
        "device": str(device),
        "reward": result["rewards"][0],
        "episode_steps": result["lengths"][0],
        "episode_frames": len(frames),
        "clip_frames": CLIP_FRAMES,
        "clip_first_episode_frame": first,
        "episode_returns": len(hits),
        "clip_return_frames": clip_returns,
        "fps": FPS,
        "scale": SCALE,
        "clip_s": CLIP_SECONDS,
        "plays_agreed": agree,
        "elapsed_s": round(elapsed, 1),
        **git_state(),
        **software_versions(),
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(f"{stem}.json", "w") as handle:
        json.dump(record, handle, indent=2)

    print(f"Saved {out}, {stem}_q.json, and {stem}.json")
    print(f"  episode: seed {seed}, reward {record['reward']:+.0f}, {record['episode_steps']} steps, "
          f"{len(hits)} returns")
    print(f"  clip: {CLIP_FRAMES} frames from frame {first:,}, {CLIP_SECONDS} s at {FPS} fps, "
          f"{len(clip_returns)} returns in it, took {record['elapsed_s']} s")
