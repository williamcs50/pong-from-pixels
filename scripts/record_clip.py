# scripts/record_clip.py
# Record one checkpoint's clip by the episode and clip window rules in
# docs/runs/run-02.md: the first episode at evaluation seed 0, played through
# evaluate(), saved only if two plays agree, cut to its first 30 seconds.
#
#   python scripts/record_clip.py $HOME\pong-runs\run-02\checkpoints\step_05400000.pt

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

from src.evaluate import EVAL_EPSILON, evaluate
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one checkpoint's evaluation episode as a clip.")
    parser.add_argument("checkpoint", help="path to a step_NNNNNNNN.pt file")
    parser.add_argument("--out", default=None,
                        help="defaults to <run_dir>/clips/<checkpoint name>.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EVAL_EPSILON)
    return parser.parse_args()


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

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {checkpoint}")
    print(f"Playing the seed {args.seed} first episode twice at epsilon {args.epsilon}")

    result, agree = play_twice(checkpoint, args.seed, args.epsilon, device)
    if not agree and device.type != "cpu":
        print("  the two plays differ, so trying CPU as the episode rule says")
        device = torch.device("cpu")
        result, agree = play_twice(checkpoint, args.seed, args.epsilon, device)
    if not agree:
        sys.exit("the two plays still differ, so nothing is recorded until the cause is found")

    frames, q_values = result["frames"], result["q_values"]
    assert len(frames) == result["lengths"][0] + 1, (
        f"{len(frames)} frames for {result['lengths'][0]} steps, expected one per step plus the reset"
    )
    assert len(q_values) == len(frames) - 1, f"{len(q_values)} Q value rows for {len(frames)} frames"
    if len(q_values) < CLIP_FRAMES:
        sys.exit(f"the episode has {len(frames)} frames, fewer than the {CLIP_FRAMES} the clip window needs")

    write_video(frames[:CLIP_FRAMES], out)
    with open(f"{stem}_q.json", "w") as handle:
        json.dump({
            "fps": FPS,
            "frames": CLIP_FRAMES,
            "action_names": ACTION_NAMES,
            "paddle_groups": PADDLE_GROUPS,
            "q_values": [[round(v, 4) for v in row] for row in q_values[:CLIP_FRAMES]],
        }, handle)
    elapsed = time.perf_counter() - started

    record = {
        "checkpoint": checkpoint,
        "env_step": result["env_step"],
        "seed": args.seed,
        "epsilon": args.epsilon,
        "device": str(device),
        "reward": result["rewards"][0],
        "episode_steps": result["lengths"][0],
        "episode_frames": len(frames),
        "clip_frames": CLIP_FRAMES,
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
    print(f"  episode: reward {record['reward']:+.0f}, {record['episode_steps']} steps")
    print(f"  clip: the first {CLIP_FRAMES} frames, {CLIP_SECONDS} s at {FPS} fps, took {record['elapsed_s']} s")
