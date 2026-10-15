# scripts/sweep_eval.py
# Evaluate a run's checkpoints, one row per checkpoint, so the evaluation curve can
# be read against the bar instead of training reward standing in for it. Rows append
# as each finishes and ones already in the file with the same settings are skipped,
# so a sweep that dies partway restarts without redoing work.
#
#   python scripts/sweep_eval.py <run_dir> --every 200000

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.evaluate import EVAL_EPISODES, EVAL_EPSILON, check_eval_args, evaluate
from src.git_info import git_state, software_versions

COLUMNS = ["env_step", "episodes", "epsilon", "seed", "mean_reward", "std_reward",
           "mean_episode_steps", "std_episode_steps", "elapsed_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every Nth checkpoint of a run.")
    parser.add_argument("run_dir", help="a run directory holding checkpoints/")
    parser.add_argument(
        "--every",
        type=int,
        default=200_000,
        help="evaluate checkpoints whose step is a multiple of this. Default 200,000, "
             "which matches the trailing mean table in docs/runs/run-01.md.",
    )
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--epsilon", type=float, default=EVAL_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--through", type=int, default=None,
                        help="evaluate only checkpoints up to and including this step")
    parser.add_argument("--out", default=None, help="defaults to <run_dir>/eval.csv")
    args = parser.parse_args()
    check_eval_args(parser, args)
    if args.every < 1:
        parser.error(f"--every must be 1 or more, got {args.every}")
    return args


def checkpoints_to_evaluate(checkpoint_dir: str, every: int) -> list[tuple[int, str]]:
    pattern = re.compile(r"^step_(\d+)\.pt$")
    found = []

    for name in sorted(os.listdir(checkpoint_dir)):
        match = pattern.match(name)
        if match and int(match.group(1)) % every == 0:
            found.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))

    return found


def record_invocation(path: str, settings: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # A list rather than one object, because the sweep resumes: a restart after a
    # code change produces rows from a different commit and both need recording.
    # Not refused on a dirty tree the way train() is, since rescoring a checkpoint
    # mid development is normal. The dirty flag says so instead.
    history = json.load(open(path)) if os.path.exists(path) else []
    history.append({**settings, **git_state(), **software_versions(),
                    "started_at": datetime.datetime.now().isoformat(timespec="seconds")})

    with open(path, "w") as handle:
        json.dump(history, handle, indent=2)


def already_done(out_path: str, *, episodes: int, epsilon: float, seed: int) -> set[int]:
    if not os.path.exists(out_path):
        return set()

    with open(out_path, newline="") as handle:
        reader = csv.DictReader(handle)

        # Appending rows of one shape under a header of another gives a file that
        # reads as valid CSV and means nothing, so refuse instead.
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"{out_path} has columns {reader.fieldnames}, this script writes "
                f"{COLUMNS}. Delete it or point --out somewhere else."
            )

        done = set()
        for row in reader:
            # Resuming matches on step alone, so a row scored with other settings
            # would count as done and the sweep would skip it without saying so.
            found = (int(row["episodes"]), float(row["epsilon"]), int(row["seed"]))
            if found != (episodes, epsilon, seed):
                raise ValueError(
                    f"{out_path} step {row['env_step']} was scored with episodes, epsilon, seed "
                    f"{found}, this sweep uses {(episodes, epsilon, seed)}. Point --out somewhere else."
                )
            done.add(int(row["env_step"]))
        return done


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join(args.run_dir, "checkpoints")
    out_path = args.out or os.path.join(args.run_dir, "eval.csv")

    targets = checkpoints_to_evaluate(checkpoint_dir, args.every)
    if args.through is not None:
        targets = [(step, path) for step, path in targets if step <= args.through]
    # Otherwise this prints "nothing to do" and exits as if the sweep had finished.
    if not targets:
        sys.exit(f"no checkpoint in {checkpoint_dir} is at a multiple of {args.every:,}"
                 + (f" up to {args.through:,}" if args.through is not None else ""))
    try:
        done = already_done(out_path, episodes=args.episodes, epsilon=args.epsilon, seed=args.seed)
    except ValueError as error:
        sys.exit(str(error))
    pending = [(step, path) for step, path in targets if step not in done]

    print(f"Device: {device}")
    print(f"Checkpoints at every {args.every:,} steps: {len(targets)}, "
          f"already evaluated: {len(done)}, to do: {len(pending)}")
    print(f"{args.episodes} episodes each at epsilon {args.epsilon}, writing {out_path}")

    if not pending:
        print("nothing to do")
        sys.exit(0)

    config_path = os.path.splitext(out_path)[0] + "_config.json"
    record_invocation(config_path, {
        "run_dir": os.path.abspath(args.run_dir),
        "every": args.every,
        "episodes": args.episodes,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "repeat_action_probability": args.repeat_action_probability,
        "through": args.through,
        "steps_to_evaluate": [step for step, _ in pending],
        "device": str(device),
    })
    print(f"commit and settings appended to {config_path}")

    # Header only on a new file, since rows are appended across restarts.
    write_header = not os.path.exists(out_path)
    
    log = open(out_path, "a", newline="")
    writer = csv.writer(log)

    if write_header:
        writer.writerow(COLUMNS)
        log.flush()

    started = time.perf_counter()

    for index, (step, path) in enumerate(pending, start=1):
        at = time.perf_counter()
        result = evaluate(
            path,
            episodes=args.episodes,
            epsilon=args.epsilon,
            seed=args.seed,
            repeat_action_probability=args.repeat_action_probability,
            device=device,
        )

        # env_step comes from inside the checkpoint, not from its filename, so a
        # renamed or mismatched file shows up rather than being taken on trust.
        assert result["env_step"] == step, (
            f"{path} is named for step {step} but holds {result['env_step']}"
        )

        elapsed = time.perf_counter() - at
        writer.writerow([
            step,
            result["episodes"],
            args.epsilon,
            args.seed,
            f"{result['mean_reward']:.2f}",
            f"{result['std_reward']:.2f}",
            f"{result['mean_episode_steps']:.0f}",
            f"{result['std_episode_steps']:.0f}",
            f"{elapsed:.0f}",
        ])
        log.flush()

        print(f"  [{index}/{len(pending)}] step {step:>9,}  "
              f"mean reward {result['mean_reward']:+6.2f} "
              f"sd {result['std_reward']:.2f}  "
              f"mean steps {result['mean_episode_steps']:>5.0f} "
              f"sd {result['std_episode_steps']:.0f}  "
              f"took {elapsed:.0f} s")

    log.close()
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min, wrote {out_path}")
