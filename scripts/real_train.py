# scripts/real_train.py
# Run 1 on the 2060, through the same train() as scripts/tiny_train.py.
# The reasoning behind each value is in the 2026-10-06 entry of docs/sessions.md.
#
# The warmup pass before Run 1 is this same script with a shorter step count, so
# the rate it measures comes from the code path Run 1 actually uses:
#   python scripts/real_train.py --total-steps 70000     warmup pass
#   python scripts/real_train.py                         Run 1

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.run_paths import RUNS_ROOT, next_run_dir
from src.train import train

REAL_CONFIG = {
    "total_steps": 2_000_000,
    "buffer_capacity": 100_000,  # 5.26 GiB of host RAM
    "warmup_steps": 50_000,
    "train_every": 4,
    "batch_size": 32,
    "target_sync_every_updates": 10_000,  # counted in gradient updates, not env steps
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 250_000,  # counts from step 0, so epsilon is 0.80 at the first update
    "learning_rate": 1e-4,  # Adam, not Nature's RMSProp, so not the paper's rate
    "gamma": 0.99,
    "repeat_action_probability": 0.0,  # v5 defaults this to 0.25
    "seed": 0,  # not bit reproducible on GPU, cuDNN picks nondeterministic kernels
    "checkpoint_every_steps": 50_000,  # 40 checkpoints at 27 MB, about 1.1 GB
    "metrics_every_steps": 10_000,  # 600 rows over a 6M step run
    "probe_path": None,  # generated from PROBE_SEED and saved into the run directory
}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a real run, or a shorter warmup pass of the same code path.")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=REAL_CONFIG["total_steps"],
        help="env steps to run. Shorten it for a warmup pass, leave it for Run 1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"defaults to the next run-NN under {RUNS_ROOT}",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted changes. Never pass this for a real launch: the "
             "recorded commit would not describe what ran.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Assembled here rather than in REAL_CONFIG so importing this file creates no
    # directories and claims no run number.
    config = dict(REAL_CONFIG)
    config["total_steps"] = args.total_steps
    config["output_dir"] = args.output_dir or next_run_dir(RUNS_ROOT)
    config["allow_dirty"] = args.allow_dirty

    train(**config)
