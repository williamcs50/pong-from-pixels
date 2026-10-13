# scripts/tiny_train.py
# Run the real training loop at a size that finishes in a minute or two on the
# MacBook Air. This proves the pipeline runs together without falling over, not
# that the agent learns. Only the numbers that control time and memory shrink.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.run_paths import RUNS_ROOT, next_run_dir
from src.train import train

TINY_CONFIG = {
    # Episode 1 ended between step 838 and 935 across five runs, so 1,000 left only
    # 65 steps of margin and a run with no episode in it would log no rows at all.
    "total_steps": 1_500,
    "buffer_capacity": 500,  # smaller than total_steps, so the buffer wraps
    "warmup_steps": 32,
    "train_every": 4,
    "batch_size": 32,
    "target_sync_every_updates": 100,  # counted in gradient updates, not env steps
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 500,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "repeat_action_probability": 0.0,  # off, matching the DQN papers
    "seed": 0,
    "checkpoint_every_steps": 250,  # small enough that the save path runs several times
    "metrics_every_steps": 200,  # small enough that the file gets several rows
    "probe_path": None,
}

# Under the shared root, but its own subtree so tiny runs do not consume the real
# run numbers.
TINY_RUNS_ROOT = os.path.join(RUNS_ROOT, "tiny")

if __name__ == "__main__":
    # Resolved here, not in TINY_CONFIG, so importing this file creates no
    # directories and claims no run number.
    # Always allowed: being run mid edit is this script's whole purpose, and its
    # output is throwaway. run_config.json still records git_dirty.
    output_dir = next_run_dir(TINY_RUNS_ROOT)
    train(**TINY_CONFIG, output_dir=output_dir, allow_dirty=True)

    # Without this, a run that ends before any episode does leaves a header and no
    # rows, and the smoke test passes without exercising the write path at all.
    with open(os.path.join(output_dir, "episodes.csv")) as log_file:
        rows = log_file.read().strip().splitlines()[1:]

    assert rows, f"no episode finished in {TINY_CONFIG['total_steps']} steps, episodes.csv has no rows"
    print(f"smoke check: episodes.csv has {len(rows)} row(s)")
