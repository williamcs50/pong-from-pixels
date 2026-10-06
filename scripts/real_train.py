# scripts/real_train.py
# Run 1 on the 2060, through the same train() as scripts/tiny_train.py.
# The reasoning behind each value is in the 2026-10-06 entry of docs/sessions.md.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
}

if __name__ == "__main__":
    train(**REAL_CONFIG)
