# scripts/tiny_train.py
# Run the real training loop at a size that finishes in a minute or two on the
# MacBook Air. This proves the pipeline runs together without falling over, not
# that the agent learns. Only the numbers that control time and memory shrink.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import train

TINY_CONFIG = {
    "total_steps": 1_000,
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
}

if __name__ == "__main__":
    train(**TINY_CONFIG)
