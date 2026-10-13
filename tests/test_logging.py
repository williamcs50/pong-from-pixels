import atexit
import csv
import functools
import json
import math
import os
import shutil
import sys
import tempfile
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import train as train_module
from src.checkpoint import load_checkpoint
from src.git_info import software_versions
from src.probe import probe_hash
from src.q_network import QNetwork
from src.train import train

PROBE_STATES = 4
CONFIG = {
    "total_steps": 2_400,
    "buffer_capacity": 1_000,
    "warmup_steps": 700,  # past three metrics rows, so rows before any update can be compared
    "train_every": 4,
    "batch_size": 8,
    "target_sync_every_updates": 50,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay_steps": 1_000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "repeat_action_probability": 0.0,
    "seed": 0,
    "checkpoint_every_steps": 1_000,
    "metrics_every_steps": 300,
    "allow_dirty": True,  # the dirty check has its own test below
}


def updates_by(step: int) -> int:
    """Gradient updates train() should have made once env_step reaches step."""
    first = CONFIG["warmup_steps"] + (-CONFIG["warmup_steps"]) % CONFIG["train_every"]
    return 0 if step < first else (step - first) // CONFIG["train_every"] + 1


def scheduled_epsilon(loop_index: int) -> float:
    fraction = min(loop_index / CONFIG["epsilon_decay_steps"], 1.0)
    return CONFIG["epsilon_start"] + fraction * (CONFIG["epsilon_end"] - CONFIG["epsilon_start"])


@functools.lru_cache(maxsize=None)
def tiny_run() -> str:
    root = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    probe_path = os.path.join(root, "probe_in.npy")
    np.save(probe_path, np.random.default_rng(0).integers(0, 256, size=(PROBE_STATES, 84, 84, 4),
                                                         dtype=np.uint8))
    output_dir = os.path.join(root, "run")
    train(**CONFIG, output_dir=output_dir, probe_path=probe_path)
    return output_dir


def read(name: str) -> list[dict]:
    with open(os.path.join(tiny_run(), name), newline="") as handle:
        return list(csv.DictReader(handle))


def test_metrics_rows_on_schedule() -> None:
    rows = read("metrics.csv")
    steps = [int(r["env_step"]) for r in rows]
    expected = list(range(0, CONFIG["total_steps"] + 1, CONFIG["metrics_every_steps"]))

    assert steps == expected, f"Expected metrics at {expected}, got {steps}"
    for row in rows:
        counts = sum(int(row[f"argmax_count_{i}"]) for i in range(6))
        assert counts == PROBE_STATES, f"argmax counts sum to {counts} at {row['env_step']}"
    print(f"PASS  {len(rows)} metrics rows on schedule, argmax counts summing to {PROBE_STATES}")


def test_metrics_before_updates_are_identical() -> None:
    before = [r for r in read("metrics.csv") if int(r["gradient_updates"]) == 0]
    after = [r for r in read("metrics.csv") if int(r["gradient_updates"]) > 0]

    assert len(before) > 1, "need at least two rows before the first update to compare"
    assert len({(r["q_max_mean"], r["q_spread_mean"]) for r in before}) == 1, (
        "Q values moved with no gradient update, so the probe or the logging is not fixed"
    )
    assert {r["q_max_mean"] for r in after} - {before[0]["q_max_mean"]}, "Q never moved after updates began"
    print(f"PASS  Q values identical across {len(before)} rows before any update, and move after")


def test_metrics_counts_match_updates() -> None:
    rows = read("metrics.csv")
    for row in rows:
        step = int(row["env_step"])
        assert int(row["gradient_updates"]) == updates_by(step), (
            f"{row['gradient_updates']} updates at {step}, expected {updates_by(step)}"
        )
        sampled = int(row["batches_in_window"])
        assert (row["batch_reward_fraction"] == "") == (sampled == 0), f"fraction blank mismatch at {step}"
    windows = sum(int(r["batches_in_window"]) for r in rows)
    assert windows == updates_by(CONFIG["total_steps"]), (
        f"batches across windows {windows}, updates {updates_by(CONFIG['total_steps'])}"
    )
    print(f"PASS  update counts and batch windows agree, {windows} batches in all")


def test_metrics_epsilon_follows_schedule() -> None:
    for row in read("metrics.csv"):
        step = int(row["env_step"])
        # Row 0 is written before the loop, later rows hold loop index env_step - 1.
        expected = CONFIG["epsilon_start"] if step == 0 else scheduled_epsilon(step - 1)
        assert row["epsilon"] == f"{expected:.6f}", f"epsilon {row['epsilon']} at {step}, expected {expected:.6f}"
    print("PASS  logged epsilon follows the configured schedule on every metrics row")


def test_episode_log_is_consistent() -> None:
    rows = read("episodes.csv")
    assert rows, "no episode finished, so nothing in the log was tested"

    previous_end = 0
    for row in rows:
        end, length = int(row["env_step"]), int(row["episode_steps"])
        assert end - previous_end == length, f"episode {row['episode']} spans {end - previous_end}, logged {length}"
        previous_end = end

        reward = float(row["reward"])
        assert reward.is_integer() and -21 <= reward <= 21, f"episode {row['episode']} reward {reward}"
        assert row["epsilon"] == f"{scheduled_epsilon(end - 1):.6f}", f"episode {row['episode']} epsilon"
        # The episode row is written before that step's update runs.
        assert int(row["gradient_updates"]) == updates_by(end - 1), f"episode {row['episode']} updates"
        if row["mean_loss"]:
            assert math.isfinite(float(row["mean_loss"])), f"episode {row['episode']} loss {row['mean_loss']}"
    print(f"PASS  {len(rows)} episodes tile the steps with matching epsilon and update counts")


def test_checkpoints_match_their_names() -> None:
    every, total = CONFIG["checkpoint_every_steps"], CONFIG["total_steps"]
    expected = sorted(set(range(0, total + 1, every)) | {total})
    directory = os.path.join(tiny_run(), "checkpoints")
    names = sorted(os.listdir(directory))

    assert names == [f"step_{s:08d}.pt" for s in expected], f"checkpoints {names}"
    for step in expected:
        counters = load_checkpoint(os.path.join(directory, f"step_{step:08d}.pt"),
                                   q_network=QNetwork(6), map_location="cpu")
        assert counters["env_step"] == step, f"step_{step:08d}.pt holds env_step {counters['env_step']}"
        assert counters["gradient_updates"] == updates_by(step), f"step {step} updates {counters}"
    print(f"PASS  checkpoints at {expected} hold matching counters")


def test_run_config_records_the_probe() -> None:
    config = json.load(open(os.path.join(tiny_run(), "run_config.json")))
    probe = np.load(config["probe_path"])

    assert config["probe_sha256"] == probe_hash(probe), "recorded probe hash does not match the probe"
    assert config["probe_states"] == PROBE_STATES, f"probe_states {config['probe_states']}"
    for key, value in CONFIG.items():
        assert config[key] == value, f"run_config {key} is {config[key]}, ran with {value}"
    for key, value in software_versions().items():
        assert config.get(key) == value, f"run_config {key} is {config.get(key)!r}, running {value!r}"
    assert config["torch_version"] == torch.__version__, f"torch_version {config['torch_version']}"
    print("PASS  run_config.json records every setting, the probe's hash and the software versions")


def test_refuses_existing_run() -> None:
    try:
        train(**CONFIG, output_dir=tiny_run(), probe_path=None)
    except FileExistsError:
        print("PASS  a directory that already holds a run is refused")
        return
    raise AssertionError("train() appended to a directory that already holds a run")


def test_refuses_dirty_tree() -> None:
    dirty = {"git_commit": "a" * 40, "git_dirty": True, "git_status": " M src/train.py"}
    with tempfile.TemporaryDirectory() as root, mock.patch.object(train_module, "git_state", return_value=dirty):
        output_dir = os.path.join(root, "run")
        try:
            train(**{**CONFIG, "allow_dirty": False}, output_dir=output_dir, probe_path=None)
        except RuntimeError:
            assert not os.path.exists(output_dir), "refused run still created its directory"
            print("PASS  a dirty tree is refused before anything is written")
            return
    raise AssertionError("train() ran on a dirty tree without allow_dirty")


if __name__ == "__main__":
    test_metrics_rows_on_schedule()
    test_metrics_before_updates_are_identical()
    test_metrics_counts_match_updates()
    test_metrics_epsilon_follows_schedule()
    test_episode_log_is_consistent()
    test_checkpoints_match_their_names()
    test_run_config_records_the_probe()
    test_refuses_existing_run()
    test_refuses_dirty_tree()
    print("\nAll tests passed!")
