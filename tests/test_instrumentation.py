import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probe import load_or_make_probe, make_probe, probe_hash
from src.train import probe_metrics
from src.transform import transform
from tests.test_train import DEVICE, make_constant_network

PROBE_STATES = 7
SMALL_PROBE = {"states": 5, "every": 3}


def test_probe_metrics_known_values() -> None:
    #   Q(s) = [0.5, 1.0, -2.0, 0.0, 3.0, 0.25] for every state
    #   max 3.0, min -2.0, so spread 5.0, and every state's argmax is action 4
    values = [0.5, 1.0, -2.0, 0.0, 3.0, 0.25]
    probe = np.random.default_rng(0).integers(0, 256, size=(PROBE_STATES, 84, 84, 4), dtype=np.uint8)
    stats = probe_metrics(make_constant_network(values), transform(probe, DEVICE))

    assert math.isclose(stats["q_max_mean"], 3.0, rel_tol=1e-6), f"q_max_mean {stats['q_max_mean']}"
    assert math.isclose(stats["q_spread_mean"], 5.0, rel_tol=1e-6), f"q_spread_mean {stats['q_spread_mean']}"
    for got, want in zip(stats["q_action_means"], values):
        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), f"q_action_means {stats['q_action_means']}"
    assert stats["argmax_counts"] == [0, 0, 0, 0, PROBE_STATES, 0], f"argmax_counts {stats['argmax_counts']}"
    print(f"PASS  probe_metrics matches hand computed values on {PROBE_STATES} states")


def test_make_probe_is_reproducible() -> None:
    first, second = make_probe(**SMALL_PROBE), make_probe(**SMALL_PROBE)
    other_seed = make_probe(**SMALL_PROBE, seed=8)

    assert first.shape == (SMALL_PROBE["states"], 84, 84, 4), f"Probe shape {first.shape}"
    assert probe_hash(first) == probe_hash(second), "same seed gave two different probes"
    assert probe_hash(first) != probe_hash(other_seed), "a different seed gave the same probe"
    print("PASS  make_probe repeats exactly at one seed and changes with another")


def test_probe_hash_sees_one_pixel() -> None:
    probe = make_probe(**SMALL_PROBE)
    changed = probe.copy()
    changed[0, 0, 0, 0] ^= 1
    assert probe_hash(probe) != probe_hash(changed), "one changed pixel kept the same hash"
    print("PASS  probe_hash changes when a single pixel does")


def test_load_or_make_probe() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "earlier.npy")
        saved = np.random.default_rng(1).integers(0, 256, size=(3, 84, 84, 4), dtype=np.uint8)
        np.save(source, saved)

        out = os.path.join(directory, "probe.npy")
        loaded, sha = load_or_make_probe(source, out)
        assert np.array_equal(loaded, saved) and sha == probe_hash(saved), "loaded probe differs from file"
        assert not os.path.exists(out), "a loaded probe should not be written out again"
    print("PASS  load_or_make_probe reuses a given probe without rewriting it")


if __name__ == "__main__":
    test_probe_metrics_known_values()
    test_make_probe_is_reproducible()
    test_probe_hash_sees_one_pixel()
    test_load_or_make_probe()
    print("\nAll tests passed!")
