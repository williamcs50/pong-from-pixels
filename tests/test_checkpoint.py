import os
import sys
import tempfile

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.q_network import QNetwork
from src.train import train_step
from tests.test_train import make_batch

N_ACTIONS = 6
COUNTERS = {"env_step": 150_000, "episode": 42, "gradient_updates": 25_000}


def make_trained() -> tuple[QNetwork, QNetwork, optim.Optimizer]:
    """Return networks and an optimizer after one real step, so Adam state exists."""
    torch.manual_seed(0)
    q_network, target_network = QNetwork(N_ACTIONS), QNetwork(N_ACTIONS)
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    train_step(q_network, target_network, optimizer, make_batch(), gamma=0.99)
    return q_network, target_network, optimizer


def fresh() -> tuple[QNetwork, QNetwork, optim.Optimizer]:
    """Return networks and an optimizer at a different init, to load a checkpoint into."""
    torch.manual_seed(1)
    q_network, target_network = QNetwork(N_ACTIONS), QNetwork(N_ACTIONS)
    return q_network, target_network, optim.Adam(q_network.parameters(), lr=1e-3)


def assert_same_state(a: dict, b: dict, what: str) -> None:
    """Assert two state dicts hold the same keys and identical tensors."""
    assert a.keys() == b.keys(), f"{what} keys differ: {a.keys() ^ b.keys()}"
    for key in a:
        assert torch.equal(a[key], b[key]), f"{what} '{key}' did not survive the round trip"


def test_round_trip_restores_everything() -> None:
    q_network, target_network, optimizer = make_trained()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step.pt")
        save_checkpoint(path, q_network=q_network, target_network=target_network,
                        optimizer=optimizer, **COUNTERS)

        q_loaded, target_loaded, optimizer_loaded = fresh()
        # A fresh init differs from the saved one, so a load that did nothing fails below.
        assert not torch.equal(q_loaded.fc2.bias, q_network.fc2.bias), "fresh network matches already"
        counters = load_checkpoint(path, q_network=q_loaded, target_network=target_loaded,
                                   optimizer=optimizer_loaded)

    assert counters == COUNTERS, f"Expected counters {COUNTERS}, got {counters}"
    assert_same_state(q_network.state_dict(), q_loaded.state_dict(), "Q weight")
    assert_same_state(target_network.state_dict(), target_loaded.state_dict(), "target weight")

    saved, loaded = optimizer.state_dict()["state"], optimizer_loaded.state_dict()["state"]
    assert saved.keys() == loaded.keys() and saved, "Adam state missing after load"
    for index in saved:
        assert_same_state(saved[index], loaded[index], f"Adam state for parameter {index}")
    print("PASS  checkpoint round trip restores weights, target, Adam state and counters")


def test_restore_rng_reproduces_draws() -> None:
    q_network, target_network, optimizer = make_trained()
    np.random.seed(5)
    torch.manual_seed(5)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step.pt")
        save_checkpoint(path, q_network=q_network, target_network=target_network,
                        optimizer=optimizer, **COUNTERS)
        expected = (np.random.random(3), torch.rand(3))

        np.random.seed(99)
        torch.manual_seed(99)
        load_checkpoint(path, q_network=QNetwork(N_ACTIONS), restore_rng=True)
        actual = (np.random.random(3), torch.rand(3))

    assert np.array_equal(expected[0], actual[0]), f"numpy draws differ: {expected[0]} vs {actual[0]}"
    assert torch.equal(expected[1], actual[1]), f"torch draws differ: {expected[1]} vs {actual[1]}"
    print("PASS  restore_rng makes numpy and torch draw the same values as after the save")


def test_without_restore_rng_draws_differ() -> None:
    # Shows the test above can fail.
    q_network, target_network, optimizer = make_trained()
    np.random.seed(5)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step.pt")
        save_checkpoint(path, q_network=q_network, target_network=target_network,
                        optimizer=optimizer, **COUNTERS)
        expected = np.random.random(3)
        np.random.seed(99)
        load_checkpoint(path, q_network=QNetwork(N_ACTIONS))
        actual = np.random.random(3)

    assert not np.array_equal(expected, actual), "draws matched with restore_rng off"
    print("PASS  without restore_rng the generators are left alone")


def test_save_leaves_no_temp_file() -> None:
    q_network, target_network, optimizer = make_trained()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step.pt")
        for _ in range(2):
            save_checkpoint(path, q_network=q_network, target_network=target_network,
                            optimizer=optimizer, **COUNTERS)
        left = sorted(os.listdir(directory))

    assert left == ["step.pt"], f"Expected only step.pt after two saves, found {left}"
    print("PASS  saving twice to one path leaves exactly one file and no .tmp")


if __name__ == "__main__":
    test_round_trip_restores_everything()
    test_restore_rng_reproduces_draws()
    test_without_restore_rng_draws_differ()
    test_save_leaves_no_temp_file()
    print("\nAll tests passed!")
