import math
import os
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.q_network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.train import train_step

BATCH_SIZE = 4
N_ACTIONS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    batch = (
        rng.integers(0, 256, size=(BATCH_SIZE, 84, 84, 4), dtype=np.uint8),
        rng.integers(0, N_ACTIONS, size=(BATCH_SIZE,), dtype=np.int64),
        np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32),
        rng.integers(0, 256, size=(BATCH_SIZE, 84, 84, 4), dtype=np.uint8),
        np.array([False, True, False, True], dtype=np.bool_),
    )

    # Checked against a real buffer so the fixture format cannot drift from it.
    reference = ReplayBuffer(capacity=1)
    for field, array in zip(batch, (reference.current_state, reference.action_taken,
                                    reference.reward, reference.next_state, reference.done)):
        assert field.dtype == array.dtype, f"Expected {array.dtype}, got {field.dtype}"
        assert field.shape == (BATCH_SIZE,) + array.shape[1:], (
            f"Expected {(BATCH_SIZE,) + array.shape[1:]}, got {field.shape}"
        )
    return batch


def make_networks() -> tuple[QNetwork, QNetwork, optim.Optimizer]:
    torch.manual_seed(0)
    q_network = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    target_network = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    return q_network, target_network, optim.Adam(q_network.parameters(), lr=1e-3)


def make_constant_network(action_values: list[float]) -> QNetwork:
    # Zeroing every parameter makes activations zero through the whole stack, so
    # the output is fc2.bias for any input and the forward pass is hand computable.
    network = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    with torch.no_grad():
        for param in network.parameters():
            param.zero_()
        network.fc2.bias.copy_(torch.tensor(action_values, device=DEVICE))
    return network


def test_train_step_updates_q_not_target() -> None:
    q_network, target_network, optimizer = make_networks()

    snapshot = {name: param.detach().clone() for name, param in q_network.named_parameters()}
    target_snapshot = {name: param.detach().clone() for name, param in target_network.named_parameters()}

    loss = train_step(q_network, target_network, optimizer, make_batch(), gamma=0.99)

    assert isinstance(loss, float), f"Expected float, got {type(loss).__name__}"
    assert math.isfinite(loss), f"Loss is not finite: {loss}"

    for name, param in q_network.named_parameters():
        assert not torch.equal(param.detach(), snapshot[name]), (
            f"Expected Q param '{name}' to change after the optimizer step"
        )

    for name, param in target_network.named_parameters():
        assert torch.equal(param.detach(), target_snapshot[name]), (
            f"Target param '{name}' changed, it should be frozen"
        )
    print(f"PASS  train step moves every Q param and leaves target frozen (loss={loss:.4f})")


def test_train_step_known_values() -> None:
    #   Q(s) = [0.5, 1.0, ...]        target Q(s') = [2.0, 0.0, ...], max 2.0
    #   actions [0, 1]   rewards [1.0, -1.0]   dones [0, 1]   gamma 0.5
    #   predicted = [0.5, 1.0]
    #   targets   = [1.0 + 0.5*1*2.0, -1.0 + 0.5*0*2.0] = [2.0, -1.0]
    #   errors    = [1.5, 2.0], both past the Huber corner at 1, so |e| - 0.5
    #   losses    = [1.0, 1.5]  ->  mean 1.25
    q_network = make_constant_network([0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    target_network = make_constant_network([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # lr 0 so the step cannot perturb the weights the expectation comes from.
    optimizer = optim.SGD(q_network.parameters(), lr=0.0)

    states = np.zeros((2, 84, 84, 4), dtype=np.uint8)
    batch = (
        states,
        np.array([0, 1], dtype=np.int64),
        np.array([1.0, -1.0], dtype=np.float32),
        states,
        np.array([False, True], dtype=np.bool_),
    )

    loss = train_step(q_network, target_network, optimizer, batch, gamma=0.5)

    assert math.isclose(loss, 1.25, rel_tol=1e-6), f"Expected loss 1.25, got {loss}"
    print(f"PASS  train step matches hand-computed loss (expected 1.25, got {loss:.6f})")


def test_terminal_batch_ignores_next_state() -> None:
    # Two different next_states must give the same loss when every done is True.
    rewards = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
    states = np.random.default_rng(1).integers(0, 256, size=(BATCH_SIZE, 84, 84, 4), dtype=np.uint8)
    actions = np.zeros(BATCH_SIZE, dtype=np.int64)
    all_done = np.ones(BATCH_SIZE, dtype=np.bool_)

    losses = []
    for next_state_seed in (2, 3):
        q_network, target_network, optimizer = make_networks()
        next_states = np.random.default_rng(next_state_seed).integers(
            0, 256, size=(BATCH_SIZE, 84, 84, 4), dtype=np.uint8
        )
        batch = (states, actions, rewards, next_states, all_done)
        losses.append(train_step(q_network, target_network, optimizer, batch, gamma=0.99))

    assert losses[0] == losses[1], (
        f"Terminal transitions should ignore next_state, but two different "
        f"next_states gave losses {losses[0]} and {losses[1]}"
    )
    print(f"PASS  terminal batch ignores next_state (loss={losses[0]:.4f} for both)")


if __name__ == "__main__":
    test_train_step_updates_q_not_target()
    test_train_step_known_values()
    test_terminal_batch_ignores_next_state()
    print("\nAll tests passed!")
