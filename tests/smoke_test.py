import os
import sys

import ale_py
import gymnasium
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.action_selector import select_action
from src.preprocess import Preprocessor
from src.q_network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.transform import transform

gymnasium.register_envs(ale_py)

N_STEPS = 10
BATCH_SIZE = 8
# EPSILON = 0.0 forces every action through select_action's argmax branch,
# the one that actually reads q_values (dim=1, .item()). The random branch
# never touches q_values, so it would not exercise the forward pass
# integration this test exists to check.
EPSILON = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_acting_loop() -> tuple[ReplayBuffer, QNetwork, int]:
    """Run the acting path for N_STEPS.

    Returns the filled buffer, the network, and the action count.
    """
    env = gymnasium.make("ALE/Pong-v5", render_mode="rgb_array")
    n_actions = env.action_space.n

    prep = Preprocessor()
    buffer = ReplayBuffer(capacity=N_STEPS)
    net = QNetwork(n_actions).to(DEVICE)

    obs, _ = env.reset(seed=42)
    stacked = prep.reset(obs)

    for _ in range(N_STEPS):
        state_input = transform(stacked, DEVICE)
        with torch.no_grad():
            q_values = net(state_input)

        assert q_values.shape == (1, n_actions), f"Expected (1, {n_actions}), got {tuple(q_values.shape)}"
        assert q_values.dtype == torch.float32, f"Expected float32, got {q_values.dtype}"
        assert q_values.device.type == DEVICE.type, f"Expected q_values on {DEVICE.type}, got {q_values.device.type}"
        assert not torch.isnan(q_values).any(), "Q-values contain NaN"

        action = select_action(q_values, EPSILON, n_actions)
        assert 0 <= action < n_actions, f"Action {action} is not a valid index into {n_actions} actions"

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            next_stacked = prep.reset(obs)
        else:
            next_stacked = prep.step(obs)

        buffer.push(stacked, action, float(reward), next_stacked, terminated or truncated)
        stacked = next_stacked

    env.close()
    return buffer, net, n_actions


def test_acting_path() -> None:
    # Runs the live batch of one state through the transform, the network,
    # select_action, and env.step at every step. The per-step assertions live
    # in run_acting_loop, so this only confirms the full loop completes.
    _, _, n_actions = run_acting_loop()
    print(f"PASS  acting path: {N_STEPS} steps, valid actions, q_values (1, {n_actions}), no NaN, on {DEVICE.type}")


def test_learning_path() -> None:
    # Runs a sampled batch through the transform and the network. Loss and the
    # backward pass belong to the training loop, not this test, so this only
    # proves a batch survives the chain.
    buffer, net, n_actions = run_acting_loop()

    current_batch, _, _, _, _ = buffer.sample(BATCH_SIZE)
    batch_input = transform(current_batch, DEVICE)
    with torch.no_grad():
        batch_q_values = net(batch_input)

    assert batch_q_values.shape == (BATCH_SIZE, n_actions), f"Expected ({BATCH_SIZE}, {n_actions}), got {tuple(batch_q_values.shape)}"
    assert batch_q_values.dtype == torch.float32, f"Expected float32, got {batch_q_values.dtype}"
    assert batch_q_values.device.type == DEVICE.type, f"Expected batch_q_values on {DEVICE.type}, got {batch_q_values.device.type}"
    assert not torch.isnan(batch_q_values).any(), "Batched Q-values contain NaN"
    print(f"PASS  learning path: batch of {BATCH_SIZE} survives buffer -> transform -> network, ({BATCH_SIZE}, {n_actions}), no NaN, on {DEVICE.type}")


if __name__ == "__main__":
    test_acting_path()
    test_learning_path()
    print("\nAll tests passed!")
