import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.action_selector import select_action

N_ACTIONS = 6
q_values = torch.tensor([[0.1, -0.2, 0.3, 2.0, 0.5, -0.1]])


def test_greedy_always_selects_argmax() -> None:
    # Epsilon = 0 should always choose the argmax
    actions = [
        select_action(q_values=q_values, epsilon=0.0, n_actions=N_ACTIONS)
        for _ in range(100)
    ]

    assert all(action == 3 for action in actions), "Expected argmax action 3 every time"
    assert all(isinstance(action, int) for action in actions), "Expected python int"
    assert all(0 <= action < N_ACTIONS for action in actions), "Action out of range"
    print("PASS  greedy branch always selects argmax action")


def test_explore_selects_all_actions_uniformly() -> None:
    actions = [
        select_action(q_values=q_values, epsilon=1.0, n_actions=N_ACTIONS)
        for _ in range(6000)
    ]

    counts = Counter(actions)

    assert set(counts.keys()) == set(range(N_ACTIONS))
    assert all(0 <= action < N_ACTIONS for action in actions)
    assert all(isinstance(action, int) for action in actions), "Expected python int"

    expected = 6000 / N_ACTIONS
    assert all(abs(count - expected) < expected * 0.25 for count in counts.values())
    print("PASS  explore branch selects all actions roughly uniformly")


def test_greedy_handles_requires_grad_tensor() -> None:
    # Selector should work with tensors that require gradients
    q_values_with_grad = q_values.clone().requires_grad_(True)
    action = select_action(q_values=q_values_with_grad, epsilon=0.0, n_actions=N_ACTIONS)

    assert isinstance(action, int), "Expected python int"
    assert action == 3, "Expected argmax action 3"
    print("PASS  greedy branch works with tensors that require gradients")


if __name__ == "__main__":
    test_greedy_always_selects_argmax()
    test_explore_selects_all_actions_uniformly()
    test_greedy_handles_requires_grad_tensor()
    print("\nAll tests passed.")
