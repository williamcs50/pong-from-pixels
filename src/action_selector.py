import torch
import numpy as np


def select_action(q_values: torch.Tensor, epsilon: float, n_actions: int) -> int:
    # Explore with probability epsilon
    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))

    # Otherwise choose the action with the highest Q-value
    return torch.argmax(q_values, dim=1).item()