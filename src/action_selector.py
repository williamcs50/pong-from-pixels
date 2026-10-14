import numpy as np
import torch


def select_action(q_values: torch.Tensor, epsilon: float, n_actions: int) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))

    return torch.argmax(q_values, dim=1).item()