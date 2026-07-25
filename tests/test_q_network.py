import os
import sys
import torch
import torch.nn.functional as F


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.q_network import QNetwork


BATCH_SIZE = 32
N_ACTIONS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch() -> torch.Tensor:
    return torch.zeros((BATCH_SIZE, 4, 84, 84), dtype=torch.float32, device=DEVICE)


def test_output_shape_and_device() -> None:
    net = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    x = make_batch()
    out = net(x)

    assert out.shape == (BATCH_SIZE, N_ACTIONS), f"Expected shape {(BATCH_SIZE, N_ACTIONS)}, got {out.shape}"
    assert out.device.type == DEVICE.type, f"Expected device '{DEVICE}', got {out.device.type}"

    sample = out[0]
    print("sample output:", sample)
    assert torch.isfinite(sample).all(), "Sample output contains non-finite values"
    assert torch.any(sample < 0), "Expected some negative Q-values at initialization"
    assert torch.abs(sample.mean()).item() < 1.0, f"Expected roughly zero-centered output, got mean {sample.mean().item():.4f}"

    print("PASS  output shape and device")

def test_backward_pass() -> None:
    net = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    x = make_batch()
    out = net(x)

    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()

    # Check that gradients are not None
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"

    print("PASS  backward pass and gradients")


if __name__ == "__main__":
    test_output_shape_and_device()
    test_backward_pass()
    print("\nAll tests passed for QNetwork.")