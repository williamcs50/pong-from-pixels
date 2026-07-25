import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim


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


def test_target_network_sync() -> None:
    online_net = QNetwork(n_actions=N_ACTIONS).to(DEVICE)
    target_net = QNetwork(n_actions=N_ACTIONS).to(DEVICE)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)

    # Confirm target params aren't in the optimizer
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    online_param_ids = {id(p) for p in online_net.parameters()}
    target_param_ids = {id(p) for p in target_net.parameters()}
    assert optimizer_param_ids == online_param_ids, "Optimizer should hold exactly online_net's parameters"
    assert optimizer_param_ids.isdisjoint(target_param_ids), "Optimizer should not hold any target_net parameters"

    # Snapshot target weights before the update
    target_snapshot = {name: param.detach().clone() for name, param in target_net.named_parameters()}

    x = torch.randn(2, 4, 84, 84, device=DEVICE)
    optimizer.zero_grad()
    q_values = online_net(x)
    loss = q_values.sum()
    loss.backward()
    optimizer.step()

    # Online net took a gradient step, params should have changed
    for name, online_param in online_net.named_parameters():
        assert not torch.allclose(online_param.detach(), target_snapshot[name]), (
            f"Expected online param '{name}' to change after the optimizer step"
        )

    # Target net isn't in the optimizer, params should be untouched
    for name, target_param in target_net.named_parameters():
        assert torch.allclose(target_param.detach(), target_snapshot[name]), (
            f"Target param '{name}' changed even though it wasn't in the optimizer"
        )

    # Periodic sync should bring them back into equality
    target_net.load_state_dict(online_net.state_dict())

    for (online_name, online_param), (target_name, target_param) in zip(
        online_net.named_parameters(), target_net.named_parameters()
    ):
        assert online_name == target_name
        assert torch.allclose(online_param.detach(), target_param.detach())

    print("PASS  target network behavior")


if __name__ == "__main__":
    test_output_shape_and_device()
    test_backward_pass()
    test_target_network_sync()
    print("\nAll tests passed for QNetwork.")