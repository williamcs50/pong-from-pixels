import torch
import torch.nn.functional as F

from src.transform import transform


def train_step(q_network, target_network, optimizer, batch, gamma) -> float:
    states, actions, rewards, next_states, dones = batch
    device = next(q_network.parameters()).device

    states = transform(states, device)
    next_states = transform(next_states, device)

    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    predicted = (
        q_network(states)
        .gather(1, actions.unsqueeze(1))
        .squeeze(1)
    )

    with torch.no_grad():
        best_next = target_network(next_states).max(dim=1).values
        targets = rewards + gamma * (1 - dones) * best_next

    # Huber loss limits the influence of large TD errors.
    loss = F.smooth_l1_loss(predicted, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()