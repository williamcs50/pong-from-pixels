import math

import ale_py
import gymnasium
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.action_selector import select_action
from src.preprocess import Preprocessor
from src.q_network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.transform import transform

gymnasium.register_envs(ale_py)


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


def train(
    *,
    total_steps,
    buffer_capacity,
    warmup_steps,
    train_every,
    batch_size,
    target_sync_every_updates,
    epsilon_start,
    epsilon_end,
    epsilon_decay_steps,
    learning_rate,
    gamma,
) -> None:
    if warmup_steps < batch_size:
        raise ValueError("warmup_steps must be >= batch_size")

    if train_every <= 0:
        raise ValueError("train_every must be > 0")

    if target_sync_every_updates <= 0:
        raise ValueError("target_sync_every_updates must be > 0")

    if epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Training device: GPU ({torch.cuda.get_device_name(device)})")
    else:
        print("Training device: CPU")

    env = gymnasium.make("ALE/Pong-v5")
    n_actions = env.action_space.n
    prep = Preprocessor()

    q_network = QNetwork(n_actions).to(device)
    target_network = QNetwork(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # Only the Q network's parameters, so the target never moves between syncs.
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    obs, _ = env.reset()
    stacked = prep.reset(obs)

    gradient_updates = 0
    episodes_finished = 0
    episode_reward = 0.0

    for step in range(total_steps):
        # Linear epsilon decay
        decay_fraction = min(step / epsilon_decay_steps, 1.0)

        epsilon = epsilon_start + decay_fraction * (epsilon_end - epsilon_start)

        # Epsilon greedy action
        state_tensor = transform(stacked, device)

        with torch.no_grad():
            q_values = q_network(state_tensor)

        action = select_action(q_values, epsilon, n_actions)

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        # Preprocess / stack the next frame
        next_stacked = prep.step(next_obs)

        # Only a real game end stops bootstrapping. A time limit cut means the
        # game could have kept going, so next_stacked still has value.
        replay_buffer.push(
            stacked,
            action,
            reward,
            next_stacked,
            terminated,
        )
        stacked = next_stacked

        # Either way the episode is over, so reset
        if terminated or truncated:
            episodes_finished += 1
            end_reason = "terminated" if terminated else "truncated"
            print(f"episode {episodes_finished} ended at step {step + 1} ({end_reason}), "
                  f"reward={episode_reward:+.0f}, epsilon={epsilon:.3f}")
            episode_reward = 0.0

            obs, _ = env.reset()
            stacked = prep.reset(obs)

        # Don't train until warmup is reached
        env_step = step + 1

        if (env_step >= warmup_steps and env_step % train_every == 0):
            batch = replay_buffer.sample(batch_size)

            loss = train_step(
                q_network,
                target_network,
                optimizer,
                batch,
                gamma,
            )

            gradient_updates += 1

            # Stop the moment a NaN or inf shows up instead of training on it for hours.
            if not math.isfinite(loss):
                raise RuntimeError(f"Loss became {loss} at update {gradient_updates}, step {env_step}")

            if gradient_updates % 50 == 0:
                print(f"update {gradient_updates} at step {env_step}, loss={loss:.6f}")

            # Count target syncs in gradient updates, not env steps.
            if gradient_updates % target_sync_every_updates == 0:
                target_network.load_state_dict(
                    q_network.state_dict()
                )
                print(f"target synced at update {gradient_updates}")

    # Printed even if no episode finished, so a silent run can't pass for a clean one.
    print(f"done: {total_steps} steps, {episodes_finished} episodes finished, "
          f"{gradient_updates} updates, current episode reward so far={episode_reward:+.0f}")

    env.close()