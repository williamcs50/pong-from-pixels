import csv
import datetime
import json
import math
import os
import time

import ale_py
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.action_selector import select_action
from src.checkpoint import save_checkpoint
from src.preprocess import Preprocessor
from src.probe import load_or_make_probe
from src.git_info import git_state
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


def probe_metrics(q_network, probe_tensor) -> dict:
    # Levels, not deltas. Drift against step 0, the previous row, or another run is
    # all derivable afterwards, and computing one of them here picks the baseline
    # for every later reader.
    with torch.no_grad():
        q_values = q_network(probe_tensor)

    best = q_values.max(dim=1).values
    spread = best - q_values.min(dim=1).values
    argmax_counts = torch.bincount(q_values.argmax(dim=1), minlength=q_values.shape[1])

    return {
        "q_max_mean": best.mean().item(),
        "q_spread_mean": spread.mean().item(),
        "q_action_means": q_values.mean(dim=0).tolist(),
        # Six counts rather than a single concentration number, so the modal action,
        # its share, and the three movement pairs are all recoverable from the row.
        "argmax_counts": argmax_counts.tolist(),
    }


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
    repeat_action_probability,
    seed,
    output_dir,
    checkpoint_every_steps,
    metrics_every_steps,
    probe_path,
    allow_dirty,
) -> None:
    if warmup_steps < batch_size:
        raise ValueError("warmup_steps must be >= batch_size")

    if not 0.0 <= repeat_action_probability <= 1.0:
        raise ValueError("repeat_action_probability must be between 0.0 and 1.0")

    if train_every <= 0:
        raise ValueError("train_every must be > 0")

    if target_sync_every_updates <= 0:
        raise ValueError("target_sync_every_updates must be > 0")

    if epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be > 0")

    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be > 0")

    if metrics_every_steps <= 0:
        raise ValueError("metrics_every_steps must be > 0")

    # Refused here rather than warned about, because a dirty tree makes git_commit
    # a lie and nothing about a finished run can fix that retroactively.
    git = git_state()

    if git["git_dirty"] and not allow_dirty:
        raise RuntimeError(
            "working tree is dirty, so the recorded commit would not describe this run:\n"
            f"{git['git_status']}\n"
            "commit the changes, or pass allow_dirty for a throwaway run"
        )

    # Checked before the buffer allocates. An existing log means this directory
    # already holds a run, and appending would interleave two in one file.
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "episodes.csv")

    if os.path.exists(log_path):
        raise FileExistsError(f"{log_path} already exists, use a new output_dir for this run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Training device: GPU ({torch.cuda.get_device_name(device)})")
    else:
        print("Training device: CPU")

    # Seeded before anything draws. np.random covers both select_action and
    # ReplayBuffer.sample. Not bit reproducible on GPU.
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gymnasium.make("ALE/Pong-v5", repeat_action_probability=repeat_action_probability)
    env.action_space.seed(seed)

    # Read back from the emulator, not the argument, so this is what is running.
    actual_sticky = env.unwrapped.ale.getFloat("repeat_action_probability")
    print(f"Seed: {seed}")
    print(f"Sticky actions: {actual_sticky}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Commit: {git['git_commit']}")

    if git["git_dirty"]:
        print("WARN  dirty tree allowed, the commit above does not describe this run")
    elif git["git_commit"] is None:
        print("WARN  git unusable, so the code version for this run is unrecorded")

    n_actions = env.action_space.n
    prep = Preprocessor()

    q_network = QNetwork(n_actions).to(device)
    target_network = QNetwork(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # The optimizer holds only the Q network, so the target changes only at a sync.
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    # Only the first reset takes the seed, so later resets continue the stream.
    obs, _ = env.reset(seed=seed)
    stacked = prep.reset(obs)

    # The output size follows the env, so a mismatch here means the two have
    # come apart. Checked on a real forward pass, not on the layer definition.
    with torch.no_grad():
        output_check = q_network(transform(stacked, device))

    assert output_check.shape[-1] == n_actions, (
        f"Network outputs {output_check.shape[-1]} actions, env has {n_actions}"
    )
    print(f"Network output size: {output_check.shape[-1]}")

    # The output directory is chosen at launch, not committed, so the commit hash
    probe, probe_sha = load_or_make_probe(probe_path, os.path.join(output_dir, "probe.npy"))
    probe_tensor = transform(probe, device)
    print(f"Probe: {len(probe)} states, sha256 {probe_sha[:12]}"
          f"{' loaded from ' + probe_path if probe_path else ', generated and saved'}")

    # alone does not say what ran where. This makes each run self describing.
    run_config = {
        "total_steps": total_steps,
        "buffer_capacity": buffer_capacity,
        "warmup_steps": warmup_steps,
        "train_every": train_every,
        "batch_size": batch_size,
        "target_sync_every_updates": target_sync_every_updates,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_steps": epsilon_decay_steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "repeat_action_probability": repeat_action_probability,
        "seed": seed,
        "output_dir": os.path.abspath(output_dir),
        "checkpoint_every_steps": checkpoint_every_steps,
        "metrics_every_steps": metrics_every_steps,
        "probe_path": probe_path,
        "probe_states": len(probe),
        "probe_sha256": probe_sha,
        "allow_dirty": allow_dirty,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        # Read back at runtime, not echoed from the arguments.
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "sticky_actions_readback": actual_sticky,
        "n_actions": int(n_actions),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with open(os.path.join(output_dir, "run_config.json"), "w") as config_file:
        json.dump(run_config, config_file, indent=2)

    def checkpoint_path(at_step: int) -> str:
        return os.path.join(checkpoint_dir, f"step_{at_step:08d}.pt")

    gradient_updates = 0
    episodes_finished = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_loss_total = 0.0
    episode_loss_count = 0
    started_at = time.perf_counter()

    # The random milestone, gone once the first gradient update lands.
    save_checkpoint(
        checkpoint_path(0),
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        env_step=0,
        episode=0,
        gradient_updates=0,
    )
    print(f"checkpoint written: {os.path.basename(checkpoint_path(0))}")

    # Every row is pushed to disk as it is written, so a crash loses nothing.
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode",
        "env_step",
        "reward",
        "episode_steps",
        "epsilon",
        "gradient_updates",
        "buffer_size",
        "mean_loss",
        "wall_clock_s",
    ])
    log_file.flush()

    # A second file on a fixed step cadence. The episode log thins out as rallies
    # lengthen, Run 1 going from a row every 909 steps to every 4,139, so anything
    # plotted against step wants even spacing.
    metrics_file = open(os.path.join(output_dir, "metrics.csv"), "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(
        ["env_step", "gradient_updates", "epsilon", "buffer_size",
         "q_max_mean", "q_spread_mean"]
        + [f"q_action_{i}_mean" for i in range(n_actions)]
        + [f"argmax_count_{i}" for i in range(n_actions)]
        + ["batch_reward_fraction", "batch_done_fraction", "batches_in_window"]
    )

    # Reward and done counts over the sampled batches since the last row, so the
    # fractions describe what the updates in that window actually saw.
    window_reward_count = 0
    window_done_count = 0
    window_batches = 0

    def write_metrics_row(at_step, epsilon_now):
        nonlocal window_reward_count, window_done_count, window_batches
        stats = probe_metrics(q_network, probe_tensor)
        sampled = window_batches * batch_size

        metrics_writer.writerow(
            [at_step, gradient_updates, f"{epsilon_now:.6f}", len(replay_buffer),
             f"{stats['q_max_mean']:.6f}", f"{stats['q_spread_mean']:.6f}"]
            + [f"{v:.6f}" for v in stats["q_action_means"]]
            + stats["argmax_counts"]
            + [f"{window_reward_count / sampled:.6f}" if sampled else "",
               f"{window_done_count / sampled:.6f}" if sampled else "",
               window_batches]
        )
        metrics_file.flush()
        window_reward_count = 0
        window_done_count = 0
        window_batches = 0

    # Before any update, for the same reason as step_00000000.pt: the untrained
    # baseline is what every later row is compared against.
    write_metrics_row(0, epsilon_start)

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
        episode_steps += 1

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

            # Blank, not 0.0, when no update ran, so a real zero loss stays distinct.
            mean_loss = f"{episode_loss_total / episode_loss_count:.6f}" if episode_loss_count else ""
            log_writer.writerow([
                episodes_finished,
                step + 1,
                f"{episode_reward:.1f}",
                episode_steps,
                f"{epsilon:.6f}",
                gradient_updates,
                len(replay_buffer),
                mean_loss,
                f"{time.perf_counter() - started_at:.1f}",
            ])
            log_file.flush()

            episode_reward = 0.0
            episode_steps = 0
            episode_loss_total = 0.0
            episode_loss_count = 0

            obs, _ = env.reset()
            stacked = prep.reset(obs)

        # Don't train until warmup is reached
        env_step = step + 1

        if (env_step >= warmup_steps and env_step % train_every == 0):
            batch = replay_buffer.sample(batch_size)

            # Counted on the batch the update actually trained on. In Pong reward is
            # zero except when a point is scored, so this is how much signal an
            # update sees, and it falls as rallies lengthen.
            window_reward_count += int(np.count_nonzero(batch[2]))
            window_done_count += int(np.count_nonzero(batch[4]))
            window_batches += 1

            loss = train_step(
                q_network,
                target_network,
                optimizer,
                batch,
                gamma,
            )

            gradient_updates += 1
            episode_loss_total += loss
            episode_loss_count += 1

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

        if env_step % metrics_every_steps == 0:
            write_metrics_row(env_step, epsilon)

        # Counted in env steps, unlike the target sync above.
        if env_step % checkpoint_every_steps == 0:
            save_checkpoint(
                checkpoint_path(env_step),
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                env_step=env_step,
                episode=episodes_finished,
                gradient_updates=gradient_updates,
            )
            print(f"checkpoint written: {os.path.basename(checkpoint_path(env_step))}")

    # Skipped when the loop's last step already triggered one.
    if total_steps % checkpoint_every_steps != 0:
        save_checkpoint(
            checkpoint_path(total_steps),
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            env_step=total_steps,
            episode=episodes_finished,
            gradient_updates=gradient_updates,
        )
        print(f"checkpoint written: {os.path.basename(checkpoint_path(total_steps))}")

    log_file.close()
    metrics_file.close()

    # Printed even if no episode finished, so a silent run can't pass for a clean one.
    print(f"done: {total_steps} steps, {episodes_finished} episodes finished, "
          f"{gradient_updates} updates, current episode reward so far={episode_reward:+.0f}")

    env.close()