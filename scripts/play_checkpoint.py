# scripts/play_checkpoint.py
# Load a checkpoint and play whole episodes with it, reporting the mean reward.
# Scaled to 100 episodes this is the evaluation the bar needs: mean above 0.

import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium
import numpy as np
import torch

from src.action_selector import select_action
from src.checkpoint import load_checkpoint
from src.preprocess import Preprocessor
from src.q_network import QNetwork
from src.transform import transform

gymnasium.register_envs(ale_py)

EVAL_EPISODES = 100  # the bar in docs/PLANS.md


def parse_args():
    parser = argparse.ArgumentParser(description="Play episodes from a saved checkpoint.")
    parser.add_argument("checkpoint", help="path to a step_NNNNNNNN.pt file")
    parser.add_argument("--episodes", type=int, default=10)
    # Not 0.0: greedy on a barely trained network can stall without serving.
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gymnasium.make("ALE/Pong-v5", repeat_action_probability=args.repeat_action_probability)
    env.action_space.seed(args.seed)
    n_actions = env.action_space.n

    # Weights only. Restoring the RNG would replay the training stream.
    q_network = QNetwork(n_actions).to(device)
    before = {key: value.clone() for key, value in q_network.state_dict().items()}
    counters = load_checkpoint(args.checkpoint, q_network=q_network, map_location=device)
    q_network.eval()

    # A fresh network plays episodes without crashing too, so a no-op load would pass
    # every check below. Comparing against another fresh init would not catch it.
    after = q_network.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before), (
        "load_checkpoint changed no weights, so nothing was loaded"
    )

    stored = torch.load(args.checkpoint, map_location=device, weights_only=False)["q_network"]
    for key, tensor in stored.items():
        assert torch.equal(after[key], tensor), f"{key} does not match the checkpoint"
    print(f"load verified: {len(stored)} tensors match the checkpoint and differ from init")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trained to: env_step {counters['env_step']}, episode {counters['episode']}, "
          f"{counters['gradient_updates']} updates")
    print(f"Playing {args.episodes} episode(s) at epsilon {args.epsilon}")

    prep = Preprocessor()
    rewards = []
    # Reward alone cannot separate "network ignored" from "network learned one
    # constant action". The distribution can.
    actions_taken = collections.Counter()

    for episode in range(1, args.episodes + 1):
        # Only the first reset takes the seed, so later episodes continue the stream.
        obs, _ = env.reset(seed=args.seed if episode == 1 else None)
        stacked = prep.reset(obs)
        total = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                q_values = q_network(transform(stacked, device))

            action = select_action(q_values, args.epsilon, n_actions)
            assert isinstance(action, int) and 0 <= action < n_actions, f"illegal action {action!r}"
            actions_taken[action] += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            stacked = prep.step(obs)
            total += reward
            steps += 1

            if terminated or truncated:
                break

        # Pong is first to 21, so anything outside this is not a real game end.
        assert -21.0 <= total <= 21.0, f"episode {episode} returned {total}, outside Pong's range"
        rewards.append(total)
        print(f"  episode {episode}: reward={total:+.0f}, {steps} steps")

    env.close()

    mean = statistics.fmean(rewards)
    print(f"actions taken: {dict(sorted(actions_taken.items()))}")
    print(f"mean reward over {len(rewards)} episode(s): {mean:+.2f}")

    # The bar is defined over 100 episodes, so a verdict on fewer is not one.
    if len(rewards) >= EVAL_EPISODES:
        print(f"beats the built in opponent: {mean > 0}")
    else:
        print(f"no verdict: the bar needs {EVAL_EPISODES} episodes, this ran {len(rewards)}")
