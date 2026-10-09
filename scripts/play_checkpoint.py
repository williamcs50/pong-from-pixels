# scripts/play_checkpoint.py
# Play episodes from one checkpoint and report the mean reward, episode by episode.
# The sweep in scripts/sweep_eval.py calls the same evaluate().

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.evaluate import EVAL_EPISODES, EVAL_EPSILON, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Play episodes from a saved checkpoint.")
    parser.add_argument("checkpoint", help="path to a step_NNNNNNNN.pt file")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=EVAL_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Playing {args.episodes} episode(s) at epsilon {args.epsilon}")

    result = evaluate(
        args.checkpoint,
        episodes=args.episodes,
        epsilon=args.epsilon,
        seed=args.seed,
        repeat_action_probability=args.repeat_action_probability,
        device=device,
    )

    print(f"load verified: {result['tensors_verified']} tensors match the checkpoint "
          f"and differ from init")
    print(f"Trained to: env_step {result['env_step']}")

    for i, (reward, steps) in enumerate(zip(result["rewards"], result["lengths"]), start=1):
        print(f"  episode {i}: reward={reward:+.0f}, {steps} steps")

    # Reward alone cannot separate "network ignored" from "network learned one
    # constant action". The distribution can.
    print(f"actions taken: {result['actions']}")
    print(f"mean reward over {result['episodes']} episode(s): {result['mean_reward']:+.2f}")

    # The bar is defined over 100 episodes, so a verdict on fewer is not one.
    if result["episodes"] >= EVAL_EPISODES:
        print(f"beats the built in opponent: {result['mean_reward'] > 0}")
    else:
        print(f"no verdict: the bar needs {EVAL_EPISODES} episodes, "
              f"this ran {result['episodes']}")
