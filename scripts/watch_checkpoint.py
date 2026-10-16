# scripts/watch_checkpoint.py
# Watch a checkpoint play Pong live in a game window, like scripts/random_agent.py
# but with the trained network choosing the actions. It plays through evaluate(), so
# the window shows the same episodes the sweep scored at the same seed and epsilon.
#
#   python scripts/watch_checkpoint.py
#   python scripts/watch_checkpoint.py ~/pong-runs/run-02/checkpoints/step_00600000.pt

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.evaluate import EVAL_EPSILON, check_eval_args, evaluate

# Run 2's best evaluated checkpoint, the Trained clip in docs/runs/run-02.md.
TRAINED = os.path.join(os.path.expanduser("~"), "pong-runs", "run-02", "checkpoints", "step_05400000.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a saved checkpoint play Pong in a window.")
    parser.add_argument("checkpoint", nargs="?", type=os.path.expanduser, default=TRAINED,
                        help="path to a step_NNNNNNNN.pt file, defaults to Run 2's step 5,400,000")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=EVAL_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    check_eval_args(parser, args)
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Watching {args.episodes} episode(s) from seed {args.seed} at epsilon {args.epsilon}")

    result = evaluate(
        args.checkpoint,
        episodes=args.episodes,
        epsilon=args.epsilon,
        seed=args.seed,
        repeat_action_probability=0.0,
        device=device,
        render=True,
    )

    print(f"Trained to: env_step {result['env_step']}")
    for i, (reward, steps) in enumerate(zip(result["rewards"], result["lengths"]), start=1):
        # Pong ends when either side reaches 21, so the reward gives the other side's score.
        score = f"won 21 to {21 - int(reward)}" if reward > 0 else f"lost 21 to {21 + int(reward)}"
        print(f"  episode {i}: reward {reward:+.0f} ({score}), {steps} steps")
