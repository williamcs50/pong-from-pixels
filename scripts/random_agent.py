# scripts/random_agent.py
# Play Pong with random actions in a game window, the baseline before any training.
# Like scripts/watch_checkpoint.py, it plays a set number of games, prints each
# score, and closes the window.
#
#   python scripts/random_agent.py
#   python scripts/random_agent.py --episodes 3 --seed 1

import argparse

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch random actions play Pong in a window.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error(f"--episodes must be 1 or more, got {args.episodes}")
    if args.seed < 0:
        parser.error(f"--seed must be 0 or more, got {args.seed}")
    return args


if __name__ == "__main__":
    args = parse_args()
    env = gym.make("ALE/Pong-v5", render_mode="human", repeat_action_probability=0.0)
    env.action_space.seed(args.seed)
    print(f"Watching {args.episodes} random episode(s) from seed {args.seed}")

    for episode in range(1, args.episodes + 1):
        # Only the first reset takes the seed, so later games continue the stream.
        env.reset(seed=args.seed if episode == 1 else None)
        total = 0.0
        steps = 0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            steps += 1
            if terminated or truncated:
                break
        score = f"won 21 to {21 - int(total)}" if total > 0 else f"lost 21 to {21 + int(total)}"
        print(f"  episode {episode}: reward {total:+.0f} ({score}), {steps} steps")

    env.close()
