import collections
import statistics

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

# Not 0.0: greedy on a barely trained network can stall without serving.
EVAL_EPSILON = 0.05
EVAL_EPISODES = 100  # the bar in docs/PLANS.md


def evaluate(
    checkpoint_path: str,
    *,
    episodes: int,
    epsilon: float,
    seed: int,
    repeat_action_probability: float,
    device: torch.device,
    record_video: bool = False,
    render: bool = False,
) -> dict:
    env = gymnasium.make("ALE/Pong-v5", repeat_action_probability=repeat_action_probability,
                         render_mode="human" if render else None)
    env.action_space.seed(seed)
    n_actions = env.action_space.n

    np.random.seed(seed)  # the exploration draw in select_action

    # Zeroed rather than left at a random init, which can equal step 0's saved weights
    # when torch is already seeded, for example by an earlier call to this function.
    q_network = QNetwork(n_actions).to(device)
    with torch.no_grad():
        for param in q_network.parameters():
            param.zero_()
    before = {k: v.clone() for k, v in q_network.state_dict().items()}
    counters = load_checkpoint(checkpoint_path, q_network=q_network, map_location=device)
    q_network.eval()
    torch.manual_seed(seed)

    # A network that was never loaded plays episodes without crashing too, so a no-op
    # load would otherwise pass every check below.
    after = q_network.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), (
        f"load_checkpoint changed no weights for {checkpoint_path}"
    )

    stored = torch.load(checkpoint_path, map_location=device, weights_only=False)["q_network"]
    for key, tensor in stored.items():
        assert torch.equal(after[key], tensor), f"{key} does not match {checkpoint_path}"

    prep = Preprocessor()
    rewards = []
    lengths = []
    frames: list[np.ndarray] = []
    # q_record[i] holds the Q values computed while frames[i] was on screen. The last
    # frame, after the final action, gets none.
    q_record: list[list[float]] = []
    actions_taken = collections.Counter()

    for episode in range(1, episodes + 1):
        # Only the first reset takes the seed, so later episodes continue the stream.
        obs, _ = env.reset(seed=seed if episode == 1 else None)
        if record_video and episode == 1:
            frames.append(obs)
        stacked = prep.reset(obs)
        total = 0.0
        steps = 0

        while True:
            with torch.no_grad():
                q_values = q_network(transform(stacked, device))
            if record_video and episode == 1:
                q_record.append(q_values[0].tolist())

            action = select_action(q_values, epsilon, n_actions)
            assert isinstance(action, int) and 0 <= action < n_actions, f"illegal action {action!r}"
            actions_taken[action] += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            if record_video and episode == 1:
                frames.append(obs)
            stacked = prep.step(obs)
            total += reward
            steps += 1

            if terminated or truncated:
                break

        # Pong is first to 21, so anything outside this is not a real game end.
        assert -21.0 <= total <= 21.0, f"episode {episode} returned {total}"
        rewards.append(total)
        lengths.append(steps)

    env.close()

    return {
        "env_step": counters["env_step"],
        "episodes": len(rewards),
        "epsilon": epsilon,
        "mean_reward": statistics.fmean(rewards),
        # Without the spread, a difference between two checkpoints cannot be told
        # from eval noise. Standard error is this over the root of the episode count.
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_episode_steps": statistics.fmean(lengths),
        "std_episode_steps": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "rewards": rewards,
        "lengths": lengths,
        "actions": dict(sorted(actions_taken.items())),
        "tensors_verified": len(stored),
        "frames": frames if record_video else None,
        "q_values": q_record if record_video else None,
    }
