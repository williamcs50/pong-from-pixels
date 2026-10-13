import hashlib

import ale_py
import gymnasium
import numpy as np

from src.preprocess import Preprocessor

gymnasium.register_envs(ale_py)

# Fixed and deliberately independent of the run seed, so two runs at different
# seeds still measure Q values on the same states and their drift is comparable.
PROBE_SEED = 7
PROBE_STATES = 500
# Consecutive frames are nearly identical, so sample across the rollout rather
# than taking a contiguous block.
PROBE_EVERY = 20


def make_probe(states: int = PROBE_STATES, every: int = PROBE_EVERY,
               seed: int = PROBE_SEED) -> np.ndarray:
    # A random policy, so these are early game states: short rallies, the ball
    # rarely returned. A fixed yardstick rather than where a trained agent lives.
    env = gymnasium.make("ALE/Pong-v5", repeat_action_probability=0.0)
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)

    prep = Preprocessor()
    stacked = prep.reset(obs)
    collected = []
    step = 0

    while len(collected) < states:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())

        if terminated or truncated:
            obs, _ = env.reset()
            stacked = prep.reset(obs)
        else:
            stacked = prep.step(obs)

        step += 1

        if step % every == 0:
            collected.append(stacked.copy())

    env.close()
    return np.stack(collected)


def probe_hash(probe: np.ndarray) -> str:
    # Hashes the array rather than the file, so it does not depend on npy framing.
    return hashlib.sha256(probe.tobytes()).hexdigest()


def load_or_make_probe(path: str, out_path: str) -> tuple[np.ndarray, str]:
    # path given: reuse an earlier run's probe so the two are byte identical.
    # Otherwise build it from PROBE_SEED and save it, which gives the same states
    # anyway but records what this run used.
    probe = np.load(path) if path else make_probe()

    if not path:
        np.save(out_path, probe)

    return probe, probe_hash(probe)
