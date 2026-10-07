import os

import numpy as np
import torch

# Enough to resume, not just replay: without the counters a resume restarts the
# epsilon schedule and the target sync. The buffer is left out, 5.26 GiB per save.


def save_checkpoint(
    path: str,
    *,
    q_network,
    target_network,
    optimizer,
    env_step: int,
    episode: int,
    gradient_updates: int,
) -> None:
    payload = {
        "q_network": q_network.state_dict(),
        "target_network": target_network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_step": env_step,
        "episode": episode,
        "gradient_updates": gradient_updates,
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # Temp name then rename, so dying mid write cannot leave a truncated file where
    # the newest checkpoint should be.
    temp_path = f"{path}.tmp"
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(
    path: str,
    *,
    q_network,
    target_network=None,
    optimizer=None,
    restore_rng: bool = False,
    map_location=None,
) -> dict:
    # weights_only=False because the payload holds RNG state, not just tensors.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    q_network.load_state_dict(payload["q_network"])

    if target_network is not None:
        target_network.load_state_dict(payload["target_network"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if restore_rng:
        np.random.set_state(payload["numpy_rng"])
        # .cpu() because map_location moves every tensor and the RNG setters
        # require a CPU ByteTensor.
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

    return {
        "env_step": payload["env_step"],
        "episode": payload["episode"],
        "gradient_updates": payload["gradient_updates"],
    }
