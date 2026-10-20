# src/device.py
# The one place every script and test picks where the network runs: an NVIDIA GPU,
# then an Apple Silicon GPU, then the CPU.

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str | None:
    # Recorded in run_config.json, so a run says what it trained on.
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return None
