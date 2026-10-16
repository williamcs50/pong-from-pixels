# src/transform.py
# The one conversion from stored frames to the network's input, shared by training
# and evaluation.

import numpy as np
import torch


def transform(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    # Convert uint8 HWC frames (single or batched) to normalized float32 NCHW on device.
    if frames.ndim == 3:
        frames = frames[np.newaxis, ...]

    normalized = frames.astype(np.float32) / 255.0
    nchw = normalized.transpose(0, 3, 1, 2)

    tensor = torch.from_numpy(nchw).to(device)
    # The transpose leaves the data in its original order, and .to() keeps that layout.
    # MPS has computed wrong results on layouts like this, so it gets a normal copy.
    # CUDA and the CPU keep the layout they've always had, so their results don't move.
    return tensor.contiguous() if device.type == "mps" else tensor
