import numpy as np
import torch


def transform(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    # Convert uint8 HWC frames (single or batched) to normalized float32 NCHW on device.
    if frames.ndim == 3:
        frames = frames[np.newaxis, ...]

    normalized = frames.astype(np.float32) / 255.0
    nchw = normalized.transpose(0, 3, 1, 2)

    return torch.from_numpy(nchw).to(device)
