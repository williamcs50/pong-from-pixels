import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import Preprocessor
from src.replay_buffer import ReplayBuffer
from src.transform import transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_fake_frame() -> np.ndarray:
    """Return a deterministic frame, so a failing assertion is reproducible."""
    return np.arange(210 * 160 * 3, dtype=np.uint8).reshape(210, 160, 3)


def test_transform_known_values() -> None:
    # A single HWC frame with H=2, W=2, C=4. Each pixel has a distinct value,
    # so the test proves the transpose moves each value to the right place,
    # not just that the shape is right.
    frame = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [[8, 9, 10, 11], [12, 13, 14, 15]],
        ],
        dtype=np.uint8,
    )

    result = transform(frame, DEVICE)

    assert result.shape == (1, 4, 2, 2), f"Expected (1, 4, 2, 2), got {tuple(result.shape)}"
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"

    # Expected NCHW values computed by hand:
    # expected[0, c, h, w] == frame[h, w, c] / 255.0.
    expected = np.array(
        [
            [[0, 4], [8, 12]],
            [[1, 5], [9, 13]],
            [[2, 6], [10, 14]],
            [[3, 7], [11, 15]],
        ],
        dtype=np.float32,
    ) / 255.0

    assert np.allclose(result.cpu().numpy(), expected), (
        f"Expected {expected}, got {result.cpu().numpy()}"
    )
    print("PASS  transform known values (single frame, batch axis added, values in right place)")


def test_transform_round_trip_through_buffer() -> None:
    # A stacked frame from the preprocessor, transformed directly, should match
    # the same frame after being pushed into and sampled back out of the
    # replay buffer. This exercises the actual write then read path.
    p = Preprocessor()
    stacked = p.reset(make_fake_frame())

    buffer = ReplayBuffer(capacity=1)
    buffer.push(
        current_state=stacked,
        action_taken=0,
        reward=0.0,
        next_state=stacked,
        done=False,
    )

    sampled_current, _, _, _, _ = buffer.sample(1)

    direct = transform(stacked, DEVICE)
    round_tripped = transform(sampled_current, DEVICE)

    assert torch.equal(direct, round_tripped), (
        "Transform of the original frame should exactly match transform of the "
        "same frame after a push/sample round trip through the buffer"
    )
    print("PASS  transform round trip through buffer matches direct transform")


if __name__ == "__main__":
    test_transform_known_values()
    test_transform_round_trip_through_buffer()
    print("\nAll tests passed!")
