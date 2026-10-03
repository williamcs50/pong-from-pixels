import os
import sys

import ale_py
import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import Preprocessor


def make_fake_frame() -> np.ndarray:
    """Return random noise for unit tests that compare frames in the stack."""
    return np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)


def make_real_pong_frame(seed: int = 42) -> np.ndarray:
    """Return a real ALE/Pong-v5 frame, so tests run on actual game pixels."""
    env = gym.make("ALE/Pong-v5")
    obs, _ = env.reset(seed=seed)
    env.close()
    return obs


def test_preprocess_shape() -> None:
    # Use a real Pong frame so the resize is checked on actual game pixels.
    p = Preprocessor()
    result = p.preprocess(make_real_pong_frame())
    assert result.shape == (84, 84), f"Expected (84, 84), got {result.shape}"
    print("PASS  preprocess shape (real frame)")


def test_preprocess_dtype() -> None:
    p = Preprocessor()
    result = p.preprocess(make_real_pong_frame())
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    print("PASS  preprocess dtype (real frame)")


def test_preprocess_value_range() -> None:
    # Verify raw uint8 grayscale values stay within the valid pixel range.
    p = Preprocessor()
    result = p.preprocess(make_real_pong_frame())
    assert result.min() >= 0, f"Min pixel value {result.min()} < 0"
    assert result.max() <= 255, f"Max pixel value {result.max()} > 255"
    print(f"PASS  preprocess value range (min={result.min()}, max={result.max()}) (real frame)")


def test_reset_shape() -> None:
    p = Preprocessor()
    stacked = p.reset(make_real_pong_frame())
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    assert stacked.dtype == np.uint8, f"Expected uint8, got {stacked.dtype}"
    print("PASS  reset output shape (real frame)")


def test_reset_fills_stack() -> None:
    p = Preprocessor()
    stacked = p.reset(make_real_pong_frame())
    for i in range(4):
        assert np.array_equal(stacked[..., 0], stacked[..., i]), "All frames should be equal after reset"
    print("PASS  reset fills stack with repeated frame (real frame)")


def test_step_shape() -> None:
    p = Preprocessor()
    p.reset(make_real_pong_frame())
    # Different seed so the second frame differs from the first.
    stacked = p.step(make_real_pong_frame(seed=43))
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    assert stacked.dtype == np.uint8, f"Expected uint8, got {stacked.dtype}"
    print("PASS  step output shape (real frame)")


def test_step_updates_stack() -> None:
    # Use fake frames here to guarantee the new frame differs from the previous
    # one. Real consecutive frames from the same env are also fine in practice,
    # but fake ensures test robustness.
    p = Preprocessor()
    p.reset(make_fake_frame())
    stacked = p.step(make_fake_frame())
    assert not np.array_equal(stacked[..., 0], stacked[..., 3]), "Frames should differ after step"
    print("PASS  step updates stack")


def test_newest_frame_at_last_index() -> None:
    # The eyeball check can't prove ordering, this can: recompute the newest
    # frame independently with preprocess() and compare it against the exact
    # slot step() is supposed to have placed it in.
    p = Preprocessor()
    p.reset(make_fake_frame())
    newest_obs = make_fake_frame()
    stacked = p.step(newest_obs)
    expected = p.preprocess(newest_obs)
    assert np.array_equal(stacked[..., 3], expected), "Newest frame should be at index 3"
    assert not np.array_equal(stacked[..., 0], expected), "Index 0 should still hold the oldest frame, not the newest"
    print("PASS  newest frame lands at index 3, not index 0")


def test_reset_clears_stale_stack() -> None:
    # Simulates an episode boundary: a dirty stack built from step() calls,
    # then reset() with a new episode's first frame. All 4 slots should be
    # that new frame, not a mix of the old episode and the new one, which is
    # what calling step() across the boundary would produce instead.
    p = Preprocessor()
    p.reset(make_fake_frame())
    for _ in range(3):
        p.step(make_fake_frame())

    new_episode_obs = make_fake_frame()
    stacked = p.reset(new_episode_obs)
    expected = p.preprocess(new_episode_obs)
    for i in range(4):
        assert np.array_equal(stacked[..., i], expected), f"Index {i} should be the new episode's frame, stack was not cleared"
    print("PASS  reset clears a stale stack across an episode boundary")


if __name__ == "__main__":
    test_preprocess_shape()
    test_preprocess_dtype()
    test_preprocess_value_range()
    test_reset_shape()
    test_reset_fills_stack()
    test_step_shape()
    test_step_updates_stack()
    test_newest_frame_at_last_index()
    test_reset_clears_stale_stack()
    print("\nAll tests passed!")
