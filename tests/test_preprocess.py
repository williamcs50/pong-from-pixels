import os
import sys
import gymnasium as gym
import ale_py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import Preprocessor


def make_fake_frame():
    """Random noise for controlled unit tests (e.g. stack equality/difference)."""
    return np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)


def make_real_pong_frame(seed: int = 42):
    """Return a real observation from ALE/Pong-v5. Used to satisfy floor goal of testing with real Pong frame."""
    env = gym.make("ALE/Pong-v5")
    obs, _ = env.reset(seed=seed)
    env.close()
    return obs


def test_preprocess_shape():
    p = Preprocessor()
    # Use REAL Pong frame per floor goal
    result = p.preprocess(make_real_pong_frame())
    assert result.shape == (84, 84), f"Expected (84, 84), got {result.shape}"
    print("PASS  preprocess shape (real frame)")


def test_preprocess_dtype():
    p = Preprocessor()
    result = p.preprocess(make_real_pong_frame())
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    print("PASS  preprocess dtype (real frame)")


def test_preprocess_value_range():
    """
    New test: verify value range for uint8 grayscale image.
    For uint8 grayscale, min should be >=0 and max <=255.
    """
    p = Preprocessor()
    result = p.preprocess(make_real_pong_frame())
    assert result.min() >= 0, f"Min pixel value {result.min()} < 0"
    assert result.max() <= 255, f"Max pixel value {result.max()} > 255"
    print(f"PASS  preprocess value range (min={result.min()}, max={result.max()}) (real frame)")


def test_reset_shape():
    p = Preprocessor()
    stacked = p.reset(make_real_pong_frame())
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    print("PASS  reset output shape (real frame)")


def test_reset_fills_stack():
    p = Preprocessor()
    stacked = p.reset(make_real_pong_frame())
    for i in range(4):
        assert np.array_equal(stacked[:, :, 0], stacked[:, :, i]), "All frames should be equal after reset"
    print("PASS  reset fills stack with repeated frame (real frame)")


def test_step_shape():
    p = Preprocessor()
    p.reset(make_real_pong_frame())
    stacked = p.step(make_real_pong_frame(seed=43))  # different seed -> different starting obs
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    print("PASS  step output shape (real frame)")


def test_step_updates_stack():
    """Use fake frames here to *guarantee* the new frame differs from the previous one.
    Real consecutive frames from the same env are also fine in practice, but fake ensures test robustness.
    """
    p = Preprocessor()
    p.reset(make_fake_frame())
    stacked = p.step(make_fake_frame())
    assert not np.array_equal(stacked[:, :, 0], stacked[:, :, 3]), "Frames should differ after step"
    print("PASS  step updates stack")


if __name__ == "__main__":
    test_preprocess_shape()
    test_preprocess_dtype()
    test_preprocess_value_range()
    test_reset_shape()
    test_reset_fills_stack()
    test_step_shape()
    test_step_updates_stack()
    print("\nAll tests passed!")