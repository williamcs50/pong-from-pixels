import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import Preprocessor


def make_fake_frame():
    return np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)


def test_preprocess_shape():
    p = Preprocessor()
    result = p.preprocess(make_fake_frame())
    assert result.shape == (84, 84), f"Expected (84, 84), got {result.shape}"
    print("PASS  preprocess shape")


def test_preprocess_dtype():
    p = Preprocessor()
    result = p.preprocess(make_fake_frame())
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    print("PASS  preprocess dtype")


def test_reset_shape():
    p = Preprocessor()
    stacked = p.reset(make_fake_frame())
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    print("PASS  reset output shape")


def test_reset_fills_stack():
    p = Preprocessor()
    stacked = p.reset(make_fake_frame())
    for i in range(4):
        assert np.array_equal(stacked[:, :, 0], stacked[:, :, i]), "All frames should be equal after reset"
    print("PASS  reset fills stack with repeated frame")


def test_step_shape():
    p = Preprocessor()
    p.reset(make_fake_frame())
    stacked = p.step(make_fake_frame())
    assert stacked.shape == (84, 84, 4), f"Expected (84, 84, 4), got {stacked.shape}"
    print("PASS  step output shape")


def test_step_updates_stack():
    p = Preprocessor()
    p.reset(make_fake_frame())
    stacked = p.step(make_fake_frame())
    assert not np.array_equal(stacked[:, :, 0], stacked[:, :, 3]), "Frames should differ after step"
    print("PASS  step updates stack")


if __name__ == "__main__":
    test_preprocess_shape()
    test_preprocess_dtype()
    test_reset_shape()
    test_reset_fills_stack()
    test_step_shape()
    test_step_updates_stack()
    print("\nAll tests passed.")
