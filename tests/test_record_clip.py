import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate import evaluate
from tests.test_evaluate import SETTINGS, write_step_0_checkpoint

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "record_clip.py")

spec = importlib.util.spec_from_file_location("record_clip", SCRIPT)
record_clip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_clip)

BALL_ROW = 100


def frames_with_ball(columns: list[int | None]) -> list[np.ndarray]:
    """Return blank raw frames with a 2 by 4 ball at each column, or no ball for None."""
    frames = []
    for column in columns:
        frame = np.zeros((210, 160, 3), np.uint8)
        if column is not None:
            frame[BALL_ROW:BALL_ROW + 4, column:column + 2] = record_clip.BALL_SHADE
        frames.append(frame)
    return frames


def test_return_turning_at_the_paddle() -> None:
    frames = frames_with_ball([100, 115, 130, 137, 128, 110])
    assert record_clip.return_frames(frames) == [3], record_clip.return_frames(frames)
    print("PASS  a ball that turns back beside the agent's paddle counts as one return")


def test_no_return_when_the_ball_passes_or_turns_elsewhere() -> None:
    missed = frames_with_ball([120, 135, 150, None, None])
    far_side = frames_with_ball([20, 10, 25, 40])
    mid_field = frames_with_ball([60, 80, 70, 50])
    for name, frames in (("missed", missed), ("far side", far_side), ("mid field", mid_field)):
        assert record_clip.return_frames(frames) == [], f"{name}: {record_clip.return_frames(frames)}"
    print("PASS  a missed ball, the opponent's hit, and a mid field turn are not returns")


def test_rally_start_needs_three_returns_in_one_window() -> None:
    window = record_clip.CLIP_FRAMES
    assert record_clip.rally_start([10, 100, 200]) == 10
    assert record_clip.rally_start([10, 100, 10 + window]) is None
    assert record_clip.rally_start([10, 500, 600, 700]) == 500
    assert record_clip.rally_start([10, 100]) is None
    print("PASS  rally_start finds the first three returns inside one clip window")


def test_untrained_network_never_returns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step_00000000.pt")
        write_step_0_checkpoint(path)
        frames = evaluate(path, **SETTINGS, record_video=True)["frames"]
    seen = sum(record_clip.ball_x(frame) is not None for frame in frames)
    assert seen > 0, "the ball was never found, so the detector checked nothing"
    assert record_clip.return_frames(frames) == [], record_clip.return_frames(frames)
    print(f"PASS  the untrained network returns nothing, with the ball found in {seen} frames")


def record(*args: str) -> subprocess.CompletedProcess:
    """Run record_clip.py with these arguments and return the finished process."""
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


def test_bad_flags_stop_before_playing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step_00000000.pt")
        write_step_0_checkpoint(path)
        out = os.path.join(directory, "clip.mp4")
        cases = {
            "--out must end in .mp4": ("--out", os.path.join(directory, "clip")),
            "--epsilon": ("--out", out, "--epsilon", "1.5"),
            "--seed": ("--out", out, "--seed", "-1"),
        }
        results = {message: record(path, *flags) for message, flags in cases.items()}
        open(out, "w").close()
        existing = record(path, "--out", out)
        left = os.listdir(directory)

    for message, result in results.items():
        assert result.returncode != 0 and message in result.stderr, f"{message}: {result.stderr}"
    assert existing.returncode != 0 and "--overwrite" in existing.stderr, existing.stderr
    assert sorted(left) == ["clip.mp4", "step_00000000.pt"], f"files left behind: {left}"
    print("PASS  a bad --out, --epsilon or --seed, or an existing clip, stops before anything is written")


if __name__ == "__main__":
    test_return_turning_at_the_paddle()
    test_no_return_when_the_ball_passes_or_turns_elsewhere()
    test_rally_start_needs_three_returns_in_one_window()
    test_untrained_network_never_returns()
    test_bad_flags_stop_before_playing()
    print("\nAll tests passed!")
