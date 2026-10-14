import os
import sys
import tempfile

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import save_checkpoint
from src.evaluate import evaluate
from src.q_network import QNetwork

SETTINGS = {"episodes": 1, "epsilon": 0.05, "seed": 0, "repeat_action_probability": 0.0,
            "device": torch.device("cpu")}


def write_step_0_checkpoint(path: str) -> None:
    # Built the way train() builds step 0, so the weights equal a seed 0 init.
    torch.manual_seed(0)
    q_network, target_network = QNetwork(6), QNetwork(6)
    save_checkpoint(path, q_network=q_network, target_network=target_network,
                    optimizer=optim.Adam(q_network.parameters()), env_step=0, episode=0, gradient_updates=0)


def test_step_0_can_be_evaluated_twice() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step_00000000.pt")
        write_step_0_checkpoint(path)
        first = evaluate(path, **SETTINGS)
        second = evaluate(path, **SETTINGS)

    assert (first["rewards"], first["lengths"]) == (second["rewards"], second["lengths"]), (
        f"two plays differ: {first['rewards']} {first['lengths']} vs {second['rewards']} {second['lengths']}"
    )
    print("PASS  step 0 evaluates twice in one process and plays the same episode both times")


def test_recording_keeps_every_frame_and_changes_nothing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "step_00000000.pt")
        write_step_0_checkpoint(path)
        plain = evaluate(path, **SETTINGS)
        recorded = evaluate(path, **SETTINGS, record_video=True)

    frames = recorded["frames"]
    assert plain["frames"] is None, "frames returned without record_video"
    assert len(frames) == recorded["lengths"][0] + 1, (
        f"{len(frames)} frames for a {recorded['lengths'][0]} step episode, expected one per step plus the reset"
    )
    assert frames[0].shape == (210, 160, 3), f"frame shape {frames[0].shape}"
    assert (plain["rewards"], plain["lengths"]) == (recorded["rewards"], recorded["lengths"]), (
        "recording changed the episode"
    )
    print(f"PASS  recording keeps {len(frames)} frames for {recorded['lengths'][0]} steps and changes nothing")


if __name__ == "__main__":
    test_step_0_can_be_evaluated_twice()
    test_recording_keeps_every_frame_and_changes_nothing()
    print("\nAll tests passed!")
