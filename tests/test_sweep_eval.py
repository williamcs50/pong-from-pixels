import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import save_checkpoint
from src.q_network import QNetwork

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "sweep_eval.py")

spec = importlib.util.spec_from_file_location("sweep_eval", SCRIPT)
sweep_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_eval)


def write_checkpoint(directory: str, name_step: int, held_step: int | None = None) -> None:
    """Save a fresh checkpoint named for name_step, holding held_step as its counter if given."""
    q_network, target_network = QNetwork(6), QNetwork(6)
    save_checkpoint(os.path.join(directory, f"step_{name_step:08d}.pt"), q_network=q_network,
                    target_network=target_network, optimizer=optim.Adam(q_network.parameters()),
                    env_step=name_step if held_step is None else held_step, episode=0, gradient_updates=0)


def run_sweep(run_dir: str, *extra: str) -> subprocess.CompletedProcess:
    """Run sweep_eval.py at one episode per checkpoint and return the finished process."""
    # Off the NVIDIA GPU, so the test can't compete with a sweep already on it. On an
    # Apple Silicon Mac it still runs on the Mac's own GPU, which nothing else is using.
    # "-1" rather than "": an empty value leaves torch reporting CUDA with no device.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    return subprocess.run([sys.executable, SCRIPT, run_dir, "--episodes", "1", *extra],
                          cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def read_rows(path: str) -> list[dict]:
    """Return the rows of a sweep's CSV file."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_checkpoint_selection() -> None:
    with tempfile.TemporaryDirectory() as directory:
        for name in ("step_00000000.pt", "step_00050000.pt", "step_00200000.pt", "step_00400000.pt",
                     "step_00200000.pt.tmp", "notes.txt"):
            open(os.path.join(directory, name), "w").close()
        steps = [step for step, _ in sweep_eval.checkpoints_to_evaluate(directory, 200_000)]

    assert steps == [0, 200_000, 400_000], f"Expected [0, 200000, 400000], got {steps}"
    print("PASS  --every keeps multiples only, and skips temp files and other names")


SETTINGS = {"episodes": 100, "epsilon": 0.05, "seed": 0}


def write_rows(path: str, rows: list[tuple[int, int, float, int]]) -> None:
    """Write a sweep CSV whose rows hold these env_step, episodes, epsilon and seed values."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(sweep_eval.COLUMNS)
        for row in rows:
            writer.writerow(list(row) + [""] * (len(sweep_eval.COLUMNS) - len(row)))


def refuses(path: str, **settings: float) -> bool:
    """Return whether already_done refuses the file at path for these settings."""
    try:
        sweep_eval.already_done(path, **settings)
    except ValueError:
        return True
    return False


def test_already_done_refuses_old_columns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        current = os.path.join(directory, "current.csv")
        write_rows(current, [(0, 100, 0.05, 0), (200_000, 100, 0.05, 0)])
        assert sweep_eval.already_done(current, **SETTINGS) == {0, 200_000}, "already_done missed rows"
        assert sweep_eval.already_done(os.path.join(directory, "missing.csv"), **SETTINGS) == set()

        old = os.path.join(directory, "old.csv")
        with open(old, "w", newline="") as handle:
            csv.writer(handle).writerow(sweep_eval.COLUMNS[:-2] + ["elapsed_s"])
        assert refuses(old, **SETTINGS), "already_done accepted a file with different columns"
    print("PASS  already_done reads finished steps and refuses a file with other columns")


def test_already_done_refuses_other_settings() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "eval.csv")
        write_rows(path, [(0, 100, 0.05, 0), (200_000, 100, 0.05, 0)])
        for name, value in (("seed", 1), ("epsilon", 0.1), ("episodes", 10)):
            assert refuses(path, **{**SETTINGS, name: value}), f"a different {name} counted as done"
        assert not refuses(path, **SETTINGS), "matching settings were refused"
    print("PASS  rows scored with another seed, epsilon or episode count are refused, not skipped")


def test_record_invocation_appends() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "eval_config.json")
        sweep_eval.record_invocation(path, {"seed": 0})
        sweep_eval.record_invocation(path, {"seed": 1})
        history = json.load(open(path))

    assert [entry["seed"] for entry in history] == [0, 1], f"history {history}"
    assert all("git_commit" in entry and "started_at" in entry for entry in history), f"history {history}"
    print("PASS  each invocation appends its settings and git state to the history")


def test_sweep_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        checkpoints = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints)
        for step in (0, 50_000, 100_000):
            write_checkpoint(checkpoints, step)
        out = os.path.join(run_dir, "eval.csv")

        first = run_sweep(run_dir, "--every", "50000", "--through", "50000")
        assert first.returncode == 0, first.stderr
        assert "Device: cpu" in first.stdout or "Device: mps" in first.stdout, (
            f"sweep did not stay off the NVIDIA GPU:\n{first.stdout}")
        rows = read_rows(out)
        assert [int(r["env_step"]) for r in rows] == [0, 50_000], f"--through gave {rows}"
        assert list(rows[0]) == sweep_eval.COLUMNS, f"columns {list(rows[0])}"
        assert all(int(r["episodes"]) == 1 for r in rows), "episode count not recorded"

        second = run_sweep(run_dir, "--every", "50000")
        assert second.returncode == 0, second.stderr
        rows = read_rows(out)
        assert [int(r["env_step"]) for r in rows] == [0, 50_000, 100_000], f"resume gave {rows}"
        assert "to do: 1" in second.stdout, f"resume should have had one to do:\n{second.stdout}"
        assert len(json.load(open(os.path.join(run_dir, "eval_config.json")))) == 2, "history not appended"
    print("PASS  sweep honors --through, writes every column, and resumes without redoing rows")


def test_sweep_rejects_mislabeled_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        checkpoints = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints)
        write_checkpoint(checkpoints, 200_000, held_step=5)
        result = run_sweep(run_dir, "--every", "200000")

    assert result.returncode != 0 and "holds 5" in result.stderr, (
        f"a checkpoint named 200000 holding step 5 was accepted:\n{result.stderr}"
    )
    print("PASS  a checkpoint whose counter disagrees with its name stops the sweep")


def test_sweep_refuses_what_it_would_skip() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        checkpoints = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoints)
        write_checkpoint(checkpoints, 50_000)
        no_match = run_sweep(run_dir, "--every", "200000")
        other_seed = os.path.join(run_dir, "eval.csv")
        write_rows(other_seed, [(50_000, 1, 0.05, 0)])
        mixed = run_sweep(run_dir, "--every", "50000", "--seed", "1")
        bad_epsilon = run_sweep(run_dir, "--every", "50000", "--epsilon", "5")

    assert no_match.returncode != 0 and "no checkpoint" in no_match.stderr, no_match.stdout
    assert mixed.returncode != 0 and "Point --out somewhere else" in mixed.stderr, mixed.stdout
    assert bad_epsilon.returncode != 0 and "--epsilon" in bad_epsilon.stderr, bad_epsilon.stdout
    print("PASS  the sweep stops on no matching checkpoint, rows of another seed, and a bad epsilon")


if __name__ == "__main__":
    test_checkpoint_selection()
    test_already_done_refuses_old_columns()
    test_already_done_refuses_other_settings()
    test_record_invocation_appends()
    test_sweep_end_to_end()
    test_sweep_rejects_mislabeled_checkpoint()
    test_sweep_refuses_what_it_would_skip()
    print("\nAll tests passed!")
