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
    # CPU, so the test neither needs a GPU nor competes with a sweep already on one.
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


def test_already_done_refuses_old_columns() -> None:
    with tempfile.TemporaryDirectory() as directory:
        current = os.path.join(directory, "current.csv")
        with open(current, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(sweep_eval.COLUMNS)
            writer.writerow([0] + [""] * (len(sweep_eval.COLUMNS) - 1))
            writer.writerow([200_000] + [""] * (len(sweep_eval.COLUMNS) - 1))
        assert sweep_eval.already_done(current) == {0, 200_000}, "already_done missed rows"
        assert sweep_eval.already_done(os.path.join(directory, "missing.csv")) == set()

        old = os.path.join(directory, "old.csv")
        with open(old, "w", newline="") as handle:
            csv.writer(handle).writerow(sweep_eval.COLUMNS[:-2] + ["elapsed_s"])
        try:
            sweep_eval.already_done(old)
        except ValueError:
            print("PASS  already_done reads finished steps and refuses a file with other columns")
            return
    raise AssertionError("already_done accepted a file with different columns")


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
        assert "Device: cpu" in first.stdout, f"sweep did not run on CPU:\n{first.stdout}"
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
    print("PASS  sweep honours --through, writes every column, and resumes without redoing rows")


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


if __name__ == "__main__":
    test_checkpoint_selection()
    test_already_done_refuses_old_columns()
    test_record_invocation_appends()
    test_sweep_end_to_end()
    test_sweep_rejects_mislabeled_checkpoint()
    print("\nAll tests passed!")
