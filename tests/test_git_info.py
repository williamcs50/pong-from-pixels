import os
import re
import subprocess
import sys
from collections.abc import Callable
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import git_info
from src.git_info import git_state, software_versions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fake_git(status: str, commit: str = "a" * 40) -> Callable[..., subprocess.CompletedProcess]:
    """Return a subprocess.run stand in: git status prints status, anything else prints commit."""
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        output = status if "status" in command else commit
        return subprocess.CompletedProcess(command, 0, stdout=output + "\n", stderr="")
    return run


def test_real_repo_matches_git() -> None:
    state = git_state()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()

    assert re.fullmatch(r"[0-9a-f]{40}", state["git_commit"] or ""), (
        f"Expected a full 40 character hash, got {state['git_commit']!r}"
    )
    assert state["git_dirty"] == bool(status), (
        f"git_dirty is {state['git_dirty']} but git status reports {status!r}"
    )
    print(f"PASS  git_state reports the full hash and the same dirty flag as git (dirty={state['git_dirty']})")


def test_clean_status_is_not_dirty() -> None:
    with mock.patch.object(git_info.subprocess, "run", fake_git("")):
        state = git_state()
    assert state == {"git_commit": "a" * 40, "git_dirty": False, "git_status": ""}, f"Got {state}"
    print("PASS  empty git status reads as clean")


def test_untracked_file_is_dirty() -> None:
    with mock.patch.object(git_info.subprocess, "run", fake_git("?? src/new_module.py")):
        state = git_state()
    assert state["git_dirty"] is True, f"An untracked file should count as dirty, got {state}"
    assert state["git_status"] == "?? src/new_module.py", f"Status text lost: {state['git_status']!r}"
    print("PASS  an untracked file counts as dirty")


def test_unusable_git_is_unknown_not_clean() -> None:
    for error in (OSError("git not installed"), subprocess.CalledProcessError(128, "git")):
        with mock.patch.object(git_info.subprocess, "run", side_effect=error):
            state = git_state()
        assert state["git_commit"] is None and state["git_dirty"] is None, (
            f"{type(error).__name__} should give None, not a verdict, got {state}"
        )
    print("PASS  missing or failing git reports None rather than clean")


def test_versions_survive_cudnn_failure() -> None:
    with mock.patch.object(git_info.torch.backends.cudnn, "version", side_effect=ValueError("no device")):
        versions = software_versions()
    assert versions["cudnn_version"] is None, f"cudnn_version {versions['cudnn_version']!r}"
    assert versions["torch_version"] == git_info.torch.__version__, f"torch_version {versions['torch_version']}"
    print("PASS  software_versions records None for cuDNN instead of crashing")


if __name__ == "__main__":
    test_real_repo_matches_git()
    test_clean_status_is_not_dirty()
    test_untracked_file_is_dirty()
    test_unusable_git_is_unknown_not_clean()
    test_versions_survive_cudnn_failure()
    print("\nAll tests passed!")
