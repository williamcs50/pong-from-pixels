import os
import subprocess


def git_state() -> dict:
    # A hash recorded while files were modified does not describe the code that ran,
    # so the status comes back with it. Untracked files count: src/checkpoint.py was
    # untracked and imported by train.py, so a new unignored file can change
    # behaviour. commit None means git was unusable, which is unverifiable rather
    # than clean.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(*command):
        return subprocess.run(
            command, cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        status = run("git", "status", "--porcelain")
        # Full hash, not --short: short hashes are a display convenience and can
        # become ambiguous as the history grows.
        return {"git_commit": run("git", "rev-parse", "HEAD"),
                "git_dirty": bool(status),
                "git_status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None, "git_status": ""}
