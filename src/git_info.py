# src/git_info.py
# The commit, whether the working tree was clean, and the software versions, recorded
# with every run, sweep, and clip.

import os
import platform
import subprocess
import sys
from importlib import metadata

import torch

# The packages whose behavior reaches a run's numbers. Read from what is installed,
# since requirements.txt only pins OpenCV and numpy.
PACKAGES = ("gymnasium", "ale-py", "numpy", "opencv-python-headless", "imageio-ffmpeg")


def git_state() -> dict:
    # A hash recorded while files were modified does not describe the code that ran,
    # so the status comes back with it. Untracked files count: src/checkpoint.py was
    # untracked and imported by train.py, so a new unignored file can change
    # behavior. commit None means git was unusable, which is unverifiable rather
    # than clean.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(*command: str) -> str:
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


def software_versions() -> dict:
    # cuDNN raises rather than returning None when CUDA reports available but no
    # device is visible, and recording a version must never stop a run.
    try:
        cudnn = torch.backends.cudnn.version()
    except (RuntimeError, ValueError):
        cudnn = None

    versions = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn,
    }
    for package in PACKAGES:
        try:
            versions[f"{package}_version"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[f"{package}_version"] = None
    return versions
