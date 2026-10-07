import os
import re

# One root for every run, real and tiny, so they are greppable from one place.
# Outside the OneDrive sync root, and home relative so it resolves on both machines.
RUNS_ROOT = os.path.expanduser(os.path.join("~", "pong-runs"))


def next_run_dir(root: str, prefix: str = "run-") -> str:
    # root/run-NN, one past the highest NN already there. Called from the launch
    # script, so the path train() receives is still an explicit argument.
    os.makedirs(root, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    used = []
    for name in os.listdir(root):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(root, name)):
            used.append(int(match.group(1)))

    return os.path.join(root, f"{prefix}{max(used, default=0) + 1:02d}")
 