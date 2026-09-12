import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium as gym

from src.replay_buffer import ReplayBuffer

gym.register_envs(ale_py)

# torch is the one guarded import. requirements.txt deliberately leaves it out
# and tells you to install it manually for your hardware, so "not installed" is
# a state this script exists to report, not one it should crash on. Everything
# torch-dependent, QNetwork included, is unavailable in that case.
try:
    import torch

    from src.q_network import QNetwork
except ImportError:
    torch = None
    QNetwork = None


def check_gymnasium():
    try:
        env = gym.make("ALE/Pong-v5")
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("OK  gymnasium + Pong environment")
    except Exception as e:
        print(f"FAIL  gymnasium: {e}")
        sys.exit(1)


def check_torch():
    if torch is None:
        print("FAIL  torch not installed (needed for training)")
        return
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"OK  torch {torch.__version__} (device: {device})")
    if not cuda:
        print("WARN  CUDA not available, training will be slow on CPU")
        return
    print(f"OK  GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print(f"OK  GPU compute verified (result device: {y.device})")


def check_vram_baseline():
    # Baseline before the training loop exists, so its growth can be attributed
    # later instead of guessed at. The number that constrains batch size is free
    # VRAM, not total: the desktop and browser already hold part of the 6GB.
    if torch is None:
        return
    if not torch.cuda.is_available():
        print("SKIP  VRAM baseline (no CUDA)")
        return

    mib = 1024 ** 2
    free_before, total = torch.cuda.mem_get_info()
    print(f"OK  VRAM total: {total / mib:.0f} MiB, free before allocation: {free_before / mib:.0f} MiB")

    # Pong has 6 actions. Target network is a second instance with frozen weights.
    q_network = QNetwork(n_actions=6).cuda()
    target_network = QNetwork(n_actions=6).cuda()
    target_network.load_state_dict(q_network.state_dict())
    weights = torch.cuda.memory_allocated()
    print(f"OK  both networks resident: {weights / mib:.1f} MiB")

    # Optimizer state is allocated lazily, so it does not appear until a step
    # runs. That, and the activation peak during forward and backward, are the
    # numbers that need the training loop and cannot be measured here yet.
    print("NOTE  optimizer state and the activation peak per training step are not measured yet, they need the training loop")


class MEMORYSTATUSEX(ctypes.Structure):
    # Field layout GlobalMemoryStatusEx expects. Only ullTotalPhys and
    # ullAvailPhys get read, but all nine have to be declared so that
    # sizeof matches what the API writes back.
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def check_host_ram():
    # The replay buffer is sized against host RAM, so this is the number that
    # decides the capacity. Win32 through ctypes because psutil is not a
    # dependency and is not worth adding for two numbers.
    if sys.platform != "win32":
        print("SKIP  host RAM (not Windows)")
        return
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    gib = 1024 ** 3
    # Total is stable, available moves with whatever else is running.
    print(f"OK  host RAM: {status.ullTotalPhys / gib:.2f} GiB total, {status.ullAvailPhys / gib:.2f} GiB available right now")


def check_buffer_ram(capacity=100_000):
    # The replay buffer is the big allocation and it is host RAM, not VRAM.
    # Measured by summing nbytes off a small buffer rather than hardcoding
    # 28,224, so this stays correct if a dtype or the 84x84x4 shape changes.
    sample_size = 1000
    sample = ReplayBuffer(capacity=sample_size)
    fields = (sample.current_state, sample.next_state, sample.action_taken, sample.reward, sample.done)
    per_transition = sum(field.nbytes for field in fields) // sample_size
    total_gib = per_transition * capacity / 1024 ** 3
    print(f"OK  replay buffer at capacity {capacity:,}: {total_gib:.2f} GiB host RAM ({per_transition:,} bytes/transition)")
    # np.zeros reserves all of it in __init__, so it fits at startup or not at all.
    print("NOTE  buffer RAM is allocated eagerly at construction, and state/next_state store every frame twice")


if __name__ == "__main__":
    print("Checking environment...\n")
    check_gymnasium()
    check_torch()
    check_vram_baseline()
    check_host_ram()
    check_buffer_ram()
    print("\nAll checks passed.")
