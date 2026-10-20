# scripts/check_environment.py
# Check that Pong, PyTorch, and the GPU work, and measure the memory a real run needs.

import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ale_py
import gymnasium as gym

from src.replay_buffer import ReplayBuffer

gym.register_envs(ale_py)

ADAM_MOMENTS = 2  # first and second moment, one float per Q weight each

# torch is the one guarded import. requirements.txt deliberately leaves it out
# and tells you to install it manually for your hardware, so "not installed" is
# a state this script exists to report, not one it should crash on. Everything
# that depends on torch, QNetwork included, is unavailable in that case.
try:
    import torch

    from scripts.real_train import REAL_CONFIG
    from src.device import pick_device
    from src.q_network import QNetwork
    from src.train import train_step
except ImportError:
    torch = None
    QNetwork = None


def check_gymnasium() -> None:
    try:
        env = gym.make("ALE/Pong-v5", repeat_action_probability=0.0)
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("OK  gymnasium + Pong environment")
    except Exception as e:
        print(f"FAIL  gymnasium: {e}")
        sys.exit(1)


def check_torch() -> None:
    if torch is None:
        print("FAIL  torch not installed (needed for training)")
        return
    device = pick_device()
    print(f"OK  torch {torch.__version__} (device: {device.type})")
    if device.type == "mps":
        print("OK  Apple GPU (MPS) found. Training uses it, but the memory checks below are NVIDIA only")
        return
    if device.type == "cpu":
        print("WARN  no GPU found, training will be slow on CPU")
        return
    print(f"OK  GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print(f"OK  GPU compute verified (result device: {y.device})")


def check_vram_baseline() -> None:
    # The number that constrains batch size is free VRAM, not total: the desktop
    # and browser already hold part of the 6GB.
    if torch is None:
        return
    if not torch.cuda.is_available():
        print("SKIP  VRAM baseline (no CUDA)")
        return

    mib = 1024 ** 2
    free_before, total = torch.cuda.mem_get_info()
    print(f"OK  VRAM total: {total / mib:.0f} MiB, free before allocation: {free_before / mib:.0f} MiB")

    # Measured as growth, since earlier checks leave allocations behind, such as the
    # cuBLAS workspace from check_torch's matrix multiply.
    before_networks = torch.cuda.memory_allocated()

    # Pong has 6 actions. Target network is a second instance with frozen weights.
    q_network = QNetwork(n_actions=6).cuda()
    target_network = QNetwork(n_actions=6).cuda()
    target_network.load_state_dict(q_network.state_dict())
    weights = torch.cuda.memory_allocated() - before_networks
    print(f"OK  both networks resident: {weights / mib:.1f} MiB")

    # Optimizer state is allocated lazily, so it only appears once a real step runs.
    optimizer = torch.optim.Adam(q_network.parameters(), lr=REAL_CONFIG["learning_rate"])
    batch_size = REAL_CONFIG["batch_size"]
    shapes = ReplayBuffer(capacity=batch_size)
    batch = (shapes.current_state, shapes.action_taken, shapes.reward, shapes.next_state, shapes.done)

    train_step(q_network, target_network, optimizer, batch, REAL_CONFIG["gamma"])
    param_bytes = sum(p.nbytes for p in q_network.parameters())
    adam_bytes = sum(t.nbytes for state in optimizer.state.values()
                     for t in state.values() if torch.is_tensor(t))
    print(f"OK  Adam state: {adam_bytes / mib:.2f} MiB, against {ADAM_MOMENTS * param_bytes / mib:.2f} "
          f"MiB for {ADAM_MOMENTS} moments of {param_bytes / mib:.2f} MiB of Q weights")

    # Measured on a second step, so the one time allocations of the first are not counted.
    steady = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    train_step(q_network, target_network, optimizer, batch, REAL_CONFIG["gamma"])
    print(f"OK  training step at batch {batch_size}: {(steady - before_networks) / mib:.1f} MiB resident, "
          f"peak {(torch.cuda.max_memory_allocated() - steady) / mib:.1f} MiB above it")
    print("NOTE  these count this process's tensors only. nvidia-smi and gpu_log.csv report the whole "
          "card, including the CUDA context and other programs")


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


def check_host_ram() -> None:
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


def check_buffer_ram(capacity: int = 100_000) -> None:
    # The replay buffer is the big allocation and it is host RAM, not VRAM.
    # Measured by summing nbytes off a small buffer rather than hardcoding
    # 28,224, so this stays correct if a dtype or the 84x84x4 shape changes.
    sample_size = 1000
    sample = ReplayBuffer(capacity=sample_size)
    fields = (sample.current_state, sample.next_state, sample.action_taken, sample.reward, sample.done)
    per_transition = sum(field.nbytes for field in fields) // sample_size
    total_gib = per_transition * capacity / 1024 ** 3
    print(f"OK  replay buffer at capacity {capacity:,}: {total_gib:.2f} GiB host RAM ({per_transition:,} bytes/transition)")


if __name__ == "__main__":
    print("Checking environment...\n")
    check_gymnasium()
    check_torch()
    check_vram_baseline()
    check_host_ram()
    check_buffer_ram()
    print("\nAll checks passed.")
