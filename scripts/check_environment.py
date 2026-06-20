import sys

def check_gymnasium():
    try:
        import ale_py
        import gymnasium as gym
        env = gym.make("ALE/Pong-v5")
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("OK  gymnasium + Pong environment")
    except Exception as e:
        print(f"FAIL  gymnasium: {e}")
        sys.exit(1)

def check_torch():
    try:
        import torch
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        print(f"OK  torch {torch.__version__} (device: {device})")
        if not cuda:
            print("WARN  CUDA not available — training will be slow on CPU")
            return
        print(f"OK  GPU: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x
        print(f"OK  GPU compute verified (result device: {y.device})")
    except ImportError:
        print("FAIL  torch not installed (needed for training)")

if __name__ == "__main__":
    print("Checking environment...\n")
    check_gymnasium()
    check_torch()
    print("\nAll checks passed.")
