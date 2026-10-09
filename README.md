# Pong from Pixels

Pong from Pixels is a Deep Q-Network (DQN) project built from scratch that learns to play Atari Pong directly from raw screen pixels on local GPU hardware, documenting the real training curve, including failures, as a reproducible reinforcement learning case study.

## Setup

```bash
git clone https://github.com/williamcs50/pong-from-pixels.git
cd pong-from-pixels
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install PyTorch first, because the right command depends on your hardware and
`requirements.txt` deliberately leaves it out. `pip install torch` on its own gives
you a CPU only build on Windows and Linux, with no warning and no error, and
training then runs on the CPU.

```bash
# Windows or Linux with an NVIDIA GPU (CUDA 12.6)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Mac with Apple Silicon, or CPU only
pip install torch
```

Then the rest:

```bash
pip install -r requirements.txt
```

## Verify your environment

```bash
python scripts/check_environment.py
```

Confirms Gymnasium + Pong, PyTorch version, CUDA availability, GPU name, and a live matrix multiply on the GPU. It also reports the memory budget: total and free VRAM, host RAM, and what the replay buffer will cost at its configured capacity.

```bash
python -m pytest -q
```

## Run the random agent

```bash
python src/random_agent.py
```

## Train

```bash
# a short run that exercises the whole pipeline, finishes in seconds
python scripts/tiny_train.py

# a warmup pass at real settings, long enough to measure the rate
python scripts/real_train.py --total-steps 120000 --output-dir <path>/run-00-warmup

# the real run, 2,000,000 steps
python scripts/real_train.py --output-dir <path>/run-01
```

Both scripts call the same `train()`, so the short run says something about the real
one, and the warmup pass is the real run with one number changed.

Pass `--output-dir` rather than letting the numbering pick, because the GPU logger
in [docs/launch_checklist.md](docs/launch_checklist.md) needs the path before the
run starts. Without the flag the directory is the next `run-NN` under `RUNS_ROOT`,
which you only learn from the startup block once the run is already going.

`train()` refuses to start on a dirty working tree, since the commit hash it records
would not then describe the code that ran. `--allow-dirty` overrides that for
throwaway runs and is recorded in `run_config.json` when used.

Each run writes its own directory outside the repository: `episodes.csv` with one
row per episode, `run_config.json` with the resolved config and the commit, and
`checkpoints/` holding the weights, optimizer and RNG state at step 0 and every N
steps. [docs/launch_checklist.md](docs/launch_checklist.md) covers the checks the
code cannot do for itself, including the GPU logging command.

## Inspect a run

```bash
# every figure a run produced, from its own logs
python scripts/analyze_run.py docs/runs/run-01

# play episodes from a saved checkpoint and report the mean reward
python scripts/play_checkpoint.py <checkpoint.pt> --episodes 100
```

```bash
# the preprocessing pipeline checked numerically against real frames
python scripts/visual_check.py
```

## Results

Run 1's result, its committed logs, and the command that reproduces every figure
are in [docs/runs/run-01.md](docs/runs/run-01.md).

## Documentation

- [docs/PLANS.md](docs/PLANS.md), the finish line and the schedule.
- [docs/architecture.md](docs/architecture.md), every component, its interface, and
  why it is built rather than imported.
- [docs/sessions.md](docs/sessions.md), one entry per working day, with the goals
  set at the start and what landed, what is open, and what was surprising.
- [docs/launch_checklist.md](docs/launch_checklist.md), the pre launch and first
  five minutes checks for a real run.
