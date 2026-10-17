# Pong from Pixels

Pong from Pixels is a Deep Q Network (DQN) project built from scratch that learns to play Atari Pong directly from raw screen pixels on local GPU hardware, documenting the real training curve, including failures, as a reproducible reinforcement learning case study.

It follows the DQN from DeepMind's 2015 Nature paper, [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) [1], first described in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) [2]. Where the docs compare against "Nature", they mean the 2015 paper.

## Results

Run 2 trained for 6,000,000 steps on an RTX 2060 in 6.88 hours. A step is one move by the agent, and every 50,000 steps the run saves a checkpoint, a copy of the network as it was at that point. Run 2's best checkpoint, at step 5,400,000, scored +16.97 over 100 evaluation games, so it beats the built in opponent, which means scoring above 0. Run 1 stopped at 2,000,000 steps, and its best checkpoint scored -1.63.

A game's reward is the points the agent won minus the points it lost. A game ends when either side reaches 21, so rewards run from -21, losing 21 to 0, to +21, winning 21 to 0. +16.97 means the agent won its games by about 21 to 4 on average.

- [docs/runs/run-02.md](docs/runs/run-02.md): Run 2's predictions, results, evaluation, instrumentation, and the four milestone clips.
- [docs/runs/run-01.md](docs/runs/run-01.md): Run 1, where no checkpoint scored above 0, and why that led to Run 2.

Every figure in both write ups comes from log files committed in [docs/runs/](docs/runs/), and `analyze_run.py` prints almost all of them (see [Reproduce the figures](#reproduce-the-figures)).

## Setup

You'll need Python 3.11, the version every run here used. Training uses a GPU when there is one: an NVIDIA card, or the built in GPU of an Apple Silicon Mac. Every run here was trained on an NVIDIA card, so the Mac's speed hasn't been measured. Without either, it runs on the CPU, and a full run takes far longer.

```bash
git clone https://github.com/williamcs50/pong-from-pixels.git
cd pong-from-pixels

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

Once the venv is active, `python` works on every system, so every command below uses it.

Install PyTorch first, because the right command depends on your hardware and `requirements.txt` deliberately leaves it out. On Windows, `pip install torch` on its own gives you a CPU only build, with no warning and no error, and training then runs on the CPU.

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

## Check it works

```bash
python scripts/check_environment.py
```

Confirms that Pong runs, which PyTorch version is installed, and which device training will use: an NVIDIA GPU, an Apple Silicon Mac's GPU, or the CPU. It also reports how much RAM training will need, since training stores the last 100,000 moves in RAM to learn from, and on Windows, how much RAM the computer has. On an NVIDIA card, it also checks the GPU can do a calculation and reports how much of its memory is free.

```bash
python -m pytest -q
```

Runs every test in [tests/](tests/). Each test file also runs on its own, for example `python tests/test_evaluate.py`.

```bash
python scripts/random_agent.py
```

Opens a game window with an agent pressing random buttons, the starting point before any training. It plays one game, prints the score, and closes. At seed 0 it loses 21 to 1. `--episodes` plays more games, and `--seed` picks a different one.

```bash
python scripts/visual_check.py
```

Preprocessing turns each game screen into the small gray image the network sees. This checks that step with numbers against real frames, and checks which way each of the six actions moves the paddle.

## Reproduce the figures

These need only the repository, not a GPU or a trained model:

```bash
# every figure in run-02.md, from Run 2's committed logs
python scripts/analyze_run.py docs/runs/run-02

# every comparison with Run 1
python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-01

# Run 1's figures
python scripts/analyze_run.py docs/runs/run-01
```

[run-02.md](docs/runs/run-02.md#reproducing-the-figures) lists which command produces which figure, and the few that need something else.

## Train

Training is three steps, each a longer version of the one before. All three run the same training code, `train()`, with different settings.

**1. Check the pipeline works.** About 15 seconds on the RTX 2060.

```bash
python scripts/tiny_train.py
```

It plays 1,500 steps, about one game, and trains on them. It doesn't learn anything. It just proves every piece runs together. Its output goes in `~/pong-runs/tiny`.

**2. Measure your speed with a warmup.** About 6 minutes on the RTX 2060.

```bash
python scripts/real_train.py --total-steps 120000 --output-dir ~/pong-runs/my-warmup
```

This is the real run cut short. Its steps per second tell you how long the full run will take, and it's a chance to check the GPU temperature under load.

**3. Train for real.** Run 2 took 6.88 hours on the RTX 2060.

```bash
python scripts/real_train.py --total-steps 6000000 --output-dir ~/pong-runs/my-run
```

6,000,000 steps is Run 2's length. Without `--total-steps`, it runs 2,000,000, Run 1's length.

`my-warmup` and `my-run` are just example folder names. Change them to whatever you want to call each run, as long as the folder doesn't already hold a run. You don't need to make the folder first, since `real_train.py` creates it. If you rename `my-run`, use the new name everywhere the launch checklist says `my-run` too, so the run and its GPU log end up in the same folder. The checklist has you make the folder ahead of time, because its GPU logger starts writing into it before training does.

Before you start:

- **Use a new folder each time.** A run won't start in a folder that already holds one. Everything goes under `~/pong-runs`, outside the repository, so a run is never committed by accident. Without `--output-dir`, it picks the next free `run-NN` folder, but you only find out which once it's running, and the GPU logger needs the path before that.
- **Commit your changes first.** `train()` won't start with uncommitted changes, since the commit it records wouldn't describe the code that ran. `--allow-dirty` overrides that for throwaway runs, and `run_config.json` records that you used it.
- **Read [docs/launch_checklist.md](docs/launch_checklist.md) before step 3.** It covers the checks the code can't do for itself, including logging the GPU's temperature.

## What a run writes

| File | Written by | What it holds |
|---|---|---|
| `run_config.json` | `train()` | every setting the run used and the commit it ran from. Runs after Run 2 also record the software versions. |
| `episodes.csv` | `train()` | one row per game: reward, length, epsilon (the chance of a random move), loss, and time |
| `metrics.csv` | `train()` | one row every 10,000 steps: the network's Q values on the probe states (its estimate of how good each move is), which move it picks for each, epsilon, and how often its training batches include a scored point or the end of a game |
| `probe.npy` | `train()` | 500 fixed game states. The same states are scored at every row of `metrics.csv`, so any change comes from the network, not the states. It regenerates from a seed. |
| `checkpoints/` | `train()` | both networks' weights, the optimizer, the step counters, and the random number generators, at step 0 and every 50,000 steps |
| `gpu_log.csv` | the logging loop in the launch checklist | GPU temperature, clock, power, use, and memory every 30 seconds |
| `eval.csv`, `eval_config.json` | `sweep_eval.py` | one row per evaluated checkpoint, and the settings and commit of each sweep |
| `clips/` | `record_clip.py` | the milestone clips, each with its Q values and a record of how it was made |

For Run 2, everything except the checkpoints and the probe is committed in [docs/runs/run-02/](docs/runs/run-02/). Run 1 came before the instrumentation and the clips, so it has no `metrics.csv`, probe, or clips. The checkpoints for every run are too big to commit, so they're in the repo's [Releases](https://github.com/williamcs50/pong-from-pixels/releases) instead.

## Get the checkpoints

Watching or evaluating a run needs its checkpoints. You can train your own, or download Run 1's and Run 2's from the Releases. There's one Release per run:

| Release | Checkpoints | Size |
|---|---|---|
| [run-02-checkpoints](https://github.com/williamcs50/pong-from-pixels/releases/tag/run-02-checkpoints) | 121, every 50,000 steps to 6,000,000, plus `probe.npy` | 3.3 GB |
| [run-01-checkpoints](https://github.com/williamcs50/pong-from-pixels/releases/tag/run-01-checkpoints) | 41, every 50,000 steps to 2,000,000 | 1.1 GB |
| [run-02-warmup-checkpoints](https://github.com/williamcs50/pong-from-pixels/releases/tag/run-02-warmup-checkpoints) | 4 | 0.1 GB |
| [run-00-warmup-checkpoints](https://github.com/williamcs50/pong-from-pixels/releases/tag/run-00-warmup-checkpoints) | 4 | 0.1 GB |

To just watch the trained agent, you only need Run 2's best checkpoint, 27 MB. This downloads it to where the scripts look for it, on any system:

```bash
python -c "import os, urllib.request; d = os.path.expanduser('~/pong-runs/run-02/checkpoints'); os.makedirs(d, exist_ok=True); urllib.request.urlretrieve('https://github.com/williamcs50/pong-from-pixels/releases/download/run-02-checkpoints/step_05400000.pt', os.path.join(d, 'step_05400000.pt'))"
```

Change both `step_05400000` names to get a different checkpoint, and `run-02` in both places to get one from another run.

To download all of them, about 4.5 GB, install the [GitHub CLI](https://cli.github.com) and sign in with `gh auth login`. Then, on Mac or Linux:

```bash
for run in run-00-warmup run-01 run-02-warmup run-02; do
  mkdir -p ~/pong-runs/$run/checkpoints
  gh release download $run-checkpoints -R williamcs50/pong-from-pixels --dir ~/pong-runs/$run/checkpoints --skip-existing
done
```

Or on Windows, in PowerShell:

```powershell
foreach ($run in "run-00-warmup", "run-01", "run-02-warmup", "run-02") {
  New-Item -ItemType Directory -Force "$HOME/pong-runs/$run/checkpoints" | Out-Null
  gh release download "$run-checkpoints" -R williamcs50/pong-from-pixels --dir "$HOME/pong-runs/$run/checkpoints" --skip-existing
}
```

**If the download stops partway,** that's normal. GitHub sometimes pauses downloads after too many files in a row. Wait a while, up to an hour, then run the same loop again. It skips the files you already have and only downloads the missing ones.

**To check you have every file,** count them. On Mac or Linux:

```bash
for run in run-00-warmup run-01 run-02-warmup run-02; do
  echo "$run: $(ls ~/pong-runs/$run/checkpoints/*.pt | wc -l) files"
done
```

Or on Windows, in PowerShell:

```powershell
foreach ($run in "run-00-warmup", "run-01", "run-02-warmup", "run-02") {
  "${run}: $((Get-ChildItem "$HOME/pong-runs/$run/checkpoints/*.pt").Count) files"
}
```

You should see 4, 41, 4, and 121. If any number is lower, run the download loop again.

Run 2's Release also has its original `probe.npy`, the 500 states its Q values were measured on. Training rebuilds the same states from a seed, so you only need it to measure on Run 2's exact states if a future version of the game emulator ever rebuilds them differently. The loops above save it in the `checkpoints` folder. Move it up one level, to `~/pong-runs/run-02/`, where a run keeps it.

## Watch it play

`watch_checkpoint.py` opens a game window and plays a checkpoint at real speed, so you can watch the agent you'd otherwise only see as numbers. It plays through the same code that scored the checkpoints, so you see the same games.

**The default is Run 2's best checkpoint,** step 5,400,000:

```bash
python scripts/watch_checkpoint.py
```

With no path, it plays `~/pong-runs/run-02/checkpoints/step_05400000.pt`, so download that one first (see [Get the checkpoints](#get-the-checkpoints)). It plays one game at seed 0, which on the RTX 2060 is a 21 to 3 win, the same game as the Trained clip. That takes about 2 and a half minutes. When the game ends, the window closes and the score prints:

```
  episode 1: reward +18 (won 21 to 3), 2386 steps
```

**To watch any other checkpoint,** pass its path. It can come from Run 2, Run 1, or a run of your own:

```bash
# Run 2 at step 2,800,000, the first scored checkpoint to beat the opponent
python scripts/watch_checkpoint.py ~/pong-runs/run-02/checkpoints/step_02800000.pt

# Run 1's best checkpoint, step 1,800,000
python scripts/watch_checkpoint.py ~/pong-runs/run-01/checkpoints/step_01800000.pt
```

**To watch a run you trained yourself,** point it at that run's `checkpoints` folder, inside the folder you passed to `--output-dir`. Each file is named `step_` plus the step number padded to 8 digits, so step 600,000 is `step_00600000.pt`. A run saves one at step 0, every 50,000 steps, and at its last step. To see which ones yours has, on Mac or Linux:

```bash
ls ~/pong-runs/my-run/checkpoints
```

Or on Windows, in PowerShell:

```powershell
Get-ChildItem ~/pong-runs/my-run/checkpoints
```

Then pass the one you want, for example the last one from a 6,000,000 step run:

```bash
python scripts/watch_checkpoint.py ~/pong-runs/my-run/checkpoints/step_06000000.pt
```

Swap `my-run` for your run's folder name.

**To change the game,** use these options:

| Option | Default | What it does |
|---|---|---|
| `--episodes` | 1 | how many games to play in a row |
| `--seed` | 0 | which games to play. The same seed always plays the same games. |
| `--epsilon` | 0.05 | how often the agent makes a random move instead of its own choice. 0.05 is what every score here was measured at. |

For example, Run 2's step 600,000 never returns the ball in its seed 0 game, but its seed 6 game has the rally the First rally clip shows:

```bash
python scripts/watch_checkpoint.py ~/pong-runs/run-02/checkpoints/step_00600000.pt --seed 6
```

To stop before the games finish, press Ctrl+C in the terminal.

On anything other than the RTX 2060 that scored these games, like a Mac or a CPU, a game may go slightly differently from the one described here. The agent is the same, so it plays just as well.

## Evaluate a run

These need a run's checkpoints too:

```bash
# play one checkpoint and report its mean reward, here Run 2's best, which scores +16.97
python scripts/play_checkpoint.py ~/pong-runs/run-02/checkpoints/step_05400000.pt --episodes 100

# evaluate every 200,000 step checkpoint over 100 games, writing a new results file
python scripts/sweep_eval.py ~/pong-runs/run-02 --every 200000 --out ~/pong-runs/run-02/eval_rerun.csv

# record a 30 second clip, the last 30 seconds of the seed 0 game
python scripts/record_clip.py ~/pong-runs/run-02/checkpoints/step_05400000.pt

# record a clip of the first game with a rally instead, as for the first rally clip
python scripts/record_clip.py ~/pong-runs/run-02/checkpoints/step_00600000.pt --rally
```

A checkpoint beats the built in opponent when its mean reward over 100 evaluation games is above 0. Evaluation plays at epsilon 0.05 with a fixed seed, so the same checkpoint always plays the same games. `record_clip.py` relies on that: it plays each game twice and only saves the clip if both plays match.

A sweep of a full run takes hours. Run 2's took about 3 hours 10 minutes. Without `--out`, it adds rows to the run's `eval.csv`. It refuses if that file was written with different columns or different settings, like another seed, rather than mixing them into one file. The `eval.csv` from Run 2's original sweep has older columns, which is why the command above writes to a new file. A sweep that stops partway picks up where it left off when you run the same command again.

Clips are saved next to the run's checkpoints, in `~/pong-runs/run-02/clips/`, not in the repository. The committed clips in [docs/runs/run-02/clips/](docs/runs/run-02/clips/) were copied there by hand. `record_clip.py` won't replace a clip that's already there unless you pass `--overwrite`.

## Documentation

- [docs/PLANS.md](docs/PLANS.md), the finish line and the schedule.
- [docs/architecture.md](docs/architecture.md), every component, its interface, and why it's built rather than imported.
- [docs/sessions.md](docs/sessions.md), one entry per working day, with the goals set at the start and what landed, what's open, and what was surprising.
- [docs/launch_checklist.md](docs/launch_checklist.md), a real run from start to finish: the checks before launch, the launch itself in two terminals, and what to check while it runs and when it ends.

## References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Human-level control through deep reinforcement learning." *Nature* 518, pages 529 to 533 (2015). https://www.nature.com/articles/nature14236
2. Mnih, V., Kavukcuoglu, K., Silver, D., et al. "Playing Atari with Deep Reinforcement Learning." arXiv:1312.5602 (2013). https://arxiv.org/abs/1312.5602
