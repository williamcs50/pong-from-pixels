# Launch Checklist

Only the checks the code can't do for itself. Measured values go in that day's
[sessions.md](sessions.md) entry, so runs can be compared. Run 1 and Run 2 both
launched from this list, and it includes what they turned up.

---

## Before launch

- [ ] The computer won't sleep or restart for updates during the run.
  - **Windows:** Settings, then System, then Power & battery (just Power on a desktop), and set sleep to Never when plugged in. Then Settings, then Windows Update, and pause updates.
  - **Mac:** the Mac launch command below keeps it awake. Then System Settings, then General, then Software Update, and turn off automatic updates.
- [ ] `CUDA_VISIBLE_DEVICES` not set. Setting it to `-1` for a CPU check silently sends every later run in that terminal to the CPU.
- [ ] `git status` clean. `train()` refuses to start with uncommitted changes, so this is enforced rather than remembered.
- [ ] `python scripts/check_environment.py` passes. It checks host RAM and VRAM in one command.
- [ ] All tests pass.
- [ ] Values in `scripts/real_train.py` match the session log.
- [ ] Output folder created, new, and outside any synced folder like OneDrive or iCloud. The GPU logger writes into it before `train()` starts, so make it first. `train()` refuses a folder that already holds a run.

```bash
python -c "import os; os.makedirs(os.path.expanduser('~/pong-runs/my-run'))"
```

Swap `my-run` for your run's name, here and in every command below. It works the same in any shell, and it fails if the folder already exists, which is the point.

- [ ] Enough free disk space for the checkpoints. They're 27 MB each, so about 1.1 GB for 2,000,000 steps and 3.3 GB for 6,000,000.
- [ ] GPU logger started in a second terminal, writing into the same folder you'll pass to `--output-dir`.
  - **Windows:** run this in PowerShell.
  - **Linux:** run the bash version below.
  - **Mac:** skip it, along with the temperature checks below. A Mac has no NVIDIA GPU, so `train()` runs on the CPU, and a full run will take far longer than the 6.88 hours Run 2 took on the RTX 2060.

```powershell
# Windows
while ($true) { nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader | Add-Content $HOME/pong-runs/my-run/gpu_log.csv; Start-Sleep 30 }
```

```bash
# Linux
while true; do nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv,noheader >> ~/pong-runs/my-run/gpu_log.csv; sleep 30; done
```

The file has no header row, so the columns are the ones in the query, in that order.
It's a loop rather than the simpler `nvidia-smi -l 30 -f`, which holds its output
until it's stopped, so the log stays empty while the run is going.

Nothing in `train()` reads the temperature, so this log is the only record of it, and
nothing acts on it. The card protects itself, dropping its clocks at 90 °C and
shutting down at 93 °C, so the 88 °C limit below is there to explain a slow run
afterward, not to protect the hardware.

## Launch

On Windows or Linux:

```bash
python scripts/real_train.py --total-steps 6000000 --output-dir ~/pong-runs/my-run
```

On a Mac, the same command with `caffeinate -i` at the start, which keeps the Mac awake until the run ends:

```bash
caffeinate -i python scripts/real_train.py --total-steps 6000000 --output-dir ~/pong-runs/my-run
```

`--total-steps` is how long the run trains. 6,000,000 is what Run 2 used, and it took 6.88 hours on the RTX 2060. A smaller number gives a shorter run. Leave it out and the run is 2,000,000 steps, Run 1's length.

## First five minutes

- [ ] The `Output directory:` line near the top of the output is the folder you made, the same one `gpu_log.csv` is going into.
- [ ] The startup block matches the config: your GPU named, or the CPU on a Mac, sticky actions `0.0`, six action meanings, output size 6, and the seed you set.
- [ ] The `git_commit` in `run_config.json` matches `HEAD`. Record it in the day's entry.
- [ ] Steps per second after the buffer fills, not before. Only that rate predicts the full run. `python scripts/analyze_run.py ~/pong-runs/my-run` prints it as the post warmup rate, and it works on a run that's still going. The first 50,000 steps run several times faster, because no gradient updates happen yet, so their rate says nothing about the full run.
- [ ] At that rate, the run finishes when you need it to. Total steps divided by the rate is the run's length in seconds. For Run 2, 6,000,000 at 240.7 steps per second is about 6.9 hours.
- [ ] `gpu_log.csv` has rows in it, and the temperature is below 88 °C. Too hot means five readings in a row above 88, at the 30 second interval, not one spike.
  - 88 is the top of the card's intended range. It aims for 83 under sustained load, so a lower limit would trip on a healthy card, and 90 is where it throttles itself.
  - A falling clock only means throttling when the temperature is near the limit. During the Run 1 warmup pass, the clock swung from 1,290 to 795 MHz with the temperature in the forties and fifties. That was just the card idling down for lack of work.
- [ ] `episodes.csv` has rows in it, confirmed by opening the file.
- [ ] `metrics.csv` has its step 0 row, and `run_config.json` records the probe's sha256. For a run meant to be compared with Run 2, that sha256 should match the one in run-02.md.
- [ ] `step_00000000.pt` written. It's saved at startup, before any training, so it can't reveal a training problem. It only shows that saving works.
- [ ] `step_00050000.pt` written, the first scheduled save. The first 50,000 steps take about a minute, so it should land well within the first five minutes. Check it's around 27 MB, not 13.5 MB. The loop updates before it saves, so Adam's state should be in the file. 13.5 MB would mean it saved before any update, which proves nothing step 0 didn't.

## Once the buffer is full, around step 100,000

- [ ] `gpu_log.csv` shows the clock holding under sustained load. By now the card has reached its steady temperature, which is when throttling would start.
- [ ] `episodes.csv` and `metrics.csv` still growing, and checkpoints still landing.
- [ ] The finish estimate revised from the steady rate. If the rate has dropped from the post warmup figure, check RAM for paging before trusting the new estimate.
