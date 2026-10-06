# Launch Checklist

Only the checks the code cannot do for itself. Measured values go in that day's
[sessions.md](sessions.md) entry, so runs can be compared. Expect this to grow
once Run 1 shows what it missed.

---

## Before launch

- [ ] Sleep set to "never" on AC power, and Windows Update paused for the run.
- [ ] Laptop plugged in. On battery the GPU throttles.
- [ ] `CUDA_VISIBLE_DEVICES` not set. Setting it to `-1` for a CPU check silently sends every later run to the CPU.
- [ ] `git status` clean, so the run launches from a committed state and nothing is left commented out from a mutation. Record `git rev-parse --short HEAD` in the day's entry.
- [ ] `python scripts/check_environment.py` passes. Host RAM and VRAM in one command.
- [ ] Test suite passing.
- [ ] Values in `scripts/real_train.py` match the session log.
- [ ] Output directory exists, writable, and outside OneDrive.
- [ ] Free disk space checked against the checkpoint schedule.
- [ ] GPU logger started in a second terminal, writing to the output directory:

```powershell
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,clocks.sm --format=csv -l 30 -f <output_dir>\gpu_log.csv
```

Use `-f` rather than `>`. PowerShell's redirect writes UTF-16, which makes the
CSV awkward to read later. Logging clock speed alongside temperature is what
lets you spot throttling, since the clock drops when the card gets hot.

## First five minutes

- [ ] Startup block matches the config: GPU and the 2060 named, sticky actions `0.0`, six action meanings, output size 6, seed as configured.
- [ ] Steps per second after warmup, not during it. Only that rate predicts the full run.
- [ ] Projected wall clock from that rate lands before the diagnosis date.
- [ ] `gpu_log.csv` has rows in it, temperature below ___ °C, and clock not dropping as temperature rises.
- [ ] Reward log on disk has rows in it, confirmed by opening the file.
- [ ] First checkpoint written.

## Once the buffer is full, around step 100k

- [ ] Buffer memory in Task Manager against the expected 5.26 GiB. Windows commits pages lazily, so this is the first honest reading.
- [ ] `gpu_log.csv` shows the clock holding after sustained load. By now the card has reached its steady temperature, which is when throttling would start.
- [ ] Reward log still growing and checkpoints still landing.
- [ ] Wall clock estimate revised from the steady state rate. If the rate has dropped from the post warmup figure, check RAM for paging before trusting the new estimate.
