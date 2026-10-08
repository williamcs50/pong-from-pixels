# Launch Checklist

Only the checks the code cannot do for itself. Measured values go in that day's
[sessions.md](sessions.md) entry, so runs can be compared. Expect this to grow
once Run 1 shows what it missed.

---

## Before launch

- [ ] Sleep set to "never" on AC power, and Windows Update paused for the run.
- [ ] `CUDA_VISIBLE_DEVICES` not set. Setting it to `-1` for a CPU check silently sends every later run to the CPU.
- [ ] `git status` clean. `train()` refuses to start on a dirty tree, so this is enforced rather than remembered. Check that the `git_commit` in `run_config.json` matches `HEAD`, and record it in the day's entry.
- [ ] `python scripts/check_environment.py` passes. Host RAM and VRAM in one command.
- [ ] Test suite passing.
- [ ] Values in `scripts/real_train.py` match the session log.
- [ ] Output directory exists, writable, and outside OneDrive.
- [ ] Free disk space checked against the checkpoint schedule.
- [ ] GPU logger started in a second terminal, writing into the same directory passed to `--output-dir`. Pass that flag explicitly rather than letting the numbering choose, so the path is known before the run starts:

```powershell
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -l 30 -f <output_dir>\gpu_log.csv
```

Use `-f` rather than `>`. PowerShell's redirect writes UTF-16, which makes the
CSV awkward to read later. Logging clock speed alongside temperature is what
lets you spot throttling, since the clock drops when the card gets hot.

Nothing in `train()` reads temperature, so this log is the only thermal record and
it is passive. The card protects itself at 90 °C by dropping its clocks and at
93 °C by shutting down, so the limit below exists to explain a slow run
afterwards, not to save the hardware.

## First five minutes

- [ ] Startup block matches the config: GPU and the 2060 named, sticky actions `0.0`, six action meanings, output size 6, seed as configured.
- [ ] Steps per second after warmup, not during it. Only that rate predicts the full run. Nothing prints it, so compute it from two `episodes.csv` rows past the warmup step: the difference in `env_step` over the difference in `wall_clock_s`. Use rows a few minutes apart rather than adjacent ones, since `wall_clock_s` is recorded to a tenth of a second and adjacent episodes make the rate noisy. Rows before the warmup step are the fill phase, which runs several times faster because no gradient updates happen.
- [ ] Projected wall clock from that rate lands before the diagnosis date.
- [ ] `gpu_log.csv` has rows in it, temperature below 88 °C, and clock not dropping as temperature rises. Sustained means five consecutive readings above 88 at the 30 second interval, not one spike. 88 is the top of the card's intended range: it targets 83 under sustained load, so a lower limit would trip on a healthy card, and 90 is where it throttles itself.
- [ ] Reward log on disk has rows in it, confirmed by opening the file.
- [ ] `step_00000000.pt` written. This one is saved at startup before any training, so it cannot fail and is a sanity check only.
- [ ] `step_00050000.pt` written, which is the first scheduled save. At the fill rate the warmup takes about a minute, so it should land well inside five. Check it is around 27 MB rather than 13.5 MB: the loop updates before it saves, so Adam state should be in the file, and 13.5 MB would mean it saved before any update and proves nothing step 0 did not.

## Once the buffer is full, around step 100k

- [ ] Buffer memory in Task Manager against the expected 5.26 GiB. Windows commits pages lazily, so this is the first honest reading.
- [ ] `gpu_log.csv` shows the clock holding after sustained load. By now the card has reached its steady temperature, which is when throttling would start.
- [ ] Reward log still growing and checkpoints still landing.
- [ ] Wall clock estimate revised from the steady state rate. If the rate has dropped from the post warmup figure, check RAM for paging before trusting the new estimate.
