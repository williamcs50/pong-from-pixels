# Run 1

The first real training run. Every figure here except the checkpoint count and size
comes from the run's own `run_config.json`, `episodes.csv`, `gpu_log.csv` and the two
evaluation files, all committed in [run-01/](run-01/) alongside the warmup pass's
three files in [run-00-warmup/](run-00-warmup/), so the numbers can be checked rather
than taken on trust. The commands that produce them are at the bottom. The checkpoints themselves
are not committed at 1.02 GB, so their count and size are measured on disk instead.

**Output:** `run-01` under `RUNS_ROOT`, which [src/run_paths.py](../../src/run_paths.py)
resolves to `~/pong-runs`, outside the OneDrive sync root.

**Commit:** `afb91fae0b1c1408009cb5385980f432c99f7ec7`, with `git_dirty` false, so
the run launched from committed code. The warmup pass ran from
`f9ce3b66d9a8c41105676bba87ae8453d4e9cb82`, and the two differ in documentation
only, so everything the warmup verified applies to this run unchanged. Check with
`git diff --stat f9ce3b6 afb91fa`.

**Started:** 2026-10-08 13:55:35. **Wall clock:** 2.19 hours.

---

## Config

| | |
|---|---|
| Total steps | 2,000,000 |
| Buffer capacity | 100,000 |
| Warmup steps | 50,000 |
| Train every | 4 env steps |
| Batch size | 32 |
| Target sync | every 10,000 gradient updates |
| Epsilon | 1.0 to 0.01 over 250,000 steps, counted from step 0 |
| Learning rate | 1e-4, Adam |
| Gamma | 0.99 |
| Sticky actions | 0.0, read back from the emulator |
| Seed | 0 |
| Checkpoints | step 0 plus every 50,000 steps |
| Device | NVIDIA GeForce RTX 2060 |
| Action space | 6 |

The reasoning behind each value is in the 2026-10-06 entry of
[sessions.md](../sessions.md).

---

## Result

**The agent did not reach the bar.** The trailing mean over the last 100 episodes
never exceeded 0. Its best and final value are both -0.11, at step 1,999,498.

| Step | Trailing 100 mean |
|---|---|
| 200,000 | -20.55 |
| 400,000 | -20.77 |
| 600,000 | -20.40 |
| 800,000 | -20.15 |
| 1,000,000 | -15.69 |
| 1,200,000 | -10.17 |
| 1,400,000 | -6.96 |
| 1,600,000 | -4.94 |
| 1,800,000 | -2.67 |
| 1,999,498 | -0.11 |

Each row is the nearest episode boundary to that step, since the trailing mean only
exists where an episode ended. The last row is the run's final boundary, 502 steps
short of 2,000,000.

1,293 episodes, 487,375 gradient updates at the last logged episode. These are
training episodes at epsilon 0.01, not the 100 episode evaluation the bar is
defined over. The two will be close at this epsilon but they are not the same
measurement.

**My pre registered prediction failed.** On 2026-10-06 I predicted the trailing
mean would cross 0 before step 1.2M. At 1.2M it was -10.17, and by the terms I
wrote there, still being below 0 at 1.2M means the schedule is insufficient.

**My pre run estimate also failed.** Before reading the rows I predicted a
positive end of run reward around +10. Actual -0.11.

**The curve was still climbing at the final step.** It gained 2.56 points across
the last 200,000 steps and shows no flattening, so by my own plateau definition, a
trailing mean moving less than 1.0 across 200 episodes, this run never plateaued.

**Supporting evidence that the learning is real rather than noise:**

- Mean episode length went from 909 steps over the first 100 episodes to 4,139
  over the last 100, so rallies are four and a half times longer.
- Zero shutouts in the last 100 episodes, against a start where every episode was
  a 21 to 0 or 21 to 2 loss.
- Mean loss per episode fell from 0.012179 to 0.001716, staying finite throughout,
  ranging 0.000189 to 0.013204.

Two candidate explanations for stopping short: the frame budget, 8,000,000 frames
against Nature's 50,000,000, and the replay buffer at 100,000 against Nature's
1,000,000. The curve's shape favours the frame budget, since a buffer limitation
would tend to flatten the curve rather than leave it climbing.

---

## Evaluation

The figures above are training episodes at epsilon 0.01. The bar is defined over
100 evaluation episodes, so every checkpoint was scored separately at epsilon 0.05,
in [run-01/eval.csv](run-01/eval.csv).

| Step | Eval mean | Training trailing 100 | Eval minus training |
|---|---|---|---|
| 0 | -20.98 | | |
| 200,000 | -21.00 | -20.55 | -0.45 |
| 400,000 | -20.75 | -20.77 | +0.02 |
| 600,000 | -20.96 | -20.40 | -0.56 |
| 800,000 | -18.47 | -20.15 | +1.68 |
| 1,000,000 | -10.69 | -15.69 | +5.00 |
| 1,200,000 | -11.79 | -10.17 | -1.62 |
| 1,400,000 | -7.79 | -6.96 | -0.83 |
| 1,600,000 | -1.99 | -4.94 | +2.95 |
| 1,800,000 | **-1.63** | -2.67 | +1.04 |
| 2,000,000 | -4.41 | -0.11 | -4.30 |

**No checkpoint meets the bar.** The best is -1.63 at step 1,800,000.

**The best checkpoint is not the last one.** 1,800,000 beats 2,000,000 by 2.78
points, so the policy got worse over the final 200,000 steps. Both were scored again
at seed 1 on independent episodes, in
[run-01/eval_seed1.csv](run-01/eval_seed1.csv), and the result held:

| Step | Seed 0 | Seed 1 | Std at seed 1 |
|---|---|---|---|
| 0 | -20.98 | -20.97 | 0.17 |
| 1,800,000 | -1.63 | -1.89 | 4.43 |
| 2,000,000 | -4.41 | -4.54 | 3.16 |

With 100 episodes those standard deviations give standard errors of 0.44 and 0.32,
so the 2.65 point gap at seed 1 is about 5 combined standard errors. The means also
moved by only 0.26 and 0.13 between seeds, well inside those errors. So this is
checkpoint oscillation rather than evaluation noise, and 1,800,000 is genuinely the
better policy rather than the luckier measurement.

**2,000,000 is both worse and less variable**, standard deviation 3.16 against
4.43. It loses more consistently, where 1,800,000 sometimes wins by more.

**The two curves cross, and both lag effects show up.** Evaluation runs above
training from 800,000 to 1,600,000, peaking at 5.00 points at 1,000,000, then falls
4.30 below it at the end. The training trailing mean covers roughly the last
400,000 steps, so it averages a changing policy rather than measuring the final
one: while the policy is improving that understates it, and at the end, where the
policy regressed, it overstates it. The extra exploration at epsilon 0.05 pushes
evaluation down throughout.

That is why training reported -0.11 at the final step while the final policy scores
-4.41. The number I quoted as Run 1's result was never a measurement of the policy
the run finished with.

**The eleven rows in `eval.csv` predate the provenance work**, so they carry no
commit hash, no standard deviation, and no seed column. They were produced at seed
0. The seed 1 rows have a full sidecar in
[run-01/eval_seed1_config.json](run-01/eval_seed1_config.json).

---

## Throughput and hardware

| | |
|---|---|
| Overall rate | 253.3 steps per second |
| Post warmup rate, this run | 248.9 steps per second |
| Post warmup rate, measured on the warmup pass | 252.1 steps per second |
| Projected duration from that rate | 2.20 hours |
| Actual duration | 2.19 hours |
| Temperature | 47 to 62 °C, limit 88, zero readings above it |
| SM clock | 300 to 1,770 MHz |
| Power draw | 9.6 to 51.3 W |
| GPU utilization | 3 to 42 percent |
| VRAM used | 1,154 to 2,266 MiB of 6,144 |
| Checkpoints | 41 files, 1.02 GB, measured on disk |

**The GPU was not the bottleneck.** 42 percent peak utilization and 51 W on a card
rated near 160 means the limit is the CPU side: one Pong instance, frame
preprocessing, and the per step Python loop. The fill phase before training began
ran at 809 steps per second with no GPU work in it, against 252 once updates
started.

The warmup pass understated the load, peaking at 52 °C, 1,290 MHz and 30 W,
because it never ran long enough to reach steady state. The limits were still
nowhere near reached.

---

## Reproducing the figures

Every training and hardware number above comes out of one command, run from the
repository root:

```powershell
python scripts/analyze_run.py docs/runs/run-01
```

The evaluation numbers are read straight out of [run-01/eval.csv](run-01/eval.csv)
and [run-01/eval_seed1.csv](run-01/eval_seed1.csv). Reproducing them from the
checkpoints, which are not committed, takes about 50 minutes for the full sweep:

```powershell
python scripts/sweep_eval.py <run_dir> --every 200000
```

The warmup pass's own figures, including the 252.1 steps per second the projection
was built on, come from the analyzer pointed at the other directory:

```powershell
python scripts/analyze_run.py docs/runs/run-00-warmup
```
