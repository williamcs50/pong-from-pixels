# Run 1

My first real training run. Every number here comes from the run's own files, which
are committed in [run-01/](run-01/): `run_config.json`, `episodes.csv`, `gpu_log.csv`
and the two evaluation files. The warmup pass's three files are in
[run-00-warmup/](run-00-warmup/). The commands that produce every number are at the
bottom. The checkpoints aren't committed, at 1.02 GiB (1.09 GB), so I measured their
count and size on disk.

**Output:** `run-01` under `RUNS_ROOT`, which [src/run_paths.py](../../src/run_paths.py)
resolves to `~/pong-runs`, outside the OneDrive sync root.

**Commit:** `afb91fae0b1c1408009cb5385980f432c99f7ec7`, with `git_dirty` false, so the
run launched from committed code. The warmup pass ran from
`f9ce3b66d9a8c41105676bba87ae8453d4e9cb82`. The two only differ in documentation, so
everything the warmup verified still applies to this run. You can check with
`git diff --stat f9ce3b6 afb91fa`.

**Started:** 2026-10-08 13:55:35. **Wall clock:** 2.19 hours.

---

## Pre registration

Quoted verbatim from the 2026-10-06 entry in [sessions.md](../sessions.md), commit
`03b99133acc6ef509f60952baf9b229f24ee224c` of Oct 6 at 18:10. I copied it into this
file on Oct 10. That commit is two days before Run 1 launched on Oct 8 at 13:55,
which is what proves the prediction came before the data.

Three items, as I wrote them then rather than how I'd write them now:

> Rewrote the pre registered prediction so it can fail. A plateau is the trailing
> mean over the last 100 episodes moving less than 1.0 across 200 episodes. It
> should cross 0 before step 1.2M, and still being below 0 at 1.2M means the
> schedule is insufficient.

| # | Prediction | Outcome |
|---|---|---|
| 1 | The trailing 100 mean crosses 0 before step 1,200,000 | Failed. It was -10.17 at 1,200,000 and never crossed at all, ending at -0.11. |
| 2 | A plateau is the trailing mean moving less than 1.0 across 200 episodes | No plateau. It gained 2.56 points over the last 200,000 steps with no flattening. |
| 3 | Still being below 0 at 1,200,000 means the schedule is insufficient | Triggered, which is what sets Run 2's step count. |

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

My reasoning for each value is in the 2026-10-06 entry of
[sessions.md](../sessions.md).

---

## Result

**The agent didn't reach the bar.** The trailing mean over the last 100 episodes never
got above 0. Its best and final value are both -0.11, at step 1,999,498.

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

The trailing mean only exists where an episode ended, so each row is the nearest
episode boundary to that step. The last row is the run's final boundary, 502 steps
short of 2,000,000.

The run had 1,293 episodes and 487,375 gradient updates by the last logged episode.
These are training episodes at epsilon 0.01, not the 100 episode evaluation the bar is
defined over. The two should be close at this epsilon, but they aren't the same
measurement.

**My pre registered prediction failed.** On 2026-10-06 I predicted the trailing mean
would cross 0 before step 1.2M. At 1.2M it was -10.17. By the terms I wrote, still
being below 0 at 1.2M means the schedule is insufficient.

**My estimate before the run failed too.** Before I looked at any rows, I predicted
the run would end with positive reward, around +10. It ended at -0.11.

**The curve was still climbing at the last step.** It gained 2.56 points over the
last 200,000 steps with no sign of flattening. My plateau definition is a trailing mean
moving less than 1.0 across 200 episodes, and by that definition it never plateaued.

**Supporting evidence that the learning is real, not noise:**

- Mean episode length went from 909 steps over the first 100 episodes to 4,139 over
  the last 100. Rallies got four and a half times longer.
- There were zero shutouts in the last 100 episodes. At the start, every episode was a
  21 to 0 or 21 to 2 loss.
- Mean loss per episode fell from 0.012179 to 0.001716. It stayed finite the whole run,
  ranging 0.000189 to 0.013204.

I can think of two reasons it stopped short. One is the frame budget: 8,000,000 frames
against Nature's 50,000,000. The other is the replay buffer, at 100,000 against
Nature's 1,000,000. The shape of the curve points to the frame budget. A buffer that's
too small would tend to flatten the curve, not leave it still climbing.

---

## Evaluation

Everything above is training episodes at epsilon 0.01. The bar is defined over 100
evaluation episodes, so I scored every checkpoint separately at epsilon 0.05, in
[run-01/eval.csv](run-01/eval.csv).

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

**No checkpoint reaches the bar.** The best is -1.63 at step 1,800,000.

**The best checkpoint isn't the last one.** 1,800,000 beats 2,000,000 by 2.78 points,
so the policy actually got worse over the last 200,000 steps. I scored both again at
seed 1 on independent episodes, in [run-01/eval_seed1.csv](run-01/eval_seed1.csv), and
the result held:

| Step | Seed 0 | Seed 1 | Std at seed 1 |
|---|---|---|---|
| 0 | -20.98 | -20.97 | 0.17 |
| 1,800,000 | -1.63 | -1.89 | 4.43 |
| 2,000,000 | -4.41 | -4.54 | 3.16 |

With 100 episodes, those standard deviations give standard errors of 0.44 and 0.32.
So the 2.65 point gap at seed 1 is about 5 combined standard errors. The means also
only moved by 0.26 and 0.13 between seeds, well inside those errors. So this is the
checkpoints actually oscillating, not evaluation noise, and 1,800,000 really is the
better policy, not just the luckier measurement.

**2,000,000 is worse and also less variable**, with a standard deviation of 3.16
against 4.43. It loses more consistently, while 1,800,000 sometimes wins by more.

**The two curves cross, and you can see the lag both ways.** Evaluation sits above
training from 800,000 to 1,600,000, by as much as 5.00 points at 1,000,000, then ends
4.30 below it. The training trailing mean covers roughly the last 400,000 steps, so it
averages a policy that's changing instead of measuring the final one. While the policy
is improving, that makes it look worse than it is. At the end, where the policy got
worse, it makes it look better. The extra exploration at epsilon 0.05 also pulls
evaluation down the whole way.

That's why training said -0.11 at the final step while the final policy scores -4.41.
The number I quoted as Run 1's result never actually measured the policy the run ended
with.

**The eleven rows in `eval.csv` came before I added provenance tracking.** So they
have no commit hash, no standard deviation and no seed column. They were produced at
seed 0. The seed 1 rows have a full sidecar in
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
| Checkpoints | 41 files, 1.02 GiB (1.09 GB), measured on disk |

**The GPU wasn't the bottleneck.** It never went above 42 percent utilization or 51 W,
on a card rated around 160. So the limit is on the CPU side: one Pong instance, frame
preprocessing, and the Python loop running every step. The fill phase before training
ran at 837.3 steps per second with no GPU work at all, against 248.9 once updates
started.

The warmup pass understated the load, peaking at 52 °C, 1,290 MHz and 30 W, because it
never ran long enough to warm up fully. Even so, nothing got close to the limits.

---

## Reproducing the figures

Every training, evaluation and hardware number above comes out of one command, run
from the repository root:

```powershell
python scripts/analyze_run.py docs/runs/run-01
```

The one thing this repo can't do is rerun the sweep itself. `sweep_eval.py` needs a
`checkpoints/` folder inside the run directory you give it, and those 1.02 GiB aren't
committed, so it only works on the machine that produced the run. The sweep takes about
50 minutes. `eval.csv` was written before the sweep's current columns, and the sweep
refuses to append to a file with different columns, so a rerun needs a new `--out`:

```powershell
python scripts/sweep_eval.py $HOME\pong-runs\run-01 --every 200000 --out $HOME\pong-runs\run-01\eval_rerun.csv
```

In PowerShell, use `$HOME` instead of `~`. PowerShell passes `~` to the script as is,
and the script doesn't expand it.

The warmup pass's own figures, including the 252.1 steps per second my projection
used, come from the analyzer pointed at the other directory:

```powershell
python scripts/analyze_run.py docs/runs/run-00-warmup
```
