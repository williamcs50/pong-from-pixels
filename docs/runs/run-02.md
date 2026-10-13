# Run 2

Every figure here except the checkpoint count and size comes from the run's own
`run_config.json`, `episodes.csv`, `metrics.csv`, `gpu_log.csv` and the two
evaluation files, all eight committed in [run-02/](run-02/) alongside the warmup
pass's three files in [run-02-warmup/](run-02-warmup/), so every number here can be
checked. The commands that produce them are at the bottom. The checkpoints themselves are not committed at 3.03 GiB (3.26 GB), so their count and
size are measured on disk instead. The 500 state probe is not committed either, at
13.5 MB, because `run_config.json` records its sha256 and it regenerates from
`PROBE_SEED`.

**Output:** `run-02` under `RUNS_ROOT`, which [src/run_paths.py](../../src/run_paths.py)
resolves to `~/pong-runs`, outside the OneDrive sync root.

**Commit:** `1e407be98054234feefd5dcbdfd87291515d2a94`, with `git_dirty` false, so
the run launched from committed code. That is the same commit that recorded the pre
registration below, made Oct 10 at 21:31:53, which is 14 hours and 17 minutes
before the run started.

The warmup pass ran from `f03f1410977e553ea26d0d0841f750160eb60aa8`. Unlike Run 1's,
the two are not documentation only: `real_train.py` gained a `--probe` flag the run
never used, `probe.py` lost an unused `import os`, and `analyze_run.py` gained a
guard for a missing GPU log. None of it touches training, and both runs recorded
the same probe sha256, which is evidence rather than an argument from the diff.
Check with `git diff --stat f03f141 1e407be`.

**Started:** 2026-10-11 11:49:22. **Wall clock:** 6.88 hours.

---

## Pre registration

Written 2026-10-10, before launch, and committed before any Run 2 data exists.

**Planned config:** identical to Run 1 except `total_steps` of 6,000,000 and the
instrumentation, meaning `metrics.csv` every 10,000 env steps and a probe of 500
states with its sha256 in `run_config.json`.

| # | Prediction | Why | Outcome |
|---|---|---|---|
| 1 | The trailing 100 mean crosses 0 before step 2,300,000 | Run 1 gained 2.02, 2.27 and 2.56 points over its last three 200,000 step windows and ended at -0.11 at 2,000,000. Same config and seed, so Run 2 should track it to 2,000,000 and cross shortly after. | Failed. It crossed at 2,715,875, missing by 415,875 steps. |
| 2 | A plateau is the trailing mean moving less than 1.0 across 200,000 env steps | Restated from Run 1's "200 episodes", which spanned roughly four times as many steps late in the run as early, so the same words were a looser test at the end. Steps are comparable across the whole run. | Fired falsely. From 3,600,000 to 3,800,000 the trailing mean stayed within 0.94, under the 1.0 threshold, while evaluation moved 8.80 at seed 0 and 9.51 at seed 1, so the definition reported stability where the policy was unstable. It fires in 12 of the 30 windows between evaluated checkpoints, counting either the change between endpoints or the range inside, including 1,800,000 to 2,000,000, where evaluation fell 3.40. |
| 3 | At least one checkpoint clears a mean above 0 over 100 evaluation episodes | Run 1's best was -1.63 at 1,800,000 and no checkpoint cleared the bar. Three times the steps should be enough if the frame budget was the binding constraint. | Passed. 17 of 31 checkpoints at seed 0 and 18 at seed 1, with the best at +16.97 at step 5,400,000. |

**What each outcome means:**

- **Crosses 0 and a checkpoint clears the bar.** The frame budget was the binding constraint and Run 1 ran out of steps. The remaining question is how far above the bar it goes, not whether it gets there.
- **Crosses 0 but no checkpoint clears the bar.** The training curve and the evaluation curve disagree the way they did in Run 1, where -0.11 training was -4.41 on evaluation. That would mean the gap is structural rather than a Run 1 accident, and the bar needs to be chased on evaluation rather than on training reward.
- **Never crosses 0.** Two failed predictions at three times the step count means the frame budget explanation is wrong. That sends me back to the learning rate, the replay buffer at 100,000 against Nature's 1,000,000, and the target sync interval rather than to a longer run.
- **Plateaus by the definition above.** Whatever value it plateaus at is the ceiling for this config, and Run 3 changes one hyperparameter rather than adding steps.

---

## Config

| | |
|---|---|
| Total steps | 6,000,000 |
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
| Metrics | step 0 plus every 10,000 steps |
| Probe | 500 states, sha256 `ec446d0b2dd8510ef2a7dcb84f657a28623dca6e631feb56249dd0bea3fa0c4a` |
| Device | NVIDIA GeForce RTX 2060 |
| Action space | 6 |

Every value is Run 1's except `total_steps`, which Run 1's result set, and the two
instrumentation rows. The reasoning behind the shared values is in the 2026-10-06
entry of [sessions.md](../sessions.md), and the reasoning behind the
instrumentation is in the 2026-10-09 entry.

---

## Results

**The agent reached the bar.** The trailing mean over the last 100 episodes crossed 0 at step 2,715,875. Its best value was +19.02 and its final value was +16.80, at step 5,998,105. All four figures are derived from `run-02/episodes.csv`.

| Step | Trailing 100 mean |
|---|---|
| 200,000 | -20.65 |
| 400,000 | -20.88 |
| 600,000 | -20.91 |
| 800,000 | -19.52 |
| 1,000,000 | -17.28 |
| 1,200,000 | -15.88 |
| 1,400,000 | -14.63 |
| 1,600,000 | -13.24 |
| 1,800,000 | -12.06 |
| 2,000,000 | -11.43 |
| 2,200,000 | -9.93 |
| 2,400,000 | -7.30 |
| 2,600,000 | -3.44 |
| 2,800,000 | +2.51 |
| 3,000,000 | +6.03 |
| 3,200,000 | +9.26 |
| 3,400,000 | +13.73 |
| 3,600,000 | +13.22 |
| 3,800,000 | +13.77 |
| 4,000,000 | +14.09 |
| 4,200,000 | +15.51 |
| 4,400,000 | +16.64 |
| 4,600,000 | +11.65 |
| 4,800,000 | +15.15 |
| 5,000,000 | +17.52 |
| 5,200,000 | +17.77 |
| 5,400,000 | +18.69 |
| 5,600,000 | +18.67 |
| 5,800,000 | +17.75 |
| 5,998,105 | +16.80 |

Each row is the nearest episode boundary to that step, since the trailing mean only
exists where an episode ended. The last row is the run's final boundary, 1,895
steps short of 6,000,000.

2,495 episodes, 1,487,027 gradient updates at the last logged episode. 2,212 of
those episodes ran at epsilon 0.01, the floor the schedule reaches at step 250,000.
The bar is defined over 100 evaluation episodes at epsilon 0.05, and the two are
not interchangeable: training finished at +16.80 while the final checkpoint scores
+12.60.

**My pre registered prediction failed.** On 2026-10-10 I predicted the trailing
mean would cross 0 before step 2,300,000. It crossed at 2,715,875, missing by
415,875 steps.

**Its premise was wrong, which matters more.** I assumed Run 2 would track Run 1 to
2,000,000 because the config and the seed are identical. At 2,000,000 Run 1 was at
-0.11 and Run 2 at -11.43, a gap of 11.32 points. The two runs are identical in every
column but wall clock time for 54 episodes and first differ at episode 55, the first one spanning step 50,000
where updates begin, so a fixed seed does not reproduce a run on this hardware.

**The curve did not end at its best.** It peaked at +19.02 and finished 2.22 points
lower at +16.80, where Run 1 ended at its own maximum.

**Supporting evidence that the learning is real rather than noise:**

- Mean loss per episode fell from 0.012040 to 0.000986, staying finite throughout,
  ranging 0.000194 to 0.013529.
- Zero shutouts in the last 100 episodes, though Run 1 reported zero too, so this
  no longer separates a losing agent from a winning one.

---

## Evaluation

The figures above are training episodes at epsilon 0.01. The bar is defined over
100 evaluation episodes, so every 200,000th checkpoint was scored separately at
epsilon 0.05, twice, at two independent evaluation seeds, in
[run-02/eval.csv](run-02/eval.csv) and
[run-02/eval_seed1.csv](run-02/eval_seed1.csv).

| Step | Seed 0 | Seed 1 | Training trailing 100 | Seed 0 minus training |
|---|---|---|---|---|
| 0 | -20.98 | -20.97 | | |
| 200,000 | -21.00 | -21.00 | -20.65 | -0.35 |
| 400,000 | -21.00 | -21.00 | -20.88 | -0.12 |
| 600,000 | -20.68 | -20.59 | -20.91 | +0.23 |
| 800,000 | -16.13 | -16.49 | -19.52 | +3.39 |
| 1,000,000 | -17.16 | -17.05 | -17.28 | +0.12 |
| 1,200,000 | -15.40 | -15.27 | -15.88 | +0.48 |
| 1,400,000 | -12.77 | -12.60 | -14.63 | +1.86 |
| 1,600,000 | -13.38 | -13.53 | -13.24 | -0.14 |
| 1,800,000 | -10.86 | -10.73 | -12.06 | +1.20 |
| 2,000,000 | -14.26 | -14.20 | -11.43 | -2.83 |
| 2,200,000 | -10.35 | -11.08 | -9.93 | -0.42 |
| 2,400,000 | -1.17 | -1.23 | -7.30 | +6.13 |
| 2,600,000 | -0.81 | +0.29 | -3.44 | +2.63 |
| 2,800,000 | +5.45 | +5.13 | +2.51 | +2.94 |
| 3,000,000 | +7.22 | +7.75 | +6.03 | +1.19 |
| 3,200,000 | +12.78 | +12.98 | +9.26 | +3.52 |
| 3,400,000 | +7.88 | +8.25 | +13.73 | -5.85 |
| 3,600,000 | +6.19 | +5.88 | +13.22 | -7.03 |
| 3,800,000 | +14.99 | +15.39 | +13.77 | +1.22 |
| 4,000,000 | +11.82 | +11.64 | +14.09 | -2.27 |
| 4,200,000 | +13.66 | +13.18 | +15.51 | -1.85 |
| 4,400,000 | +14.60 | +13.90 | +16.64 | -2.04 |
| 4,600,000 | +7.23 | +7.13 | +11.65 | -4.42 |
| 4,800,000 | +12.14 | +12.37 | +15.15 | -3.01 |
| 5,000,000 | +15.40 | +14.85 | +17.52 | -2.12 |
| 5,200,000 | +15.56 | +14.91 | +17.77 | -2.21 |
| 5,400,000 | +16.97 | +16.67 | +18.69 | -1.72 |
| 5,600,000 | +15.96 | +15.80 | +18.67 | -2.71 |
| 5,800,000 | +15.00 | +14.87 | +17.75 | -2.75 |
| 6,000,000 | +12.60 | +12.18 | +16.80 | -4.20 |

**17 checkpoints clear the bar at seed 0, and 18 at seed 1.** The best is +16.97 at
step 5,400,000, which is also the best at seed 1 at +16.67.

**The best checkpoint is not the last one.** 5,400,000 beats 6,000,000 by 4.37
points, which against standard errors of 0.21 and 0.25 is 13.3 combined. At seed 1
the same comparison gives 4.49 points and 12.6. The standard error is
`std_reward / 10`, since the root of 100 episodes is 10.

**The two seeds agree everywhere.** Their mean absolute difference across all 31
checkpoints is 0.29 points, the largest single difference is 1.10 at 2,600,000, and
**none of the 31 differs by more than two standard errors**. So 100 episodes at
epsilon 0.05 ranks checkpoints reliably, and every feature of this curve is a
property of the policy rather than of the episodes it happened to be scored on.

**Two checkpoints collapse and recover, and both reproduce.** 4,600,000 scores
+7.23 and +7.13 between neighbours at +14.60 and +12.14, and 3,600,000 scores +6.19
and +5.88 between +7.88 and +14.99. These are the same dips that make the plateau
definition fire falsely, and they are 7 to 9 points deep.

**The training to evaluation gap is structural.** At the final step training read
+16.80 while the checkpoint scores +12.60 at seed 0 and +12.18 at seed 1, gaps of
-4.20 and -4.62. Run 1's equivalent gap was -4.30, at a reward level 17 points
lower. Two runs that ended that far apart producing the same gap means the headline
figure for any run has to come from a sweep rather than from the training log.

**Episode length measures how even the game is, not how good the agent is.** The
longest evaluation episodes in the sweep are at 2,600,000, averaging 5,299 steps
where the agent scores -0.81, which is the closest sampled point to the crossing.
The best checkpoint plays episodes of 2,444 steps, less than half that. run-01.md
offers its rise from 909 to 4,139 steps as evidence of learning, but Run 1 finished
near parity, which is exactly where this metric peaks. A stronger agent produces a
lower number, because a 21 to 3 win is shorter than a 21 to 19 grind.

**The +18 aspiration is not met.** [PLANS.md](../PLANS.md) sets the bar at a mean
above 0 and the aspiration at +18 or better. The best checkpoint is +16.97 plus or
minus 0.21, short by 1.03 points, which is about five standard errors.

---

## Instrumentation

**The measurement itself does not drift.** The probe is the same 500 states at
every row, so a change in either Q column can only come from the network. The
first five rows, steps 0 to 40,000, have zero gradient updates and identical
values to six decimals, `q_max_mean` 0.034242 and `q_spread_mean` 0.056905.

**`q_max_mean` fell before it rose.** It went below 0 at 140,000 and reached its
minimum of -0.2180 at 760,000. Rewards are +1 and -1 per point, so a negative value
means the network expected to lose points from these states. It was above 0 for
good from 1,570,000, peaked at 0.6686 at 5,980,000 and ended at 0.6521.

| Steps | Change in `q_max_mean` |
|---|---|
| 0 to 760,000 | -0.2522 |
| 760,000 to 3,000,000 | +0.5290 |
| 3,000,000 to 5,000,000 | +0.3346 |
| 5,000,000 to 6,000,000 | +0.0066 |

The last 1,000,000 steps added almost nothing, with every row between 0.6118 and
0.6686.

**`q_spread_mean` collapsed at the first updates, then grew.** The single update
logged at step 50,000 took it from 0.0569 to 0.0413, and by 60,000, 2,501 updates
in, it was 0.0050, its minimum, meaning the network rated all six actions almost
the same. It peaked at 0.2067 at 2,230,000, and from 3,000,000 on every row sat
between 0.1401 and 0.2065, averaging 0.1721, ending at 0.1538. These are random
policy states, as Limitations notes, so both columns describe positions the trained
agent seldom reaches.

**Did Q spread rise before reward?**

**Epsilon followed the schedule exactly.** `run_config.json` sets a linear decay
from 1.0 to 0.01 over 250,000 steps. Every metrics row after step 0 matches that
line to within 0.00000004, which is rounding at six decimals, and all 575 rows after
250,000 read exactly 0.01. The row at 250,000 reads 0.010004 because each row is
written at `env_step`, one past the loop index the epsilon was computed from, so it
shows the value used for the step before. The floor took effect at step 250,001,
and the first row that shows it is 260,000.

**Replay dynamics.**

**Paddle direction.**

---

## Throughput and hardware

| | |
|---|---|
| Overall rate | 242.2 steps per second |
| Post warmup rate | 240.7 steps per second |
| Fill phase rate, before any updates | 844.4 steps per second |
| Duration | 6.88 hours |
| Temperature | 40 to 62 °C, limit 88, zero readings above it |
| SM clock | 300 to 1,545 MHz |
| Power draw | 8.55 to 41.3 W |
| GPU utilization | 1 to 42 percent |
| VRAM used | 1,047 to 2,287 MiB of 6,144 |
| GPU samples | 827, spanning 6.89 hours |
| Checkpoints | 121 files, 3.03 GiB (3.26 GB), measured on disk |

**The GPU was not the bottleneck.** 42 percent peak utilization and 41.3 W on a
card rated near 160 means the limit is the CPU side: one Pong instance, frame
preprocessing, and the per step Python loop. The fill phase runs the same emulator
and preprocessing with no gradient updates in it and managed 844.4 steps per
second, against 240.7 once updates started.

**The logger covered the whole run.** Its 827 timestamps run from 11:48:51 to
18:42:33, starting 31 seconds before the run and continuing past its end, and the
gaps between consecutive samples range only 30.031 to 30.123 seconds, so nothing
was dropped. Run 1's first attempt produced a zero byte file because
`nvidia-smi -l 30 -f` buffered its output, so this is the check that the
`Add-Content` replacement writes continuously.

**The instrumentation's cost is below what the measurement resolves.** The
instrumented warmup in [run-02-warmup/](run-02-warmup/) ran at 256.1 steps per
second against the uninstrumented warmup in [run-00-warmup/](run-00-warmup/) at
252.1, so the instrumented one was faster, which is run to run variation rather
than a cost. The projection built on 256.1 was 6.47 hours for 6,000,000 steps, and
the run took 6.88. That 6.3 percent miss is the rate not holding, not the
arithmetic: 256.1 against the actual 240.7 is 6.4 percent, which is the same gap.

**Run 2 ran 3.3 percent slower than Run 1 and the logs do not say why.** 240.7
against 248.9 post warmup, while the fill phase was slightly faster at 844.4
against 837.3. The instrumentation is far too small to account for it: 601 probe
rows of 500 states is 300,500 forward passes, against roughly 95 million in the
training batches. Run 2 also ran cooler at a lower peak clock, 1,545 MHz against
1,770 and 41.3 W against 51.27, so whatever differed was on the CPU side.

---

## Clip selection

Written 2026-10-13 and committed before any rule below is applied.

**Checkpoints.** Each milestone is chosen by rule from the evaluated checkpoints at
seed 0:

| Milestone | Rule |
|---|---|
| Random | Step 0, the untrained network |
| First rally | The first evaluated checkpoint whose mean episode length beats step 0's by more than 2 combined standard errors |
| First over the bar | The first evaluated checkpoint whose mean reward is above 0 |
| Trained | The evaluated checkpoint with the highest mean reward, with its seed 1 score quoted next to it |

First rally uses length, not reward, because a rally is the ball coming back, and
the agent can return the ball for a while before it wins more points. The sweep
recorded only the mean episode length, so checkpoints are rescored from step 0, with
the spread of episode length recorded, until the rule fires. First over the bar is a
single checkpoint and can cross 0 on an oscillation. If Trained and First over the
bar are the same checkpoint, one clip covers both, and the case study's slider has
three positions, not four.

**Episode.** Each clip is the first episode at evaluation seed 0 and epsilon 0.05,
played through `evaluate()`, the same code that scored the sweep. Before recording,
that episode is played twice, and the clip only counts if both plays give identical
reward and length. If they differ, the clip is played on CPU and the same check is
applied there. If CPU also disagrees with itself, nothing is recorded until the cause
is found.

---

## Limitations

**The training run is a single seed.** The evaluation replicates at two seeds and
agrees to 0.29 points on average, so every figure in the Evaluation section
measures this policy reliably. That says nothing about whether a second 6,000,000
step run would also reach +16.97. Run 1 and Run 2 ran identical config at the same
seed and sat 11.32 points apart at step 2,000,000, so the spread between runs is
large and nothing here measures it at this run's length.

**The probe measures states a random policy visits.**
[src/probe.py](../../src/probe.py) builds its 500 states by stepping with
`env.action_space.sample()`, so they are early game positions with short rallies
and the ball rarely returned. That is deliberate, because a fixed yardstick has to
be independent of the policy it measures, but it means the Q figures describe
states the trained agent seldom reaches rather than the ones it plays from.

**Only 31 of the 121 checkpoints were scored.** The sweep ran every 200,000 steps,
so 90 checkpoints between the sampled ones were never evaluated. The best
checkpoint is therefore the best of those 31, and a better one could sit between
two of them.

---

## Reproducing the figures

Every number above that comes from Run 2's own files comes out of one command, run
from the repository root. That covers the config, training, evaluation,
instrumentation and hardware figures:

```powershell
python scripts/analyze_run.py docs/runs/run-02
```

Standard errors are `std_reward` over 10, the root of 100 episodes. The checkpoint
count and size print only on the machine that holds the checkpoints, since they are
read from `output_dir` in `run_config.json`.

Every comparison with Run 1 comes from the same command with `--compare`:

```powershell
python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-01
```

Re running the sweep itself is the one thing this repo cannot do. `sweep_eval.py`
needs a `checkpoints/` folder inside the run directory it is given, and those
3.03 GiB are not committed, so the only directory that works is the one on the
machine that produced the run. The two seeds took 189.65 and 190.52 minutes.

```powershell
python scripts/sweep_eval.py ~/pong-runs/run-02 --every 200000
python scripts/sweep_eval.py ~/pong-runs/run-02 --every 200000 --seed 1 --out ~/pong-runs/run-02/eval_seed1.csv
```

The 256.1 steps per second the projection was built on comes from the warmup pass,
and comparing against it gives the 6.47 hour projection and the 6.3 percent miss:

```powershell
python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-02-warmup
```

The 252.1 for the uninstrumented warmup comes from
`python scripts/analyze_run.py docs/runs/run-00-warmup`.

The probe is not committed either, and regenerates from `PROBE_SEED` to the sha256
recorded in `run_config.json`:

```powershell
python -c "from src.probe import make_probe, probe_hash; print(probe_hash(make_probe()))"
```

