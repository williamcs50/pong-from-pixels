# Run 2

Every number here comes from the run's own files, which are committed in
[run-02/](run-02/): `run_config.json`, `episodes.csv`, `metrics.csv`, `gpu_log.csv`,
the two evaluation files, and the length rescore. The warmup pass's three files are in
[run-02-warmup/](run-02-warmup/). The commands that produce every number are at the
bottom, so you can check any of them.

Two things aren't committed. The checkpoints are 3.03 GiB (3.26 GB), so I measured
their count and size on disk. The 500 state probe is 13.5 MB, and it doesn't need to
be committed: `run_config.json` records its sha256, and it regenerates from
`PROBE_SEED`.

**Output:** `run-02` under `RUNS_ROOT`, which [src/run_paths.py](../../src/run_paths.py)
resolves to `~/pong-runs`, outside the OneDrive sync root.

**Commit:** `1e407be98054234feefd5dcbdfd87291515d2a94`, with `git_dirty` false, so the
run launched from committed code. It's the same commit that recorded the pre
registration below. I made it on Oct 10 at 21:31:53, 14 hours and 17 minutes before
the run started.

The warmup pass ran from `f03f1410977e553ea26d0d0841f750160eb60aa8`. Unlike Run 1,
the changes between the warmup and the run aren't just documentation. `real_train.py`
gained a `--probe` flag the run never used, `probe.py` lost an unused `import os`, and
`analyze_run.py` gained a guard for a missing GPU log. None of that touches training.
Both runs also recorded the same probe sha256, which is actual evidence, not just me
reading the diff. You can check with `git diff --stat f03f141 1e407be`.

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
| 2 | A plateau is the trailing mean moving less than 1.0 across 200,000 env steps | Restated from Run 1's "200 episodes", which spanned roughly four times as many steps late in the run as early, so the same words were a looser test at the end. Steps are comparable across the whole run. | It fired, but what I said it would mean turned out wrong. From 3,600,000 to 4,000,000 the trailing mean stayed within 1.0 in each 200,000 step window, exactly as defined. Then the run kept going to +19.02, so that flat stretch wasn't the ceiling I said it would be. In the same stretch, evaluation moved 8.80 at seed 0 and 9.51 at seed 1. A trailing mean averages about 250,000 steps of play, so it can't show swings that short. The definition fires in 12 of the 30 windows between evaluated checkpoints, counting either the change between endpoints or the range inside. |
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

Everything matches Run 1 except `total_steps`, which Run 1's result decided, and the
two instrumentation rows. My reasoning for the shared values is in the 2026-10-06
entry of [sessions.md](../sessions.md), and for the instrumentation in the 2026-10-09
entry.

---

## Results

**The agent reached the bar.** The trailing mean over the last 100 episodes crossed
0 at step 2,715,875. It peaked at +19.02 and finished at +16.80, at step 5,998,105.
All four of those come from `run-02/episodes.csv`.

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

The trailing mean only exists where an episode ended, so each row is the nearest
episode boundary to that step. The last row is the run's final boundary, 1,895 steps
short of 6,000,000.

The run had 2,495 episodes and 1,487,027 gradient updates by the last logged episode.
2,212 of those episodes ran at epsilon 0.01, the floor the schedule reaches at step
250,000. These are training episodes, though, and the bar is defined over 100
evaluation episodes at epsilon 0.05. The two aren't interchangeable: training finished
at +16.80, but the final checkpoint scores +12.60.

**My pre registered prediction failed.** On 2026-10-10 I predicted the trailing mean
would cross 0 before step 2,300,000. It crossed at 2,715,875, 415,875 steps late.

**The reason it failed matters more than the miss.** I assumed Run 2 would track Run 1
up to 2,000,000, since the config and the seed are identical. It didn't. At 2,000,000
Run 1 was at -0.11 and Run 2 was at -11.43, 11.32 points apart. The two runs match in
every column except wall clock time for the first 54 episodes. They split at episode
55, the first one that crosses step 50,000, which is where gradient updates start. So
on this hardware, a fixed seed doesn't reproduce a run.

**The curve didn't end at its best.** It peaked at +19.02 and finished 2.22 points
lower, at +16.80. Run 1, by contrast, ended at its own maximum.

**Supporting evidence that the learning is real, not noise:**

- Mean loss per episode fell from 0.012040 to 0.000986. It stayed finite the whole
  run, ranging 0.000194 to 0.013529.
- There were zero shutouts in the last 100 episodes. Run 1 had zero too, though, so
  this doesn't separate a losing agent from a winning one anymore.

---

## Evaluation

Everything above is training episodes at epsilon 0.01. The bar is defined over 100
evaluation episodes, so I scored every 200,000th checkpoint separately at epsilon 0.05.
I scored each one twice, at two independent evaluation seeds, in
[run-02/eval.csv](run-02/eval.csv) and [run-02/eval_seed1.csv](run-02/eval_seed1.csv).

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
step 5,400,000, and it's the best at seed 1 too, at +16.67.

**The best checkpoint isn't the last one.** 5,400,000 beats 6,000,000 by 4.37 points.
With standard errors of 0.21 and 0.25, that's 13.3 combined standard errors, so it's
nowhere near noise. Seed 1 gives 4.49 points and 12.6. The standard error is
`std_reward / 10`, since the square root of 100 episodes is 10.

**The two seeds agree everywhere.** Across all 31 checkpoints they differ by 0.29
points on average. The biggest single difference is 1.10, at 2,600,000, and **none of
the 31 differ by more than two standard errors**. That tells me 100 episodes at epsilon
0.05 is enough to rank checkpoints reliably. Every bump in this curve belongs to the
policy, not to which episodes it happened to get scored on.

**Two checkpoints collapse and then recover, and both seeds show it.** 4,600,000
scores +7.23 and +7.13, between neighbours at +14.60 and +12.14. 3,600,000 scores +6.19
and +5.88, between +7.88 and +14.99. Both dips are 7 to 9 points deep. The 3,600,000
one lands inside the stretch where the training plateau definition fired, and a
trailing mean covering about 250,000 steps can't see a dip that short.

**The gap between training and evaluation looks structural.** At the final step,
training read +16.80, while the checkpoint scores +12.60 at seed 0 and +12.18 at seed
1. That's a gap of -4.20 and -4.62. Run 1's gap was -4.30, even though its reward was
17 points lower. Two runs that ended that far apart still landed on almost the same
gap. So from here on, a run's headline number has to come from a sweep, not the
training log.

**Episode length tells you how even the game is, not how good the agent is.** The
longest evaluation episodes are at 2,600,000, averaging 5,299 steps, where the agent
scores -0.81. That's the closest sampled point to the crossing. The best checkpoint
plays 2,444 step episodes, less than half as long. In run-01.md I treated the rise from
909 to 4,139 steps as evidence of learning. But Run 1 finished close to even, which is
exactly where this number peaks. A better agent actually gets a lower number, because
a 21 to 3 win is over faster than a 21 to 19 grind.

**I didn't reach the +18 aspiration.** [PLANS.md](../PLANS.md) sets the bar at a mean
above 0 and the aspiration at +18 or better. The best checkpoint is +16.97 plus or minus
0.21, 1.03 points short, which is about five standard errors.

---

## Instrumentation

`metrics.csv` gets a row every 10,000 env steps. Each row runs the network over the
same 500 probe states and records the Q values, which action it would pick in each
state, epsilon, and what the gradient updates in that window sampled from the replay
buffer.

**I checked the logger before trusting anything in it.** `metrics.csv` has all 601
rows, one every 10,000 steps with none missing. On every row, the six argmax counts add
up to 500, one per probe state. The probe gets built once before training starts
([src/train.py](../../src/train.py)) and reused for every row. Regenerating it from
`PROBE_SEED` gives the same sha256 that `run_config.json` recorded, so every row
measured the same states. That means any change in the Q columns has to come from the
network. The first five rows back that up. Steps 0 to 40,000 come before any gradient
update, and every logged digit is identical: `q_max_mean` 0.034242 and
`q_spread_mean` 0.056905.

**`q_max_mean` dropped before it climbed.** It went below 0 at 140,000 and bottomed
out at -0.2180 at 760,000. Rewards are +1 and -1 per point, so a negative value means
the network expected to lose points from these states. It went above 0 for good at
1,570,000, peaked at 0.6686 at 5,980,000, and ended at 0.6521.

| Steps | Change in `q_max_mean` |
|---|---|
| 0 to 760,000 | -0.2522 |
| 760,000 to 3,000,000 | +0.5290 |
| 3,000,000 to 5,000,000 | +0.3346 |
| 5,000,000 to 6,000,000 | +0.0066 |

The last 1,000,000 steps barely moved it. Every row in that stretch is between 0.6118
and 0.6686.

**`q_spread_mean` collapsed with the first updates, then grew.** The single update
logged at step 50,000 dropped it from 0.0569 to 0.0413. By 60,000, 2,501 updates in, it
hit its minimum of 0.0050, meaning the network rated all six actions almost the same.
It peaked at 0.2067 at 2,230,000. From 3,000,000 on it stayed between 0.1401 and
0.2065, averaging 0.1721, and it ended at 0.1538. One caveat, which Limitations covers:
these are states a random policy visits, so both columns describe positions the
trained agent rarely gets into.

**Q spread started rising before reward did, by about 520,000 steps.** The analyzer
defines onset as the first step after a series' minimum where it stays above that
minimum plus a fraction of its range for the rest of the run. At 5 percent, Q spread
crosses at 340,000 and the trailing 100 mean at 860,000. The rule was chosen after the
data had already been looked at, so this is an observation and not a real test. To
avoid leaning on one threshold, the table shows four:

| Fraction | Q spread onset | Reward onset | Lead |
|---|---|---|---|
| 0.05 | 340,000 | 860,000 | 520,000 |
| 0.10 | 430,000 | 1,090,000 | 660,000 |
| 0.20 | 570,000 | 1,650,000 | 1,080,000 |
| 0.30 | 710,000 | 2,310,000 | 1,600,000 |

Q spread leads at every fraction, but how far it leads depends on the fraction. The
lead grows as the fraction goes up, because Q spread finishes most of its rise by its
peak at 2,230,000, while reward keeps climbing to the end. So 520,000, at the lowest
fraction, is the most honest answer to which one started rising first.

**The trained network leans toward moving up, and picks still less than an even
share.** The argmax counts are grouped by what each action does to the agent's
paddle. That grouping comes from Check 6 in `scripts/visual_check.py`, which drove each
action and measured the paddle, since the action names alone don't prove anything: 2
and 4 move it up, 3 and 5 move it down, 0 and 1 leave it still. With no preference,
the network would split the 500 states evenly, about 167 per group.

At step 0, all 500 states pick still. The untrained network just happens to rate
action 0 highest everywhere, so that isn't anything learned. Still dropped to 316
states by 200,000 and to 20 by 800,000, but it didn't stay down. From 1,000,000 on, the
200,000 step rows put it between 57 and 176. From 3,000,000 on, the averages were
213.9 up, 170.0 down and 116.1 still. After 510,000, still reached an even share on
only seven rows: six between 2,290,000 and 2,650,000, and one at 5,570,000.

These summaries were picked after the 200,000 step table had been seen. As with Q,
the probe states come from a random policy, so the lean toward up could just reflect
where the probe happens to put the ball, not a bias in the policy. Grouping RIGHT with
RIGHTFIRE also means this can't say anything about FIRE.

**Epsilon followed the schedule exactly.** `run_config.json` sets a linear decay from
1.0 to 0.01 over 250,000 steps. Every metrics row after step 0 matches that line to
within 0.00000004, which is just rounding at six decimals, and all 575 rows after
250,000 read exactly 0.01. The one odd value is the row at 250,000, which reads
0.010004. Each row is written at `env_step`, one past the loop index the epsilon was
computed from, so the row shows the value from the step before. The floor really took
effect at step 250,001, and the first row that shows it is 260,000.

**The reward signal per update fell by about four fifths while the agent learned to
rally, then came partway back once it started winning.** `batch_reward_fraction` is
the share of transitions sampled in each 10,000 step window where a point was scored.
`batch_done_fraction` is the share that end an episode. Both count the batches the
updates actually trained on. Over the 595 windows with a full 2,500 batches, the
reward fraction peaked at 0.025525 at 540,000, about one point every 40 steps. It fell
to 0.005387 at 3,160,000 as rallies got longer. From 3,400,000 on, every 200,000 step
row sits between 0.00914 and 0.01181, and it ended at 0.009787. That late rise lines up
with the episode length finding in Evaluation: an agent that's winning plays shorter
games, so points come more often. One thing to keep in mind is that the buffer holds
the last 100,000 transitions, so these fractions lag the agent's behaviour by up to
that many steps.

**The batch counters agree with the episode log.** One side always reaches 21, so
every episode has 42 minus the absolute reward in points. That means the episodes
inside the buffer's 100,000 step span predict what a uniform sample from it should
contain. Across the 595 full windows, the logged reward fraction is off from that
prediction by a mean of +0.06 sampling standard errors, with a spread of 0.94, and 577
rows land within 2. The done fraction gives +0.02, 1.01 and 572. Pure sampling noise
would give a mean of 0, a spread of 1 and about 95 percent within 2, which is what
these look like. So the counters measure what they're supposed to.

| Step | `q_max_mean` | `q_spread_mean` | Up | Down | Still | Reward fraction |
|---|---|---|---|---|---|---|
| 0 | 0.0342 | 0.0569 | 0 | 0 | 500 | |
| 1,000,000 | -0.1384 | 0.1303 | 146 | 263 | 91 | 0.00971 |
| 2,000,000 | 0.0553 | 0.1879 | 193 | 202 | 105 | 0.00684 |
| 3,000,000 | 0.3109 | 0.1715 | 198 | 168 | 134 | 0.00592 |
| 4,000,000 | 0.4993 | 0.1825 | 217 | 163 | 120 | 0.00989 |
| 5,000,000 | 0.6455 | 0.1697 | 214 | 182 | 104 | 0.00918 |
| 6,000,000 | 0.6521 | 0.1538 | 227 | 164 | 109 | 0.00979 |

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

**The GPU wasn't the bottleneck.** It never went above 42 percent utilization or 41.3
W, on a card rated around 160. So the limit is on the CPU side: one Pong instance,
frame preprocessing, and the Python loop running every step. The fill phase runs the
same emulator and preprocessing without any gradient updates, and it hit 844.4 steps
per second, against 240.7 once updates started.

**The GPU logger covered the whole run.** Its 827 timestamps go from 11:48:51 to
18:42:33. It started 31 seconds before the run and kept going after it ended. The gaps
between samples only range from 30.031 to 30.123 seconds, so nothing got dropped. Run
1's first attempt gave me a zero byte file, because `nvidia-smi -l 30 -f` buffered its
output. This is the proof that the `Add-Content` replacement writes as it goes.

**The instrumentation costs less than I can measure.** The instrumented warmup in
[run-02-warmup/](run-02-warmup/) ran at 256.1 steps per second, and the uninstrumented
one in [run-00-warmup/](run-00-warmup/) ran at 252.1. So the instrumented one was
actually faster, which is just run to run variation. My projection from 256.1 was 6.47
hours for 6,000,000 steps, and the run took 6.88. That 6.3 percent miss comes from the
rate not holding up, not from bad arithmetic: 256.1 against the actual 240.7 is 6.4
percent, the same gap.

**Run 2 ran 3.3 percent slower than Run 1, and I can't tell why from the logs.** It
averaged 240.7 against Run 1's 248.9 after warmup, while its fill phase was slightly
faster, at 844.4 against 837.3. The instrumentation is way too small to explain it.
601 probe rows of 500 states is 300,500 forward passes, against roughly 95 million in
the training batches. Run 2 also ran cooler, with a lower peak clock of 1,545 MHz
against 1,770 and 41.3 W against 51.27. So whatever was different was on the CPU side.

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

**Outcome.** I applied the rules after committing them, and `analyze_run.py` applies
them again every time it runs:

| Milestone | Checkpoint | Evidence |
|---|---|---|
| Random | 0 | The untrained network |
| First rally | 600,000 | Mean episode length 1,324 steps against step 0's 837, beating it by 487 against a bar of 95.4 |
| First over the bar | 2,800,000 | Seed 0 mean +5.45 |
| Trained | 5,400,000 | Seed 0 mean +16.97, seed 1 +16.67 |

All four are different checkpoints, so the slider keeps four positions. 200,000 and
400,000 didn't count as a first rally, because their episodes were actually shorter
than step 0's, at 764 and 769 steps. The episode check, playing each clip twice, comes
when I record the clips.

**Clip window.** Written after a first recording of all four episodes, and committed
before any clip is cut. Each clip on the page is the first 30 seconds of its
episode, 450 frames at 15 fps, starting from the reset frame. The window is the same
for all four, and every episode is longer than it, so no clip is chosen by where it
ends. The page loops each clip, muted, so it ends wherever frame 450 falls. Each clip
also saves the six Q values the network computed for every frame it shows, so the
page's Q value bars and win estimate come from the same episode as the footage.

**Clip window, revised.** Changed after cutting the first 30 seconds, and committed
before cutting again. Each clip is now the last 30 seconds of its episode, the last 450
frames, ending on the frame where the final point is scored. The first 30 seconds only
showed how each game opens, while the last 30 end on its result, with the final score on
screen. The window is still the same for all four, so the change applies to every clip
and not just the one that ends in a win. No action follows the final frame, so it has
no Q values of its own, and it repeats the frame before it.

**First rally clip, revised.** The seed 0 episode at 600,000 never returns the ball, so
its last 30 seconds didn't show the milestone the clip is named for. Counting returns
across seeds 0 to 8 showed they're real at 600,000 but patchy: six of the nine episodes
return the ball at least once, and only a few return it several times close together.
This rule was written after seeing those counts, and committed before recording. The
first rally clip comes from the first episode, trying evaluation seeds 0, 1, 2 and so
on in order, with at least 3 returns within 30 seconds, and starts 2 seconds before the
first of those returns. A return is the ball moving toward the agent's paddle and
turning back beside it. The other three clips keep the seed 0 episode and its last 30
seconds.

---

## Limitations

**The training run is a single seed.** The evaluation repeats at two seeds and agrees
to 0.29 points on average, so everything in the Evaluation section measures this
policy reliably. But it says nothing about whether a second 6,000,000 step run would
also get to +16.97. Run 1 and Run 2 used identical config and the same seed, and they
were still 11.32 points apart at step 2,000,000. So runs can vary a lot, and nothing
here measures how much at this length.

**The probe only covers states a random policy visits.**
[src/probe.py](../../src/probe.py) builds its 500 states by stepping with
`env.action_space.sample()`, so they're early game positions with short rallies where
the ball rarely comes back. That's on purpose, because a fixed yardstick has to be
independent of the policy it's measuring. The downside is that the Q figures describe
states the trained agent rarely sees, not the ones it actually plays from.

**The run didn't record its software versions.** The venv I launched it from has Python
3.11.0, torch 2.12.1+cu126 with CUDA 12.6, gymnasium 1.3.0, ale-py 0.12.0 and numpy
2.4.6, all installed on 2026-06-20 and unchanged since. So these are very likely what
it ran on. Runs from now on record them in `run_config.json`.

**Only 31 of the 121 checkpoints were scored.** The sweep ran every 200,000 steps, so
the 90 checkpoints in between were never evaluated. The best checkpoint is the best of
those 31, and a better one could be sitting between two of them.

---

## Reproducing the figures

Every number above that comes from Run 2's own files comes out of one command, run
from the repository root. That includes the config, training, evaluation,
instrumentation, hardware and clip selection figures, along with the rescore in
`eval_lengths.csv` and each clip rule's pick:

```powershell
python scripts/analyze_run.py docs/runs/run-02
```

Standard errors are `std_reward` over 10, the square root of 100 episodes. The
checkpoint count and size only print on the machine that has the checkpoints, since
they're read from `output_dir` in `run_config.json`. Two figures need other commands.
The software versions in Limitations come from the venv, since the run didn't record
them, and the probe hash match comes from the command at the end of this section.

Every comparison with Run 1 comes from the same command with `--compare`:

```powershell
python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-01
```

The one thing this repo can't do is rerun the sweep itself. `sweep_eval.py` needs a
`checkpoints/` folder inside the run directory you give it, and those 3.03 GiB aren't
committed, so it only works on the machine that produced the run. The two seeds took
189.65 and 190.52 minutes.

Both sweeps were written before the sweep added its `std_episode_steps` column. It
refuses to append to a file with different columns, so a rerun needs a new `--out`:

```powershell
python scripts/sweep_eval.py $HOME\pong-runs\run-02 --every 200000 --out $HOME\pong-runs\run-02\eval_rerun.csv
python scripts/sweep_eval.py $HOME\pong-runs\run-02 --every 200000 --seed 1 --out $HOME\pong-runs\run-02\eval_seed1_rerun.csv
```

The first rally rule in Clip selection uses
[run-02/eval_lengths.csv](run-02/eval_lengths.csv), a rescore of checkpoints 0 to
1,000,000 that adds the spread of episode length. It ran at commit `a6ce59e` plus the
`std_episode_steps` column and `--through` option, which weren't committed yet.
[run-02/eval_lengths_config.json](run-02/eval_lengths_config.json) records that as
`git_dirty`. They were committed afterward in `63ea0da`, along with one later edit that
doesn't affect scoring: recording the software versions in the sweep's config. Its mean reward, reward spread and mean episode length match `eval.csv`
exactly on all six checkpoints. It took 17.9 minutes:

```powershell
python scripts/sweep_eval.py $HOME\pong-runs\run-02 --every 200000 --through 1000000 --out $HOME\pong-runs\run-02\eval_lengths.csv
```

In PowerShell, use `$HOME` instead of `~`. PowerShell passes `~` to the script as is,
and the script doesn't expand it.

The 256.1 steps per second my projection used comes from the warmup pass. Comparing
against it gives the 6.47 hour projection and the 6.3 percent miss:

```powershell
python scripts/analyze_run.py docs/runs/run-02 --compare docs/runs/run-02-warmup
```

The 252.1 for the uninstrumented warmup comes from
`python scripts/analyze_run.py docs/runs/run-00-warmup`.

The probe isn't committed either. It regenerates from `PROBE_SEED` to the sha256
recorded in `run_config.json`:

```powershell
python -c "from src.probe import make_probe, probe_hash; print(probe_hash(make_probe()))"
```
