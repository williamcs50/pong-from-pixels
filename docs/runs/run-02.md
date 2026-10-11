# Run 2

## Pre registration

Written 2026-10-10, before launch, and committed before any Run 2 data exists.

**Config:** identical to Run 1 except `total_steps` of 6,000,000 and the
instrumentation, meaning `metrics.csv` every 10,000 env steps and a probe of 500
states with its sha256 in `run_config.json`. The measured cost of the
instrumentation is around zero: 256.1 steps per second after warmup against Run 1's
252.1, which is inside run to run variation. 6,000,000 steps projects to 6.47 hours.

| # | Prediction | Why |
|---|---|---|
| 1 | The trailing 100 mean crosses 0 before step 2,300,000 | Run 1 gained 2.02, 2.27 and 2.56 points over its last three 200,000 step windows and ended at -0.11 at 2,000,000. Same config and seed, so Run 2 should track it to 2,000,000 and cross shortly after. |
| 2 | A plateau is the trailing mean moving less than 1.0 across 200,000 env steps | Restated from Run 1's "200 episodes", which was 182,000 steps early and 828,000 late, so the same words were a four times looser test at the end. Steps are comparable across the whole run. |
| 3 | At least one checkpoint clears a mean above 0 over 100 evaluation episodes | Run 1's best was -1.63 at 1,800,000 and no checkpoint cleared the bar. Three times the steps should be enough if the frame budget was the binding constraint. |

**What each outcome means:**

- **Crosses 0 and a checkpoint clears the bar.** The frame budget was the binding constraint and Run 1 ran out of steps. The remaining question is how far above the bar it goes, not whether it gets there.
- **Crosses 0 but no checkpoint clears the bar.** The training curve and the evaluation curve disagree the way they did in Run 1, where -0.11 training was -4.41 on evaluation. That would mean the gap is structural rather than a Run 1 accident, and the bar needs to be chased on evaluation rather than on training reward.
- **Never crosses 0.** Two failed predictions at three times the step count means the frame budget explanation is wrong. That sends me back to the learning rate, the replay buffer at 100,000 against Nature's 1,000,000, and the target sync interval rather than to a longer run.
- **Plateaus by the definition above.** Whatever value it plateaus at is the ceiling for this config, and Run 3 changes one hyperparameter rather than adding steps.

## Results

Empty until Run 2 finishes.
