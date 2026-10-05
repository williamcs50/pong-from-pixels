# Plans

**Buildable work done:** Saturday, October 17, 2026

**Hard stop:** Saturday, October 24, 2026. The [case study](https://williamopyrchal.com/pong) gets its final update with whatever I have.

---

## Working hours

Monday through Saturday, 8:30 AM to 6:00 PM, with shorter Saturdays. Sundays off.

Each working day gets an entry in [sessions.md](sessions.md), with a Floor and Aspiration set at the start and what landed, what's open, and anything surprising written at the end.

---

## Finish line

1. Train the agent until it reliably beats the built in opponent, on the 6 GB card.
2. Log reward per episode to disk, so the real learning curve replaces the representative one.
3. Checkpoint the model at milestones (random, first rally, breakthrough, trained) and record a short gameplay clip at each, so the demo's slider scrubs through real footage.
4. Instrument the internals: Q value drift, the epsilon schedule, replay dynamics, plus wall clock time, VRAM, and GPU temperature per run.
5. Write up what actually worked and what didn't, including the dead ends.
6. Update the [case study](https://williamopyrchal.com/pong) with the real data.

**The bar:** average reward above 0 over 100 evaluation episodes means the agent beats the built in opponent. +18 or better, close to the original DQN paper, is the aspiration.

---

## Task blocks

| Dates | Block |
|---|---|
| Oct 6 to Oct 8 | **Ready to launch.** Real run config script, reward logging to disk, checkpointing every N steps, the six actions vs three bars decision, the overfit break check, and a launch checklist. Launch the first real run on the PC. |
| Oct 9 to Oct 10 | **Instrumentation** while the first run trains: Q value drift, epsilon, replay dynamics, wall clock time, VRAM, GPU temperature. |
| Oct 12 to Oct 14 | **Diagnose and relaunch.** Check the first run's curve, fix what's wrong, and relaunch with instrumentation in. |
| Oct 15 to Oct 17 | **Clips.** Find the milestone moments in the reward log and record a clip at each checkpoint. Buildable work done. |
| Oct 19 to Oct 23 | **Case study.** Write up what worked and what didn't, and put the real curve and clips on the case study. |
| Oct 24 | **Final update to the case study.** |

Checkpointing and the actions decision go in before the first launch. Early milestones like "random" and "first rally" happen in the first hours of training and can't be saved after the fact, and training on 3 actions instead of 6 changes the network's output size, so either one would force a restart.

---

## If it hasn't converged by October 24

I update the case study with the real result: the actual reward log per episode, recorded gameplay at the training checkpoints, the training diagnostics, what worked and what did not, including the dead ends, and a clear conclusion about whether the agent converged within the available training time. The case study becomes an honest account of what the reproduction achieved, what it did not, and what the evidence suggests should happen next.
