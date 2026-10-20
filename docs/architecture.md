# Architecture

## System diagram

Acting: Env → Preprocessor → Transform → Q network → Action selector → Env, with every step's transition pushed into the Replay buffer

Training: Replay buffer → Transform → Q network and Target network → Huber loss → Adam → Q network weights, with the Target network synced every 10,000 updates

Recording: Training loop → `episodes.csv`, `metrics.csv` (Q values on the Probe), and checkpoints

Evaluation: Checkpoint → `evaluate()` → the sweep, single checkpoint scores, the live game window, and clips

Analysis: committed log files → `analyze_run.py` → every figure in the run write ups

## Components

### Environment

In: action (integer, one of 6)

Out: raw frame (210x160x3 uint8 RGB), reward (-1, 0, or 1), terminated and truncated (bool)

Build or import: import

Why: Gymnasium and ALE are pure plumbing. They don't touch how the agent learns. `ALE/Pong-v5` repeats each action for 4 frames, and sticky actions are turned off (0.0, against the default 0.25) to match the DQN papers. The training loop reads that value back from the emulator instead of trusting the argument.

### Preprocessing

In: raw frame (210x160x3 uint8 RGB)

Out: stacked observation (84x84x4 uint8, raw pixel values)

Build or import: build

Why: The grayscale conversion, resize, and frame stacking define what the network sees, which directly shapes what it can learn.

Known choice: no crop before the resize. The full 210x160 frame is squashed to 84x84. That's standard practice now: Gymnasium's own `AtariPreprocessing` wrapper makes the identical call, `cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)` on the uncropped frame, as does OpenAI Baselines' `WarpFrame`. The crop to the playing area belongs to the 2015 Nature paper, which downsampled to 110x84 and then cropped 84x84, and later implementations dropped it.

I never actually decided it, though. There's been no crop since the first commit, because the spec said 210x160 in and 84x84 out and I implemented exactly that. Three things I measured, which are true either way:

- The score digits survive into the observation at rows 0 to 8, columns 19 to 24 and 61 to 66. The network sees them. They change during play, they're irrelevant to control, and they correlate with reward.
- The scales differ per axis, 84/210 = 0.4 vertically and 84/160 = 0.525 horizontally, so the frame is squashed more in one direction than the other. Cropping first would bring the two closer.
- Adopting the crop later moves every measured constant in `scripts/visual_check.py`. The vertical scale would become 0.525, shifting the field bounds, the paddle columns, and every expected paddle height.

I'm writing it down here so the resize call isn't the only place it shows up. There's no reason to change it, but from now on it's a choice, not something I inherited by accident.

### Replay buffer

In: transitions (state, action, reward, next_state, done)

Out: a random batch of transitions

Build or import: build

Why: The sampling strategy directly shapes what the agent learns and how stable that learning is.

### Transform

In: stacked observation (84x84x4 uint8, single frame or batch)

Out: normalized observation (Nx4x84x84 float32, [0,1])

Build or import: build

Why: The buffer stores raw uint8 for memory reasons, and the network needs normalized float32 with the channels first. One shared function does that conversion in exactly one place, used the same way by the training path and the acting path, instead of two copies that could drift apart.

### Q network

In: normalized observation (Nx4x84x84 float32)

Out: one Q value estimate per action

Build or import: build

Why: The number of layers, kernel sizes, and strides in the conv stack determine what the agent can learn, and a pretrained model built for different images wouldn't fit Pong. The layers match the 2015 Nature paper: three conv layers, then 512 fully connected units, then one output per action.

### Target network

In: normalized observation (Nx4x84x84 float32)

Out: one Q value estimate per action (frozen weights)

Build or import: build

Why: Same architecture as the Q network. It's a second `QNetwork`, and syncing is a hard copy with `load_state_dict` every 10,000 gradient updates. The sync schedule is a learning stability decision, not plumbing.

### Action selector

In: Q values, current epsilon, number of actions

Out: action (integer)

Build or import: build

Why: Epsilon greedy choice controls the trade off between exploring and exploiting, which directly shapes what experiences the agent collects and therefore what it learns. The selector only makes the choice. The epsilon schedule lives in the training loop, and whatever calls the network owns the `no_grad` context.

### Training loop

In: everything above, plus the run's config

Out: updated Q network weights, `episodes.csv`, `metrics.csv`, `run_config.json`, and checkpoints

Build or import: build

Why: When to sample, when to update, when to sync the target network, how epsilon decays, and which loss to use are all learning decisions. The loss is Huber (`smooth_l1_loss`), and only `terminated` stops bootstrapping, since a truncated game could have kept going. The optimizer is imported from PyTorch, since it doesn't change what the agent learns, only how fast and how stably the weights move once the loss is computed.

`scripts/tiny_train.py` and `scripts/real_train.py` both call this one `train()` with different settings, so the tiny run tests the same code the real runs use.

### Checkpoints

In: both networks, the optimizer, the step, episode, and update counters, and the numpy, torch, and CUDA random number generator states

Out: one `.pt` file per save, and a load that puts all of it back

Build or import: build, on top of `torch.save` and `torch.load`

Why: A resume needs more than weights. Without the counters, it would restart the epsilon schedule and the target sync. Saves go to a `.tmp` file and then get renamed, so a crash in the middle of a save can't leave a half written newest checkpoint. The replay buffer is left out, since it's 5.26 GiB per save.

### Probe and instrumentation

In: the Q network, and 500 fixed game states from `src/probe.py`

Out: one row of `metrics.csv` every 10,000 steps: the mean highest Q value, the mean gap between the highest and lowest, the mean Q value per action, how many states pick each action, epsilon, and how often training batches include a scored point or the end of a game

Build or import: build

Why: Reward only moves once the agent starts winning points. The probe shows what the network is learning before that. The states come from a random policy at a fixed seed, `PROBE_SEED`, separate from the run seed, so every row and every run measures the same states. Their sha256 goes in `run_config.json`, so the file doesn't need committing.

### Evaluation

In: a checkpoint, the number of games, epsilon, and a seed

Out: mean reward and its spread, game lengths, the actions taken, and, when asked, every frame and every set of Q values from the first game

Build or import: build

Why: The bar is defined over 100 evaluation games at epsilon 0.05, not over training reward, which averages a policy that keeps changing. `src/evaluate.py` is the one path every evaluation goes through: `sweep_eval.py`, `play_checkpoint.py`, `watch_checkpoint.py`, and `record_clip.py`. So the sweep's scores, the live window, and the clips all show the same games. A fixed seed makes it deterministic, and it checks that loading actually changed the network's weights, since an unloaded network plays games without crashing too.

### Run records

In: the repository and the installed packages

Out: the commit, whether the working tree was clean, the software versions, and the folder each run writes to

Build or import: build, on top of `git` and `importlib.metadata`

Why: A number can only be checked if you know what code produced it. `src/git_info.py` adds this to every run, sweep, and clip, and `train()` refuses to start with uncommitted changes. `src/run_paths.py` puts every run under `~/pong-runs`, outside the repository and outside OneDrive. Every script expands `~` in the paths it's given, since PowerShell passes it through as is, so the same command works in any shell.

## Hardware constraints

RTX 2060, 6 GB VRAM. About 1.2 GB of it is already in use at idle.

The replay buffer lives in CPU RAM. At 100,000 transitions it takes 5.26 GiB, which wouldn't fit on the card alongside everything else.

Run `scripts/check_environment.py` for the current numbers. It measures the buffer's cost per transition and at capacity, total and free VRAM, both networks resident, the Adam state, and a training step's memory. Those come from real arrays, so they can't go stale the way numbers written here would.

One structural property is worth knowing without running it. The buffer is allocated all at once by `np.zeros` when it's created, rather than growing with use, so a capacity that doesn't fit fails in the first second rather than at hour six.

Known choice: the buffer stores each frame twice. `current_state` and `next_state` are separate arrays, and consecutive states overlap in 3 of their 4 frames, so every frame is held once as part of a state and again as part of the previous transition's next_state. The standard alternative is a buffer that stores each frame once and rebuilds states from indices when sampling, which halves the cost. I haven't done it, for two reasons. It's a rewrite of a component that's already built and verified, and RAM has never been the limit, so there's nothing to gain yet. I'll only revisit it if host RAM forces it. `scripts/check_environment.py` reports whether the configured capacity fits.

The batch size is 32, matching the Nature paper. The GPU isn't the limit: a training step measures 40.3 MiB resident with an 18.1 MiB peak on top, and Adam's state is 12.87 MiB. Run 1 and Run 2 never used more than 2,287 MiB in total, and most of that was the CUDA context and other programs. The bottleneck is the CPU side, one Pong game and the Python loop, which is why Run 2 averaged 242.2 steps per second while the GPU never went above 42 percent use.

## Tests

Run them all with `python -m pytest -q`, or one file on its own, like `python tests/test_train.py`. The whole suite takes about 50 seconds on the PC. None of it needs a trained checkpoint, and it uses the GPU if there is one and the CPU if there isn't.

How I write them:

- Where I can, a test checks exact values I worked out by hand first, not just shapes. A shape check still passes when the numbers are wrong.
- For the important checks, I broke the code on purpose and watched the test fail before trusting it. A test that can't fail isn't testing anything.
- Tests that need game frames use real Pong frames from the emulator. The few that need every frame to be different use random noise instead.

What each file covers:

- **`smoke_test.py`** runs both paths once, end to end. The acting path plays 10 steps of a real game through preprocessing, the transform, the network, and the action selector. The learning path samples a batch of 8 from the buffer and runs it through the transform and the network. Both check shape, dtype, device, and that nothing is NaN. This is the test that catches pieces that work alone but don't fit together.
- **`test_action_selector.py`** checks that at epsilon 0, 100 picks are all the highest Q value, and at epsilon 1, 6,000 picks land within 25 percent of an even 1,000 per action. It also checks the result is a plain Python int, which caught a real bug: the random branch used to return a numpy int.
- **`test_q_network.py`** checks the output is one Q value per action on the right device, and that gradients reach every layer. Then the target network: the optimizer doesn't hold it, it stays frozen through a step on the Q network, and it matches the Q network after a sync. Last, it checks the device order: an NVIDIA GPU first, then an Apple GPU, then the CPU.
- **`test_preprocess.py`** checks shape, dtype, and value range on a real Pong frame, then the frame stack. The newest frame has to land at the last index, and a reset has to clear the last game's frames, so a stack never mixes two games.
- **`test_replay_buffer.py`** checks shapes and dtypes after push and sample, the oldest move getting overwritten once it's full, each state staying paired with its own next state, and an error when asked for a bigger batch than it holds.
- **`test_transform.py`** sends a known frame through the transform and checks every value lands in the right place, not just the shape. Then the round trip: a real frame through preprocessing, the buffer, and the transform has to match transforming it directly.
- **`test_train.py`** checks the training step against losses I worked out by hand, 1.25 for a normal batch and 4.0 for a batch where every move ends a game, since the target is then the reward alone. It also checks that the Q network moves and the target doesn't, that a game ending ignores the next state, and that 200 steps on one fixed batch drive the loss below 1 percent of where it started. I broke the training step on purpose to make sure the overfit test and the 4.0 test can fail. The next state test alone missed a broken target once, which is why the 4.0 test exists.
- **`test_checkpoint.py`** checks that a save then a load puts back both networks, the Adam state, and the counters. With the random number generators restored, numpy and torch draw the same values they would have after the save. Saving twice to one path leaves one file and no `.tmp`.
- **`test_logging.py`** runs `train()` once at a tiny size, 2,400 steps, and checks everything it wrote: metrics rows on schedule, Q values identical before the first update and moving after, epsilon matching the schedule on every row, episode rows that cover every step with no gaps, and checkpoints holding the counters their names say. It also checks that `train()` refuses a folder that already holds a run, and refuses uncommitted changes before writing anything. One last test checks that `real_train.py` expands `~` in `--output-dir`, so a run started from PowerShell doesn't land in a folder named `~`.
- **`test_instrumentation.py`** checks the probe metrics against hand computed values, on a network that always returns the same Q values. The probe has to come out identical at one seed and different at another, and its hash has to change when a single pixel does.
- **`test_git_info.py`** checks the commit and the dirty flag match what git itself says. An untracked file counts as dirty, and a missing or broken git reads as unknown instead of clean. The software versions still get recorded when cuDNN throws an error.
- **`test_evaluate.py`** checks that step 0 can be scored twice in one process and plays the same game both times, which used to fail. Recording keeps one frame per step plus the first, and doesn't change the game.
- **`test_sweep_eval.py`** runs the real sweep script on small untrained checkpoints. It has to pick only the right steps and resume without redoing rows. It has to stop on a checkpoint whose saved step doesn't match its name, a results file with other columns or settings, no matching checkpoint, or a bad epsilon.
- **`test_record_clip.py`** checks return counting on made up frames with the ball placed by hand. A ball turning back at the agent's paddle counts, and a missed ball or the opponent's hit doesn't. A rally needs 3 returns within 30 seconds, and the untrained network never returns the ball. A bad flag or an existing clip stops the script before it writes anything.
- **`test_analyze_run.py`** checks that every number in run-01.md and run-02.md shows up in `analyze_run.py`'s output, apart from a short list of outside facts like the GPU's rated power. One test proves that check can fail. It also checks the analyzer's math on short series worked out by hand.

## Files

```
src/action_selector.py         Picks the action: random with probability epsilon, otherwise the highest Q value
src/checkpoint.py              Checkpoints: save and load everything a resume needs
src/device.py                  Picks where the network runs: an NVIDIA GPU, then an Apple GPU, then the CPU
src/evaluate.py                Evaluation: the one path every checkpoint is scored through
src/git_info.py                Run records: commit, clean or not, and software versions
src/preprocess.py              Turns game screens into 84x84 grayscale, stacked four frames deep
src/probe.py                   The 500 fixed probe states
src/q_network.py               The network that estimates a Q value per action, used for the target network too
src/replay_buffer.py           Stores the most recent moves, 100,000 in the real runs, and samples training batches
src/run_paths.py               Run records: where runs are written
src/train.py                   Training loop, the training step, and the probe metrics
src/transform.py               Shared transform (uint8 HWC to normalized float32 NCHW)

scripts/analyze_run.py         Prints every figure in a run write up from its committed logs
scripts/check_environment.py   Environment check and memory budget
scripts/play_checkpoint.py     Scores one checkpoint
scripts/random_agent.py        Random actions playing in a game window, one game by default, with the score
scripts/real_train.py          The real runs' config
scripts/record_clip.py         Records a milestone clip and its Q values
scripts/sweep_eval.py          Scores every Nth checkpoint of a run
scripts/tiny_train.py          A tiny run through the whole pipeline
scripts/visual_check.py        Checks preprocessing against real frames, and what each action does
scripts/watch_checkpoint.py    A trained checkpoint playing in a game window

tests/                         15 test files, each described under Tests above
```
