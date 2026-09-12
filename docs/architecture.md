# Architecture

## System diagram

Collection: Preprocessor → Transform → Action Selector → Env → Preprocessor → Replay Buffer

Training: Replay Buffer → Transform → Q-Network + Target Network → Loss → Weight Update

## Components

### Environment

In: action (integer)

Out: raw frame (210x160x3 uint8 RGB), reward (-1, 0, or 1), done (bool)

Build or import: import

Why: Gymnasium/ALE is pure plumbing. It does not touch how the agent learns.

### Preprocessing

In: raw frame (210x160x3 uint8 RGB)

Out: stacked observation (84x84x4 uint8, raw pixel values)

Build or import: build

Why: The grayscale conversion, resize, and frame-stacking pipeline defines what the network sees, which directly shapes the representations it can learn.

Known choice: no crop before the resize. The full 210x160 frame is squashed to 84x84. This matches current standard practice rather than departing from it: Gymnasium's own `AtariPreprocessing` wrapper makes the identical call, `cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)` on the uncropped frame, as does OpenAI Baselines' `WarpFrame`. The crop to the playing area belongs to the 2015 Nature paper, which downsampled to 110x84 and then cropped 84x84, and later implementations dropped it.

It was not arrived at by deciding, though. There has been no crop since the first commit, because the spec said 210x160 in and 84x84 out and the code implemented that literally. Three measured consequences, which hold either way:

- The score digits survive into the observation at rows 0 to 8, columns 19 to 24 and 61 to 66. The network sees them. They change during play, they are irrelevant to control, and they correlate with reward.
- The scales differ per axis, 84/210 = 0.4 vertically and 84/160 = 0.525 horizontally, so the frame is anisotropically squashed. Cropping first would bring the two closer.
- Adopting the crop later moves every measured constant in `scripts/visual_check.py`. The vertical scale would become 0.525, shifting the field bounds, the paddle columns, and every expected paddle height.

Recorded here so the resize call is not the only place the choice lives. No reason to change it, but it should be a choice from now on rather than an inherited default.

### Replay buffer

In: transition tuples (state, action, reward, next_state, done)

Out: random mini-batch of transitions

Build or import: build

Why: The sampling strategy directly shapes what the agent learns and how stable that learning is.

### Transform

In: stacked observation (84x84x4 uint8, single frame or batch)

Out: normalized observation (Nx4x84x84 float32, [0,1])

Build or import: build

Why: The buffer stores raw uint8 for memory reasons and the network needs normalized float32 in channel-first order. One shared function does that conversion in exactly one place, used identically by the training path and the action selection path, instead of the conversion existing twice and disagreeing with itself.

### Q-Network

In: normalized observation (Nx4x84x84 float32)

Out: Q-value estimate for each action

Build or import: build

Why: The number of layers, kernel sizes, and strides in the conv stack determine what representations the agent learns, and those choices cannot transfer from a pretrained model trained on a different domain.

### Target network

In: normalized observation (Nx4x84x84 float32)

Out: Q-value estimate for each action (frozen weights)

Build or import: build

Why: Same architecture as the Q-network. The sync schedule is a learning stability decision, not plumbing.

### Action selector

In: Q-values, current epsilon

Out: action (integer)

Build or import: build

Why: The epsilon-greedy policy and decay schedule control the explore/exploit tradeoff, which directly shapes what experiences the agent collects and therefore what it learns.

### Training loop

In: everything above

Out: updated Q-network weights

Build or import: build

Why: The orchestration of when to sample, when to update, when to sync the target network, and the loss function are all learning decisions. The optimizer is imported from PyTorch. It is plumbing. It does not change what the agent learns, only how fast and stably the weights move once the loss is computed.

## Hardware constraints

RTX 2060, 6GB VRAM.

The replay buffer lives in CPU RAM. Even a few hundred thousand transitions at 84x84x4 would consume hundreds of megabytes to gigabytes on the GPU.

Run `scripts/check_environment.py` for the current numbers. It measures buffer cost per transition and at capacity, VRAM total and free, and both networks resident, reading them off the actual arrays rather than restating arithmetic here where it would go stale.

Two structural properties that are worth knowing without running it. The buffer is allocated eagerly by `np.zeros` at construction rather than growing with use, so a capacity that does not fit fails in the first second rather than at hour six.

Known choice: the buffer stores each frame twice. `current_state` and `next_state` are separate arrays, and consecutive states overlap in 3 of their 4 frames, so every frame is held once as part of a state and again as part of the previous transition's next_state. The standard alternative is a frame-indexed buffer that stores each frame once and reconstructs states from indices on sample, which halves the cost. Not done, for two reasons: it is a rewrite of a component that is already built and verified, and vanilla first means not rewriting a working component for a constraint that is not binding. Revisit only if host RAM forces it. `scripts/check_environment.py` reports whether the configured capacity fits.

Batch size starts at 32 or 64 and profiles upward. Both networks, activations, gradients, and optimizer state all have to fit in 6GB during a training step.

## Files

```
src/action_selector.py       Action selector
src/preprocess.py            Preprocessing
src/q_network.py             Q-Network and target-network logic
src/random_agent.py          Baseline random agent
src/replay_buffer.py         Replay buffer
src/transform.py             Shared transform (uint8 HWC to normalized float32 NCHW)
scripts/check_environment.py Environment verification
scripts/visual_check.py      Optional visual inspection helper
```