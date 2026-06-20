# Architecture

## System diagram

Collection: Preprocessor → Action Selector → Env → Preprocessor → Replay Buffer

Training: Replay Buffer → Q-Network + Target Network → Loss → Weight Update

## Components

### Environment

In: action (integer)

Out: raw frame (210x160x3 uint8 RGB), reward (-1, 0, or 1), done (bool)

Build or import: import

Why: Gymnasium/ALE is pure plumbing. It does not touch how the agent learns.

### Preprocessing

In: raw frame (210x160x3 uint8 RGB)

Out: stacked observation (84x84x4 uint8)

Build or import: build

Why: The grayscale conversion, resize, and frame-stacking pipeline defines what the network sees, which directly shapes the representations it can learn.

### Replay buffer

In: transition tuples (state, action, reward, next_state, done)

Out: random mini-batch of transitions

Build or import: build

Why: The sampling strategy directly shapes what the agent learns and how stable that learning is.

### Q-Network

In: stacked observation (84x84x4)

Out: Q-value estimate for each action

Build or import: build

Why: The number of layers, kernel sizes, and strides in the conv stack determine what representations the agent learns, and those choices cannot transfer from a pretrained model trained on a different domain.

### Target network

In: stacked observation (84x84x4)

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

Batch size starts at 32 or 64 and profiles upward. Both networks, activations, gradients, and optimizer state all have to fit in 6GB during a training step.

## Files

```
src/preprocess.py            Preprocessing
src/replay_buffer.py         Replay buffer
src/model.py                 Q-Network and Target network
src/agent.py                 Action selector
src/train.py                 Training loop
src/random_agent.py          Baseline random agent
scripts/check_environment.py Environment verification
```