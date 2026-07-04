# Saturday: Kickoff

**Date:** 2026-06-06

**Floor:** Pong rendering on screen (random agent) + repo live with first commit

**Aspiration:** GPU verified and doing ML work

---

## What landed today

- Pong renders on the screen (random agent)
- Repo is live

## Anything surprising or worth flagging

- Coming into this project, I had no prior experience with the Gymnasium library. Getting familiar with its environment API was the first real hurdle of the day.

---

# Saturday: Architecture

**Date:** 2026-06-20

**Floor:** GPU verified and working. A tensor on CUDA, a small matmul, confirmation that the 2060 is live. Plus: a written architecture diagram in a markdown file, showing every component of the DQN with build/import labels and a sentence on why for each one.

**Aspiration:** All of that, plus the first built component has code. Maybe it's the replay buffer. Maybe it's the preprocessing pipeline. Something you can push to the repo that isn't just a plan. It's the plan starting to become real.

---

## What landed today

- The GPU is verified and working. `check_environment.py` runs a tensor on CUDA and a small matmul, confirming the 2060 is live and that PyTorch is using it.
- The architecture is documented in `docs/architecture.md`, with every component named, its interface defined, and a build or import call defended with a reason.

## Anything surprising or worth flagging

- I did not realize that pip install torch only installs a CPU-only build by default. There is no warning, and the GPU never gets used even if the hardware is ready. PyTorch has to be installed from its own index to get the CUDA build.

---

# Saturday: Core Build

**Date:** 2026-06-27

**Floor:** `src/preprocess.py` written, tested with a real Pong frame, output verified (shape, dtype, and value range), and committed. Visual inspection of a saved frame still pending.

**Aspiration:** All of that, plus the replay buffer also written, verified in isolation (store, sample, overflow behavior), and committed. Two components, both verified, both in the repo.

---

## What landed today

- `src/preprocess.py` written and committed. Grayscale conversion, resize to 84x84, and frame stacking across a sliding window of 4 frames.
- `tests/test_preprocess.py` passes all tests against a real Pong frame. Shape, dtype, and value range verified. Visual inspection of a saved frame still pending.

## What's open (carrying forward)

- Replay buffer will be deferred to next session.
- Before next session: save a preprocessed frame as a PNG with `cv2.imwrite` and confirm the ball is visible. This is the one verification the test suite cannot do.
- The preprocessor outputs HWC format (84, 84, 4). The NCHW transpose needed by the network will be handled in the agent, not here.

## Anything surprising or worth flagging

- Coming into this session, I had no prior experience with OpenCV. Getting familiar with the grayscale conversion and resize API was the first real step before writing `src/preprocess.py`.

---

# Saturday: Replay Buffer

**Date:** 2026-07-04

**Floor:** `src/replay_buffer.py` hand-built and verified in isolation. Push transitions, sample batches, assert shapes, and confirm circular overwrite when the buffer fills. Committed.

**Aspiration:** All of that, plus the Q-network and target network written and verified with a GPU forward pass. A dummy batch through CUDA, output shape confirmed at (B, n_actions), loss.backward() runs without error, and target weights decoupled and syncable. The HWC to NCHW transpose and /255 normalization also land here.

---

## What landed today

- `src/replay_buffer.py` was written and committed.
- `tests/test_replay_buffer.py` passes all tests: shapes and dtypes, circular overwrite, state and next_state pairing, and error handling when `batch_size` exceeds the number of stored transitions.

## What's open (carrying forward)

- Floor was met. Aspiration not reached. Q-network and target network deferred to next session.

## Anything surprising or worth flagging

- I assumed the replay buffer capacity would match the original DQN paper at 1 million transitions. Working through the actual memory budget brought it down. Each state is 84 x 84 x 4 = 28,224 bytes as uint8. Storing both state and next_state per transition doubles that to 56,448 bytes. With 16 GB RAM minus roughly 5 GB for OS, PyTorch, and the CUDA context, about 11 GB is available. That buys around 195,000 transitions. 100,000 was chosen as a conservative starting point.