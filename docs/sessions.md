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

**Floor:** `src/preprocess.py` written, tested with a real Pong frame, output verified (shape, dtype, value range), and committed.

**Aspiration:** All of that, plus the replay buffer also written, verified in isolation (store, sample, overflow behavior), and committed. Two components, both verified, both in the repo.

---

## What landed today

- `src/preprocess.py` written and committed. Grayscale conversion, resize to 84x84, and frame stacking across a sliding window of 4 frames.
- `tests/test_preprocess.py` passes all tests against a real Pong frame. Shape, dtype, and value range all verified.

## What's open (carrying forward)

- Replay buffer will be deferred to next session.

## Anything surprising or worth flagging

- Coming into this session, I had no prior experience with OpenCV. Getting familiar with the grayscale conversion and resize API was the first real step before writing `src/preprocess.py`.