# Saturday: Kickoff

**Date:** 2026-06-06

**Floor:** Pong rendering on screen with the random agent, and the repo live with its first commit.

**Aspiration:** The GPU verified and doing real ML work.

---

## What landed today

- Pong renders on the screen with the random agent.
- The repo is live.

## Anything surprising or worth flagging

- Coming into this project, I had no prior experience with the Gymnasium library. Getting familiar with its environment API was the first real hurdle of the day.

---

# Saturday: Architecture

**Date:** 2026-06-20

**Floor:** The GPU verified and working. A tensor on CUDA, a small matmul, confirmation that the 2060 is live. Also, a written architecture diagram in a markdown file, showing every component of the DQN with build or import labels and a sentence on why for each one.

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

- `src/preprocess.py` written and committed. Grayscale conversion, resize to 84 x 84, and frame stacking across a sliding window of 4 frames.
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

**Aspiration:** All of that, plus the Q-network and target network written and verified with a GPU forward pass. A dummy batch through CUDA, output shape confirmed at (B, n_actions), `loss.backward()` runs without error, and target weights decoupled and syncable. The HWC to NCHW transpose and /255 normalization also land here.

---

## What landed today

- `src/replay_buffer.py` was written and committed.
- `tests/test_replay_buffer.py` passes all tests: shapes and dtypes, circular overwrite, state and next_state pairing, and error handling when `batch_size` exceeds the number of stored transitions.

## What's open (carrying forward)

- Floor was met. Aspiration not reached. Q-network and target network deferred to next session.

## Anything surprising or worth flagging

- I assumed the replay buffer capacity would match the original DQN paper at 1 million transitions. Working through the actual memory budget brought it down. Each state is 84 x 84 x 4 = 28,224 bytes as uint8. Storing both state and next_state per transition doubles that to 56,448 bytes. With 16 GB RAM minus roughly 5 GB for OS, PyTorch, and the CUDA context, about 11 GB is available. That buys around 195,000 transitions. 100,000 was chosen as a conservative starting point.

---

# Saturday: Q-Network

**Date:** 2026-07-11

**Floor:** The day counts as a win if the Q-network and target network are both hand-built and verified in isolation, then committed. That means a dummy batch of shape (B, 4, 84, 84) on CUDA produces output of shape (B, n_actions), lands on the GPU, runs `loss.backward()` cleanly, and leaves the target network weights unchanged after an optimizer step on the Q-network until they are synced. The transpose and division by 255 also live in one consistent place on every path into the network.

**Aspiration:** If time allows, the stretch goal is to build and verify an epsilon-greedy action selector. With epsilon = 1, it should behave like uniform random selection, and with epsilon = 0, it should behave like deterministic argmax selection. That would exercise the full action path end to end: stacked frame, transform, network, and action.

---

## What landed today

- The Q-network was designed, built, and tested. Its shape, device placement, gradients, and output sanity were all confirmed. The transform site, activation placement, and dynamic action-count handling were all deliberate choices, and each one is easy to defend.

## What's open (carrying forward)

- The target network and action selector are still open. Both are small implementation steps, and the target network is basically a second QNetwork instance plus a simple weight-copy routine.

## Anything surprising or worth flagging

- The main surprise was how little code was needed to verify the Q-network end to end once the architecture and tests were in place.
- I did not fully account for the target network until the end, so I deferred it to the next session.

---

# Saturday: Target Network

**Date:** 2026-07-25

**Floor:** The day counts as a win if the target network is hand-built and verified: snapshot its weights, take an optimizer step on the Q-network, assert the target is unchanged, sync with `load_state_dict`, assert they now match, and confirm target params are never held by the optimizer. The action selector should also be built and verified: epsilon = 1 behaves like uniform random selection, epsilon = 0 behaves like deterministic argmax, and the forward pass runs under `no_grad`.

**Aspiration:** If time allows, the stretch goal is to close the seam between preprocessing and the network: one transform function, shared by the training path and the action-selection path, that reconciles the preprocessor's normalized, channel-first output with the replay buffer's raw uint8 storage and the network's expected input. Then run one integration smoke test that exercises the full pipeline end to end. `env.reset()` through the preprocessor, through that shared transform, through the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the same transform and network again. No shape, dtype, or device errors anywhere in that chain would count as the stretch goal met.

---

## What landed today

- The target network was designed, built, and tested. Its isolation from the optimizer, its frozen weights through a gradient step on the online network, and its equality with the online network after a sync were all confirmed. Hard-copy sync via `load_state_dict` was chosen over Polyak averaging. It matches the original Atari DQN paper, and it's simpler to test: a discrete frozen-then-synced event instead of a continuous per-step drift.

## What's open (carrying forward)

- The action selector is still open. It gets its own file, `src/action_selector.py`, and its own test. Q-values and epsilon in, an integer action out. Epsilon = 1 should behave like uniform random selection, epsilon = 0 like deterministic argmax, and the forward pass through the network should run under `no_grad`.
- The shared transform function is still open: one function, called by both the training path and the action-selection path, that reconciles the preprocessor's normalized `(4, 84, 84)` float32 output with the replay buffer's raw `(84, 84, 4)` uint8 storage and converts it into what the network expects. This doesn't exist on either path yet, it isn't a matter of deduping a copy that's already there.
- The integration smoke test is still open: `env.reset()` through the preprocessor, the shared transform, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the same transform and network again.

## Anything surprising or worth flagging

- It had been two weeks since I'd last looked at this code, so getting back up to speed on `q_network.py` took longer than expected. Building the target network and its tests ended up taking more time than planned as a result.

---

# Saturday: Action Selector

**Date:** 2026-08-08

**Floor:** The action selector hand-built and verified. `src/action_selector.py` written, kept separate from the network so it only takes in Q-values and epsilon, and `tests/test_action_selector.py` covering determinism at epsilon 0, uniform exploration at epsilon 1, correct return type, and valid action range. Committed.

**Aspiration:** If time allows, the stretch goal is the transform layer. Two path-specific functions, one for the training path and one for the action selection path, sharing a contract rather than code, that reconcile the preprocessor's normalized 4 x 84 x 84 float32 output with the replay buffer's raw 84 x 84 x 4 uint8 storage and convert it into what the network expects. Neither exists on either path yet, so it is not a matter of combining code that already exists in two places, it is new code. If that lands, the next stretch goal is one integration smoke test that runs the whole pipeline end to end. `env.reset()` through the preprocessor, the transform, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the transform layer and the network again, with no shape, dtype, or device errors anywhere in that chain.

---

## What landed today

- `src/action_selector.py` is built. It takes Q-values, epsilon, and the number of actions, and returns a plain Python int, either the argmax action or a random one. I kept it separate from the network on purpose, so whatever calls the network still owns the `no_grad` context, not the selector.
- `tests/test_action_selector.py` passes everything I wanted covered. Epsilon 0 always picks the same argmax action, epsilon 1 explores every action roughly evenly, the return type is a real Python int on both branches, and the action always falls inside the valid range. I caught a real bug while writing these tests, the random branch was returning a numpy int instead of a plain Python int.
- While tracing this, I noticed `architecture.md` had the preprocessor's output and the network's input written down wrong. It described them as 84 x 84 x 4 in uint8, which is actually the replay buffer's storage shape, not what the preprocessor or network use. The real shape is 4 x 84 x 84 in float32. I fixed `architecture.md`.

## What's open (carrying forward)

- The transform layer is still open. Two path-specific functions, one for the training path and one for the action selection path, sharing a contract rather than code, that reconcile the preprocessor's normalized 4 x 84 x 84 float32 output with the replay buffer's raw 84 x 84 x 4 uint8 storage and convert it into what the network expects. Neither exists on either path yet, it is not a matter of combining code that already exists in two places.
- The integration smoke test is still open. `env.reset()` through the preprocessor, the transform layer, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the transform layer and the network again.

## Anything surprising or worth flagging

- Looking closely at push in `replay_buffer.py`, I found there is no real conversion logic there at all, just a plain assignment. Right now that fails loudly because the shapes do not match. But if I only fixed the shape and left the scaling alone, it would fail silently instead, since a shape-only fix would not correctly map [0, 1] floats into [0, 255] uint8 storage. Both issues come from the same place. I assumed a conversion existed between the preprocessor and the buffer, and it turns out I never actually wrote one.

---

# Saturday: Transform Layer

**Date:** 2026-08-15

**Floor:** `src/preprocess.py` updated: preprocessing stops after resize, returns uint8, frame stack shifts from axis 0 to the last axis, output shape (84, 84, 4) matching the buffer's storage exactly. `push` needs zero conversion logic. `test_preprocess_dtype`, `test_preprocess_value_range`, and the frame stack shape tests updated to match: uint8, 0 to 255, (84, 84, 4). `architecture.md` corrected in the same sitting to say uint8 HWC, not float32 CHW. The shared transform function built: uint8 HWC in, cast to float32, divide by 255, transpose to NCHW, once, right before the network. A test pushes a known frame through preprocess and the transform and asserts the actual returned values, not just shape and dtype, confirming the numbers coming out are the numbers that should come out. Committed.

**Aspiration:** The full pipeline runs end to end as a smoke test: `env.reset()` through the preprocessor, into the buffer, a sampled batch through the transform and the network, out to an action through `select_action`, back through `env.step()`. No shape, dtype, or device errors anywhere in that chain.

---

## What landed today

- No code was committed today. The session went entirely to verification and design, and it earned its keep anyway.
- The buffer's storage dtype got decided on evidence, not habit. Two state sized arrays at capacity 100,000 cost 5.6448 GB under uint8 versus 22.5792 GB under float32, checked against 31.86 GB of system RAM. Float32 technically fits, but it buys nothing, since the emulator only ever produces uint8 pixels in the first place.
- Traced the transform's round trip by hand and found it wasn't invertible. `astype(np.uint8)` truncates rather than rounds: 0.5 times 255 is 127.5, casting to uint8 gives 127, not 128, and dividing back gives 0.498039, not 0.5 back.
- Rounding before the cast would have shrunk that error, not removed it. The actual fix was noticing that two conversions in the pipeline exist only to undo each other, the preprocessor dividing by 255 and the transform multiplying it back. Deleting both instead of patching the cast removes the truncation bug entirely rather than making it smaller.
- The resulting design is written down precisely enough to build from directly: the preprocessor stops after resize and returns uint8 HWC with no normalization, the buffer takes it raw with no conversion in `push`, and one shared transform sits on the read side, casting to float32, dividing by 255, and transposing to NCHW, used identically by the training path and the action selection path.

## What's open (carrying forward)

- Floor was not met. Nothing got written today, only verified and designed.
- `src/preprocess.py` still needs the actual change: stop normalizing, return uint8, shift the frame stack from axis 0 to the last axis, output shape (84, 84, 4).
- `test_preprocess_dtype`, `test_preprocess_value_range`, and the frame stack shape tests still need rewriting to uint8, 0 to 255, (84, 84, 4).
- `architecture.md` still needs correcting again, to uint8 HWC instead of float32 CHW.
- The shared transform function itself is still unwritten: uint8 HWC in, float32 NCHW out, one function, both paths.
- The round trip test is still open: push a known frame through preprocess and the transform and assert on the actual returned values, not just shape and dtype.
- The end to end smoke test is still the aspiration: `env.reset()` through the preprocessor, into the buffer, a sampled batch through the transform and network, out to an action through `select_action`, back through `env.step()`.

## Anything surprising or worth flagging

- Caught the `push` problem last week by reading the code instead of trusting that a conversion existed. Today's entire session was downstream of that one habit.
- Three and three quarter hours went to verification and design, zero to code. Not a failure on its own, but a pacing data point worth sitting with honestly: whether the diagnostics ran long because they needed to, or because running one more check felt safer than committing to a claim on the page.

---