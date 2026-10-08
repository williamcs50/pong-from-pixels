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

**Aspiration:** All of that, plus the first built component committed to the repo, the replay buffer or the preprocessing pipeline, not decided yet. Code, not just a plan.

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

**Aspiration:** The full pipeline runs end to end as a smoke test, two separate paths, both with no shape, dtype, or device errors. Acting: `env.reset()` through the preprocessor, the transform, the network, `select_action`, and `env.step()`, on the live batch of one state. Learning: a batch sampled from the buffer, through the transform and the network, no action, no `env.step()`.

---

## What landed today

- No code was committed today. The session went entirely to verification and design, and it earned its keep anyway.
- The buffer's storage dtype was decided on evidence, not habit. Two state sized arrays at capacity 100,000 cost 5.6448 GB under uint8 versus 22.5792 GB under float32, checked against 31.86 GB of system RAM. Float32 technically fits, but it buys nothing, since the emulator only ever produces uint8 pixels in the first place.
- Traced the transform's round trip and found it wasn't actually invertible. `astype(np.uint8)` truncates instead of rounding, so a value that isn't an exact multiple of 1 over 255 comes back slightly wrong after the round trip, silently, with no error thrown.
- Rounding before the cast would have shrunk that error, not removed it. The real fix was noticing that two conversions in the pipeline exist only to undo each other, the preprocessor dividing by 255 and the transform multiplying it back. Deleting both instead of patching the cast removes the bug entirely rather than making it smaller.
- The resulting design is written down precisely enough to build from directly: the preprocessor stops after resize and returns uint8 with no normalization, the buffer takes it raw with no conversion in `push`, and one shared transform sits on the read side, used identically by the training path and the action selection path.

## What's open (carrying forward)

- Floor was not met. Nothing was written today, only verified and designed.
- The preprocessor change itself: stop normalizing, return uint8, and stack the frames on the last axis instead of the first.
- The three preprocessor tests, updated to match the new dtype and shape.
- `architecture.md`, corrected again to reflect the new design.
- The shared transform function, still unwritten.
- The round trip test, still unwritten.
- The end to end smoke test is still the aspiration.

## Anything surprising or worth flagging

- Caught the `push` problem last week by reading the code instead of trusting that a conversion existed. Today's entire session was downstream of that one habit.
- Three and three quarter hours went to verification and design, zero to code. Not a failure on its own, but a pacing data point worth sitting with honestly: whether the diagnostics ran long because they needed to, or because running one more check felt safer than committing to a claim on the page.

---

# Saturday: Preprocess Pipeline

**Date:** 2026-08-22

**Floor:** `src/preprocess.py`: `preprocess()` stops normalizing, returns uint8; `_get_stacked()` stacks on the last axis; output `(84, 84, 4)` uint8. Six of seven `test_preprocess.py` tests updated for the new dtype/shape/axis (`test_preprocess_shape` is the one that doesn't move). The shared transform function: uint8 HWC in, cast to float32, divide by 255, transpose to NCHW. The isolated value test on the transform (known input, exact expected output). The round trip test: frame through `preprocess`, through `buffer.push`/`buffer.sample`, through the transform, checked against transforming the original frame directly. `architecture.md` corrected again to say uint8 HWC. All committed and pushed.

**Aspiration:** The full end to end smoke test, two separate paths, both with no shape, dtype, or device errors. Acting: `env.reset()` through the preprocessor, the transform, the network, `select_action`, and `env.step()`, on the live batch of one state. Learning: a batch sampled from the buffer, through the transform and the network, no action, no `env.step()`.

---

## What landed today

- `src/preprocess.py` no longer normalizes. `preprocess()` returns the resized uint8 frame as-is, and `_get_stacked()` stacks on the last axis instead of the first, so `reset()`/`step()` return `(84, 84, 4)` uint8, matching the replay buffer's storage exactly.
- Six of seven `test_preprocess.py` tests updated for the new dtype, shape, and axis. All seven pass against a real Pong frame.
- `src/transform.py` built: one function, uint8 HWC in, single frame or batch, normalized float32 NCHW out on a given device. Handles the missing batch axis on a single frame internally, so neither caller has to know which shape it started with.
- `tests/test_transform.py` built: an isolated value test against hand-computed expected output, and a round trip test, a frame through `Preprocessor`, through `ReplayBuffer.push`/`sample`, through the transform, checked against transforming the original directly. Both pass.
- `architecture.md` corrected: the diagram now shows `Transform` in both the collection and training paths, Preprocessing's output and the network's input are back to matching the actual code.

## What's open (carrying forward)

- The full end to end smoke test is still the aspiration: `env.reset()` through the preprocessor, into the buffer, a sampled batch through the transform and the network, out to an action through `select_action`, back through `env.step()`. No shape, dtype, or device errors anywhere in that chain.

## Anything surprising or worth flagging

- I updated the preprocess tests right after the code change, before running pytest against the unedited version. So the updates were driven by knowing what the new shape and dtype should be, not by real failure messages.

---

# Saturday: Smoke Test

**Date:** 2026-08-29

**Floor:** Make sure `scripts/visual_check.py` works correctly, and build the end to end smoke test as two separate paths. The acting path: `env.reset()` through the preprocessor, transform, the network, `select_action`, and `env.step()`, using the live batch of one state, asserting shape `(1, n_actions)`, dtype `float32`, and device at the network output, plus a valid action index. The learning path: a batch sampled from the buffer, through the transform and the network, no action selection, no `env.step()`, asserting shape `(batch_size, n_actions)`, dtype `float32`, and device at the network output. No NaNs anywhere in either path.

**Aspiration:** Start the full training loop.

---

## What landed today

- `scripts/visual_check.py` corrected: no more `*255` overflow, `transform_output.png` now compares against the same frame as `preprocessed_frame.png` instead of a later, mutated one, episode boundaries pair `env.reset()` with `prep.reset()` instead of stepping across them, and the ball candidate printout is cropped to the field so it's not buried under the score strip and the bottom wall. It used to call `preprocess()` directly, so it structurally could not see last week's stacking axis change. Now it runs through the buffer, sample, and the shared transform, the same path training will actually use. Reran it just now, still clean.
- `tests/test_preprocess.py` now has two real assertions instead of eyeball checks: `test_newest_frame_at_last_index` and `test_reset_clears_stale_stack`. Frame ordering was confirmed independently too, by differencing frames and tracking the ball's raw coordinates, scaled by 84 over 160 to match the preprocessed frame.
- `tests/smoke_test.py` built: two paths, acting (`env.reset()` through the preprocessor, the transform, the network, `select_action`, `env.step()`) and learning (a batch sampled from the buffer through the transform and the network), both asserting shape, dtype, device, and no NaN on the network output. Ten steps and a batch of eight ran clean.

## What's open (carrying forward)

- The training loop itself: loss, `backward()`, the optimizer step, target network syncing on a schedule, and epsilon decay over time. Nothing built yet, that's today's aspiration. First thing to write next session is the weight update test: snapshot the Q and target weights, one optimizer step on Q with a synthetic batch, assert Q changed and target didn't.
- Action is hardcoded to `0` in `visual_check.py`'s `buffer.push()` call. Needs to be the actual action taken once the training loop exists.
- `smoke_test.py`'s learning path only checks `current_batch` from `buffer.sample()`. The other four fields, action, reward, next state, done, get discarded with `_` and never asserted.
- `run_acting_loop` gets called once by each test function, so the smoke test builds a fresh env and network twice for one run instead of once.

## Anything surprising or worth flagging

- I want to remember the `*255` bug as a category, not just an incident. Multiplying already normalized uint8 data by 255 wraps around instead of erroring, and the result was still a recognizable, plausible looking Pong frame, just inverted like a photo negative. It survived my eyeball checks because it looked fine. That's the dangerous kind of bug, the one that doesn't crash.
- I added a comment in `visual_check.py` explaining why pushing a cross episode `next_state` on a terminal transition is safe: the Bellman target masks the bootstrap term by `(1 - done)`, so that `next_state` never actually gets read. I wrote it so a future refactor doesn't decouple `done` from that assumption without knowing why it mattered.
- The smoke test's wording in this file was wrong in three places, not just today's Floor but the Aspiration lines from two earlier sessions too. All of them described a sampled batch feeding directly into `select_action`, which would crash on `select_action`'s `.item()` call. It stayed uncaught until I had to write the actual code and be specific about which tensor goes where. I reworded all three into two separate paths instead of one chain: the acting path takes the live batch of one state through `select_action`, and the learning path takes the sampled batch through the transform and the network only, `select_action` never sees it.
- I set `EPSILON = 0.0` in the smoke test on purpose, not as a leftover default. A nonzero epsilon would route the action through `select_action`'s random branch, which never reads `q_values`, so the forward pass and the network's actual output would never get exercised. A test set up that way would pass without testing anything.
- I stopped at the aspiration boundary on purpose today instead of drifting past it. Floor was met, both items, and once that was confirmed I pushed the training step function to next session deliberately, not because time ran out.

---

# Saturday: Training Step

**Date:** 2026-09-12

**Floor:** The two open pre flight items closed with actual output, not memory: four ball coordinates across the stack, and right paddle frames at both extremes. If either one fails, that's the day and the training step moves. The VRAM baseline written into this file, plus the one line on score digits as a known choice. The training step written and verified on a fixture where I know the answer by hand: loss finite, Q-network weights move, target network weights don't.

**Aspiration:** That training step integrated into an end to end loop that runs a few hundred steps without falling over.

---

## What landed today

- `src/train.py` built: `train_step` takes the raw batch from `ReplayBuffer.sample()`, calls `transform` itself, and returns the loss as a float.
- `tests/test_train.py` built: Q params move and target params don't, the loss matches a hand computed value on a fixture where both networks return fixed Q values, and a terminal batch ignores `next_state`. All three pass.
- `scripts/visual_check.py` reworked: paddle height derived per run from the raw frame, the stack asserted on every step and after every reset, the ball tracked through all 4 slots, and an episode boundary forced at step 10 so that branch finally runs. Reran it, clean.
- `scripts/check_environment.py` now reports the baselines: 6144 MiB VRAM total with 5081 free, 21.0 MiB for both networks, 31.86 GiB host RAM, 5.26 GiB for the buffer at capacity 100,000.
- `docs/architecture.md`: no crop before the resize and the buffer storing every frame twice both recorded as known choices.

## What's open (carrying forward)

- The loop. Still the aspiration, didn't land. First thing is proving a fixed batch trained repeatedly drives the loss toward zero, since nothing shows `train_step` learns yet.
- `env.action_space` is unseeded, so the last run had both paddles fully visible and neither extreme got checked. Seeding it plus a driven UP and DOWN rollout would put both in every run.
- `FORCED_BOUNDARY_STEP` is scaffolding. Comes out once real episodes terminate inside the loop.
- Action is still hardcoded to `0` in `buffer.push()`. Carried from last session, resolves when the loop supplies a real action.

## Anything surprising or worth flagging

- I widened a failing expectation instead of fixing the derivation. My expected 3 or 4 started flagging legitimate frames at 6 and 7, and I widened the band to 1 through 7 rather than ask why. A band that can't fail reads as green while it's stopped testing anything.
- The motion check used to drop empty slots and difference what was left, so an occluded middle slot made slots 0 and 3 look one frame apart. Today's run had slot 3 empty and reported 2 of 3 pairs instead of hiding it.
- The buffer stack and the same stack through `transform` came back with identical ball coordinates. That's the slot ordering proof, and until today it was me looking at a PNG.

---

# Saturday: Reorient and Train

**Date:** 2026-10-03

**Floor:** Reorient with the codebase and complete the masking test.

**Aspiration:** Begin training the agent.

---

## What landed today

- Reoriented: walked the pipeline file by file to refamiliarize myself with the codebase.
- Added `test_terminal_target_equals_reward` to `tests/test_train.py`. It uses constant networks so the current Q value is `[-1.0, 5.0]` and the best next state Q value is `10.0`. Every transition is marked done, so the target is the reward alone and the hand computed loss is 4.0.
- Verified the new test by breaking the `targets` line two ways. One multiplied the reward by the future value and gave 17.5. The other zeroed the whole target on terminal transitions, reward included, and gave 2.5. I computed both losses by hand before running, and both failures matched.
- Did a comment and docstring consistency pass across all test files.
- 27 of 27 pass.

## What's open (carrying forward)

- The overfit test: prove that `train_step` actually learns by repeatedly training on a fixed batch and checking that the loss moves toward zero.
- Five training decisions need answers before the loop gets built. Each one is its own item below.
- When to call `train_step`. Every env step, or once every few? That sets how many gradient updates the agent gets for each frame it plays.
- When to start training. The buffer starts empty, so the first updates would sample from a handful of transitions. How much data should get collected before the first update?
- What happens at an episode boundary. The env resets, but so does the frame stack, along with anything else that only makes sense inside one episode.
- `terminated` vs. `truncated`. `terminated` means the game actually ended, so the target is just the reward. `truncated` means a time limit cut the game off when it could have kept going. Right now both get pushed as `done`, which stops bootstrapping for both.
- How often to sync the target network. Copy the Q network into it too often and the targets move with every update. Too rarely and they go stale.
- The tiny training loop. Same structure as the real loop, just small enough to run in a minute or two on the MacBook Air. Proves the pipeline holds together, not that the agent learns.
- Launching the first real run on the PC next Saturday, with checkpointing. Needs a launch checklist.
- Whether the MacBook Air's RAM is enough for the 5.26 GiB replay buffer.
- The three carried items from Sept 12: action is still hardcoded to `0`, `FORCED_BOUNDARY_STEP` is still scaffolding, and the action space is still unseeded.

## Anything surprising or worth flagging

- `test_terminal_batch_ignores_next_state` wasn't enough on its own. It only checked that changing the next state didn't change the loss when `done` was true. I broke the target so it zeroed both the future value and the reward on terminal transitions, and that test still passed. In Pong that bug erases every point scored or lost. The new test checks that the reward is still the target for a terminal transition, and it caught the same bug.

---

# Monday: Overfit Test

**Date:** 2026-10-05

**Floor:** Finish the fixed batch overfit test and get it passing, proving that `train_step` can actually learn from a fixed batch.

**Aspiration:** Finish the overfit test, resolve the five remaining training loop decisions and have them in writing, and have the training loop ready to implement tomorrow.

---

## What landed today

- Added `test_train_step_overfits_fixed_batch` to `tests/test_train.py`. It calls `train_step` 200 times on one seeded batch of four transitions and asserts the final loss falls below 1% of the initial loss. It passes.
- Answered the five training decisions for the real run:
  - **When to train:** every 4 env steps.
  - **When to start:** after 50,000 transitions are in the buffer.
  - **Episode boundary:** reset the env and the frame stack, and keep the buffer, the networks, and the optimizer.
  - **Terminated vs. truncated:** only `terminated` stops bootstrapping. A truncated game was cut off by a time limit and could have kept going, so its next state still has value.
  - **Target sync:** every 10,000 gradient updates.
- Built `train()` in `src/train.py` and `scripts/tiny_train.py`, which calls the same function with tiny settings. One loop, two configs, so the tiny run actually says something about the real one. Actions go through `select_action` instead of inline epsilon greedy logic, so that behavior lives in one tested place.
- Ran the tiny loop on the MacBook Air. It ran to completion, started training after warmup, synced the target on schedule, reset cleanly after an episode ended, and kept the loss finite throughout.

## What's open (carrying forward)

- The real run config. `scripts/tiny_train.py` covers the tiny run, but the full settings still need their own script. The epsilon schedule and the total step count for the real run aren't decided yet.
- Logging and checkpointing. Episode rewards, losses, and epsilon need to go to a file so the real curve survives the run. Model weights need saving on a schedule so a crash doesn't lose everything and the clips have checkpoints to come from.
- A launch checklist for the first real run on the PC, including confirming the loop prints `Training device: GPU` before leaving it to train.
- The overfit test hasn't been checked against a deliberate break yet. Removing `optimizer.step()` should make it fail.
- Still in `scripts/visual_check.py` from Sept 12: action hardcoded to `0`, `FORCED_BOUNDARY_STEP`, and the unseeded action space. The training loop handles the first two properly now, so the script can follow.

## Anything surprising or worth flagging

- The overfit test's 1% bar is looser than it needs to be. The loss went essentially to zero, so a partly broken update could still pass.
- My first loop draft had `stacked - next_stacked` instead of `stacked = next_stacked`. It wouldn't have crashed. It would have quietly stored each episode's first frame as every state in that episode, the same category as the `*255` bug.
- The first tiny run printed nothing, and not crashing doesn't prove much. I added prints for episode ends, the loss, and target syncs, plus a check that stops the run if the loss isn't finite.

---

# Tuesday: Launch Prep

**Date:** 2026-10-06

**Floor:**

1. **Prove the overfit test can catch a broken training step:** deliberately remove `optimizer.step()`, run the test, and confirm that it fails for the expected reason. Restore the update afterward and confirm the test passes again.
2. **Finalize the exploration schedule and training length for the real run:** decide the starting epsilon, minimum epsilon, decay schedule, and total number of environment steps the agent will train for.
3. **Finalize the action space for the real run:** decide whether the agent will use 3 actions or all 6 available actions, and confirm that the environment, network output size, and action selection code all match the decision before launch.
4. **Create the complete real run configuration and launch checklist:** put the chosen hyperparameters and run settings in a dedicated config/script, and document the checks needed before starting the PC run, including device, environment, model, replay buffer, output paths, and temperature during warmup.

**Aspiration:** 

1. **Add the minimum logging and checkpointing needed for the first real run:** write episode reward, episode number, and relevant training information to disk, and save model checkpoints on a defined schedule so the run produces recoverable data and usable milestone models from the beginning.
2. **Bring `scripts/visual_check.py` in line with the real training code:** remove the hardcoded action `0`, remove `FORCED_BOUNDARY_STEP`, and seed the action space so the visual check uses the same action selection and episode boundary behavior as the training loop. Run the script afterward and confirm it still produces the expected gameplay behavior.
3. **Launch Run 1 on the PC:** verify the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and begins updating the network without errors. If this point is reached with logging and checkpointing fully operational, let the run continue as the actual first real training run.

---

## What landed today

- Proved the overfit test can fail. Baseline passed at a ratio of 8.446e-05. With `optimizer.step()` commented out the loss never moved, 2.880e-01 to 2.880e-01, ratio 1.0. With `optimizer.zero_grad()` commented out the loss rose instead, ratio 3.2. The test passed again after each restore, at 1.744e-07 and 5.432e-04, and `git diff src/train.py` comes back empty.
- Keeping the 1% bar on the overfit test. Six healthy runs ranged from 1.744e-07 to 5.432e-04 and the two breaks sat at 1.0 and 3.2, so the bar sits in a gap of nearly four orders of magnitude, 18x above the worst healthy run and 100x below the nearest failure. Nothing I can produce lands in that gap.
- Set the exploration schedule and run length:
  - **Epsilon start:** 1.0.
  - **Epsilon end:** 0.01, so the back half of training has very little random action interference and improvements after 250k mostly reflect learning.
  - **Decay steps:** 250,000, so exploration falls away early enough that most of Run 1 tests the learned policy.
  - **Total steps:** 2,000,000, leaving 1.75M steps after the decay to show whether the policy improves and stabilizes.
- Decay counts from env step 0, not from the end of warmup. Verified in the code. Epsilon is 0.80 at step 50,000 where training starts, and reaches 0.01 at step 250,000.
- Sticky actions off for Run 1. `ALE/Pong-v5` defaults to `repeat_action_probability=0.25`, the DQN papers had none, and the +18 aspiration is anchored to that setup. Passed as a config value, not hardcoded.
- Action space: all 6. The DQN papers used each game's minimal legal action set, which for Pong is six, and the +18 aspiration depends on that comparison. The six collapse to three behaviors in pairs, so the full set is kept for comparability, not extra control.
- The demo's three bars collapse each pair with `max`. The policy is argmax over all six, so the tallest bar is always the action the agent chose. Mean or sum can put a bar on top while the agent does something else.
- Rewrote the pre registered prediction so it can fail. A plateau is the trailing mean over the last 100 episodes moving less than 1.0 across 200 episodes. It should cross 0 before step 1.2M, and still being below 0 at 1.2M means the schedule is insufficient.
- Replay buffer stays at 100,000, 5.26 GiB of the 18.69 GiB free. 200k would take 10.5 GiB and risk paging.
- Decided the last three hyperparameters. `batch_size` 32 and `gamma` 0.99 both match Nature and neither depends on the optimizer. `learning_rate` 1e-4 is a deliberate deviation, since Nature used RMSProp at 0.00025 and rates do not carry over to Adam.
- Added `repeat_action_probability` and `seed` as parameters to `train()`. The sticky value is read back from the emulator at startup rather than echoed from the argument, so what prints is what the env is actually running.
- Implemented seeding. `np.random.seed` covers the exploration draw in `select_action` and the batch draw in `ReplayBuffer.sample`, `torch.manual_seed` covers network init, and the env takes the seed on the first reset only.
- Verified the seeding two ways. Two CPU runs at seed 0 came back byte identical, and a run at seed 1 differed in loss, episode length and reward. The matching pair proved something fixes the RNG, and only the seed 1 run proved it's the seed parameter.
- Added a startup block printing device, seed, sticky actions, action meanings and network output size, plus an assert on the output size from a real forward pass. Floor 3's confirmations now happen every run instead of by hand, and the extra forward pass was checked against an earlier CPU run to confirm it does not perturb anything.
- Wrote `scripts/real_train.py`. Its key set matches `train()`'s signature and `tiny_train.py` exactly, so the two configs cannot drift apart. Derived figures check out: 487,501 gradient updates, 48 target syncs, epsilon 0.802 at the first update, 5.26 GiB for the buffer.
- Changed `epsilon_end` in `scripts/tiny_train.py` from 0.1 to 0.01. It was the only value differing from the real config without being a time or memory number, which made that file's header comment false.
- Wrote `docs/launch_checklist.md`. Three sections by moment in the run, and only the checks the code cannot do for itself.
- Accepting GPU nondeterminism for Run 1. The case study's claim is that DQN learns Pong, not that one trajectory replays, so bit reproducibility is not worth an unmeasured slowdown on a run that has to finish before Oct 12.
- Verified the `nvidia-smi` logger command before putting it in the checklist. All five query fields are valid and `-f` writes clean ASCII. Idle baseline is 40 C at 315 MHz, with 1192 MiB of the 6144 already in use, so VRAM headroom at launch is nearer 4.9 GiB than 6.

## What's open (carrying forward)

- The whole aspiration moves to tomorrow: logging, checkpointing, `scripts/visual_check.py`, and the Run 1 launch. Still inside the Oct 6 to 8 block.
- Three checklist items cannot be ticked until logging and checkpointing exist. The output path decision to sit outside OneDrive has nowhere to live until then either.
- No evaluation path. The bar is defined over 100 evaluation episodes, but `train()` has no evaluation epsilon and no evaluation loop, so the 1.2M prediction uses training episodes as a proxy.
- Which direction `RIGHT` moves the paddle is unverified. The names are joystick names, so labelling the demo's bars from them could invert Up and Down.
- The determinism throughput comparison, for the Run 2 decision.
- Find one implementation that actually uses Adam at 1e-4, so the writeup can name it instead of saying it is common.
- Still in `scripts/visual_check.py` from Sept 12: action hardcoded to `0`, `FORCED_BOUNDARY_STEP`, and the unseeded action space.

## Anything surprising or worth flagging

- Sticky actions are on by default in `ALE/Pong-v5` at 0.25. At the epsilon I had just chosen, the environment would have overridden the policy about 25 times more often than my own exploration does, which would have made the whole argument for 0.01 over 0.1 meaningless. I reasoned out the schedule before checking the environment defaults.
- The six actions are not six behaviors. My first written reason for keeping all six was that the full action space lets the agent learn a complete control policy, which is not true for Pong. The decision survived, the reason did not.
- Seeding does not make a GPU run reproducible, and the divergence is behavioral rather than numerical. Two GPU runs at seed 0 ended episode 1 at steps 900 and 928, because a perturbed weight flips an argmax, which changes the action, which changes the data.
- Two identical runs weren't enough to prove the seeding worked. They showed the run was deterministic, not that seed was what made it so. Only changing the seed and watching the output change proved the parameter threads through.
- 1192 MiB of the 6144 is already in use at idle, so VRAM headroom at launch is nearer 4.9 GiB than 6.

---

# Wednesday: Run 1 Launch

**Date:** 2026-10-07

**Floor:**

1. **Add the minimum logging and checkpointing needed for the first real run:** write episode reward, episode number, and relevant training information to disk, and save model checkpoints on a defined schedule so the run produces recoverable data and usable milestone models from the beginning.
2. **Bring `scripts/visual_check.py` in line with the real training code:** remove the hardcoded action `0`, remove `FORCED_BOUNDARY_STEP`, and seed the action space so the visual check uses the same action selection and episode boundary behavior as the training loop. Run the script afterward and confirm it still produces the expected gameplay behavior.
3. **Launch Run 1 on the PC:** verify the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and begins updating the network without errors. If this point is reached with logging and checkpointing fully operational, let the run continue as the actual first real training run.

**Aspiration:**

1. **Use real numbers in the launch checklist.** Fill in `temperature below ___ °C` before launch. Record warmup steps per second, projected time for 2,000,000 steps, and the launch commit hash. If Run 1 will extend past the diagnosis block, reduce the step count or change the run now.
2. **Build relaunch instrumentation.** Log Q value drift, actual epsilon, replay dynamics, and episode reward. Verify it on a tiny run while Run 1 collects the baseline curve. The instrumentation will require a restart.
3. **Close the evaluation gap.** The criterion is average reward above 0 over 100 evaluation episodes, but `train()` has no evaluation loop or evaluation epsilon. Add checkpoint evaluation with 100 episodes at fixed low epsilon so the 1.2M prediction uses the actual criterion.

---

## What landed today

Floor 1 and Floor 2 are met. Floor 3 moves to tomorrow.

- `episodes.csv` written one row per finished episode, pushed to disk as each row lands: `episode`, `env_step`, `reward`, `episode_steps`, `epsilon`, `gradient_updates`, `buffer_size`, `mean_loss`, `wall_clock_s`. `buffer_size` is beyond the agreed schema, added so Floor 3's "fills the replay buffer" clause is observable rather than inferred. `mean_loss` is blank rather than 0.0 when no update ran.
- `src/checkpoint.py` built. A checkpoint holds Q weights, target weights, Adam state, the numpy, torch and CUDA RNG states, and the `env_step`, `episode` and `gradient_updates` counters, without which a resume restarts the epsilon schedule and the target sync. The buffer is excluded at 5.26 GiB a save. Writes go to a temp name and are renamed, so dying mid write cannot leave a truncated newest checkpoint. Step 0 plus every 50,000 steps, so 40 files at 27 MB.
- `src/run_paths.py` built. One `RUNS_ROOT` at `~/pong-runs`, outside the OneDrive sync root, imported by both launch scripts so they cannot drift. Tiny runs sit under `~/pong-runs/tiny` so they never take a real run number.
- `run_config.json` per run holds every config value plus the device, GPU name, sticky actions readback, action count, `git_dirty` and the full commit hash. `train()` refuses to start if `episodes.csv` already exists, and refuses on a dirty tree unless `allow_dirty` is passed, since a hash recorded with modified files does not describe what ran. Untracked files count, because `src/checkpoint.py` and `src/run_paths.py` were untracked this morning and `train()` imports both.
- `real_train.py` now takes `--total-steps`, `--output-dir` and `--allow-dirty`, so the warmup pass is Run 1 with one number changed rather than a separate script. Since epsilon decays from step 0, a 70,000 step warmup is literally Run 1's first 70,000 steps.
- `tiny_train.py` went from 1,000 to 1,500 steps and now asserts the log holds at least one row. Episode 1 ended at 838, 838, 935, 838 and 838 across five runs, so 1,000 left only 65 steps of margin before a run would log nothing and still print `done:`.
- `scripts/play_checkpoint.py` built: loads a checkpoint, plays whole episodes, and reports mean reward, an action histogram, and the counters from the file. It verifies the load rather than assuming it, and the verdict only prints at 100 episodes or more.
- `scripts/visual_check.py`: hardcoded action `0` replaced with the sampled action, `FORCED_BOUNDARY_STEP` removed, and one `SEED = 42` feeding `np.random.seed`, `env.reset` and `env.action_space.seed`. `BUILD_STEPS` went from 60 to 1,200, derived from a measurement rather than picked: the first episode terminates at 1100 at seed 42 and 959 at seed 7. Paddle labels renamed to `left (opponent)` and `right (agent)`, since CPU collided with the training device.
- Reran it clean: transform round trip diff 0, one ball component, left paddle height 7 inside the derived set of 6 or 7, column deltas 2.0, 2.0, 2.0 with no sign changes, and one real episode boundary at step 1100.
- Verified rather than assumed: the overwrite refusal raises, a blank `mean_loss` appears when nothing trained, the final checkpoint fallback produces 0, 400, 800, 900 at `total_steps` 900, a loaded checkpoint returns the counters the run printed, `taskkill /F` left 48 rows intact with `close()` never running and no `.tmp` behind, and 100 episodes played at 4.05 percent random against the 0.05 target. Both new assertions were also proven able to fail. 28 of 28 tests pass.

## What's open (carrying forward)

- **All of Floor 3.** The warmup pass, the rate measurement, the projection, and the Run 1 launch.
- Pre launch items not done: the temperature blank in `docs/launch_checklist.md`, Windows Update not paused, and sleep on AC still at 1200 seconds rather than never, which matters because Windows measures idle by user input and not GPU load. The commit is local only.
- No test covers logging or checkpointing. Today's evidence is manual runs, not something the suite rechecks.
- Aspiration 2 was not started. Aspiration 3 is partly closed, since `play_checkpoint.py` is the evaluation kernel but nothing wires an evaluation pass into `train()`, so the 1.2M prediction still uses training episodes as a proxy.
- `train()` runs a forward pass every step and `select_action` discards it on the random branch, so almost all of that work is wasted during the warmup at epsilon near 1.0.
- The driven UP and DOWN rollout in `visual_check.py` is still open from Sept 12, and which direction `RIGHT` moves the paddle is still unverified.
- Proven by construction rather than by test: a kill landing mid checkpoint save, and the `True` side of the evaluation verdict, which stays unexercised until a checkpoint wins.

## Anything surprising or worth flagging

- Loading a checkpoint is where a real bug surfaced, and saving one would never have found it. `map_location` moves every tensor onto the GPU, including the RNG state, and `torch.set_rng_state` rejects anything that is not a CPU ByteTensor, so a resume would have crashed. The fix is `.cpu()` before restoring.
- Seeding the action space alone did not make `visual_check.py` repeatable. Two seeded runs still disagreed, and only on the two sampled stack sections, because `ReplayBuffer.sample` draws with `np.random.choice` and nothing seeded `np.random`.
- Seeding that script was a trade, not a pure win. Unseeded, every run inspected a different frame, so the paddles got checked at a variety of heights by accident. Seeded, it is the same frame forever, and a renderer bug at the bottom extreme is now permanently invisible rather than occasionally caught.
- The failure criterion written in advance for the playback test was wrong. It said -21 every episode in minimum steps would mean the network was not influencing the action. The greedy policy picks action 2 on 2,199 of 2,292 steps, pinning the paddle where it never touches the ball, and with no contact the ball's path does not depend on the paddle, so every episode is identical. A learned but degenerate constant policy produces exactly the signature called failure. The action histogram is the real discriminator and is now printed.
- The 806 to 816 env steps per second from the kill tests is not the figure the projection needs. `gradient_updates` is 0 on every row, so it is the fill phase only, and projecting 2,000,000 steps from it would give 41 minutes and be badly wrong.
- The run is bit for bit deterministic until the first gradient update. Two kill tests produced identical episode boundaries through step 41,681, because no updates means the weights never change and both RNG sources are seeded.

---

# Thursday: Warmup and Launch

**Date:** 2026-10-08

**Floor:**

**Launch Run 1 on the PC:** verify the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and begins updating the network without errors. Before launching, get the checklist numbers from a warmup pass at Run 1 settings: load temperature, steps per second after warmup, and the projected time for the full run. If the projected finish does not land before Oct 12, when the diagnosis block opens, cut the step count first and restate the pre registered crossing. If this point is reached with logging and checkpointing fully operational, let the run continue as the actual first real training run.

**Aspiration:**

**Close the evaluation gap.** Run the checkpoint evaluation across a whole run rather than one checkpoint at a time, writing a row per checkpoint so the evaluation curve can be read against the bar instead of training reward standing in for it. Done means that file exists and Run 1's first checkpoints are in it. Evaluation shares the card with Run 1, so it runs only after Run 1's rate and temperature readings are taken, and only on the first few checkpoints, so a slowdown afterwards is attributable rather than confused with throttling.

---

## What landed today

- Ran the warmup pass at Run 1 settings, 120,000 steps into `run-00-warmup`. 131 episodes, 17,501 updates, one target sync at update 10,000, and checkpoints at 0, 50,000, 100,000 and 120,000. `run_config.json` recorded commit `f9ce3b66d9a8c41105676bba87ae8453d4e9cb82` with `git_dirty` false, matching `HEAD`. The 120,000 file is the first time the final checkpoint fallback has fired under the real config.
- `step_00050000.pt` came out at 27 MB against step 0's 13.5 MB, so Adam state is in it and the loop does update before it saves.
- Measured the three checklist numbers. 252.1 steps per second after warmup, measured across 274 seconds from step 50,811 to 119,876 rather than from adjacent rows. That projects 2.20 hours for 2,000,000 steps, which clears Oct 12, so I am not cutting the step count and the pre registered 1.2M crossing stands as written. Peak load temperature 52 °C against the 88 limit.
- The fill phase ran at 809 steps per second, matching yesterday's 806 and 816. Training is 3.2 times slower, which is why the fill rate was never the number to project from.
- Episode boundaries through step 49,987 matched yesterday's kill tests exactly and diverged after the first gradient update, which is the determinism property behaving as expected.
- Launched Run 1 at 13:55 into `run-01`, 2,000,000 steps, with commit `afb91fae0b1c1408009cb5385980f432c99f7ec7` recorded and `git_dirty` false. The loop based logger writes `gpu_log.csv` live this time, header included.
- Predicted the end of run reward before reading the rows: positive, around +10. My Oct 6 pre registration has the trailing mean crossing 0 before step 1.2M, which leaves about 800,000 steps of runway at epsilon 0.01. The buffer at 100,000 against Nature's 1,000,000 is why it could fall short, and 8,000,000 frames against Nature's 50,000,000 is why I am not predicting the +18 aspiration.

## What's open (carrying forward)

## Anything surprising or worth flagging

- The GPU is barely working. 27 to 39 percent utilization, 26 to 30 W on a card rated near 160, clocks at 800 to 1,000 MHz rather than near boost, and VRAM at 2,207 MiB of 6,144. At 252 steps per second with the card two thirds idle, the bottleneck is the CPU side: one Pong instance, the frame preprocessing, and the per step Python loop. That also explains the fill phase managing 809 steps per second with no GPU work in it.
- `nvidia-smi -f` buffers its output instead of flushing each sample. `gpu_log.csv` sat at 0 bytes for the whole run and only appeared when I stopped the process, so the first five minutes thermal check cannot be done live with that command.
- The checklist item "clock not dropping as temperature rises" would have flagged this healthy run. The clock swung from 1,290 down to 795 MHz while temperature stayed in the forties and fifties, which is the card idling down for lack of work. A falling clock only means throttling when the temperature is near the limit.

---
