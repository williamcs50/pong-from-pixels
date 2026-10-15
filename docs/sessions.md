# Saturday: Kickoff

**Date:** 2026-06-06

**Floor:** Pong rendering on screen with the random agent, and the repo live with its first commit.

**Aspiration:** The GPU verified and doing real ML work.

---

## What landed today

- I got Pong rendering on screen with the random agent.
- I put the repo live with its first commit.

## Anything surprising or worth flagging

- I'd never used Gymnasium before this project, so learning its environment API was the first real hurdle.

---

# Saturday: Architecture

**Date:** 2026-06-20

**Floor:** The GPU verified and working. A tensor on CUDA, a small matmul, confirmation that the 2060 is live. Also, a written architecture diagram in a markdown file, showing every component of the DQN with build or import labels and a sentence on why for each one.

**Aspiration:** All of that, plus the first built component committed to the repo, the replay buffer or the preprocessing pipeline, not decided yet. Code, not just a plan.

---

## What landed today

- I verified the GPU works. `check_environment.py` puts a tensor on CUDA and runs a small matmul, which confirms the 2060 is live and PyTorch is using it.
- I documented the architecture in `docs/architecture.md`. Every component is named, with its interface and my reason for building it or importing it.

## Anything surprising or worth flagging

- I didn't know `pip install torch` gives you a CPU only build by default. There's no warning, and the GPU never gets used even when the hardware is ready. You have to install PyTorch from its own index to get the CUDA build.

---

# Saturday: Core Build

**Date:** 2026-06-27

**Floor:** `src/preprocess.py` written, tested with a real Pong frame, output verified (shape, dtype, and value range), and committed. Visual inspection of a saved frame still pending.

**Aspiration:** All of that, plus the replay buffer also written, verified in isolation (store, sample, overflow behavior), and committed. Two components, both verified, both in the repo.

---

## What landed today

- I wrote and committed `src/preprocess.py`: grayscale, a resize to 84 x 84, and a sliding stack of the last 4 frames.
- `tests/test_preprocess.py` passes against a real Pong frame, checking shape, dtype, and value range. I still haven't looked at a saved frame myself.

## What's open (carrying forward)

- The replay buffer moves to next session.
- Before next session, I need to save a preprocessed frame as a PNG with `cv2.imwrite` and check the ball is visible. It's the one thing the tests can't check.
- The preprocessor outputs HWC, (84, 84, 4). The network needs NCHW, and I'll do that transpose in the agent, not here.

## Anything surprising or worth flagging

- I'd never used OpenCV before, so I spent the start of the session learning its grayscale and resize functions before writing `src/preprocess.py`.

---

# Saturday: Replay Buffer

**Date:** 2026-07-04

**Floor:** `src/replay_buffer.py` hand built and verified in isolation. Push transitions, sample batches, assert shapes, and confirm circular overwrite when the buffer fills. Committed.

**Aspiration:** All of that, plus the Q network and target network written and verified with a GPU forward pass. A dummy batch through CUDA, output shape confirmed at (B, n_actions), `loss.backward()` runs without error, and target weights decoupled and syncable. The HWC to NCHW transpose and /255 normalization also land here.

---

## What landed today

- I wrote and committed `src/replay_buffer.py`.
- `tests/test_replay_buffer.py` passes. It covers shapes and dtypes, circular overwrite, state and next_state pairing, and the error when `batch_size` is bigger than the number of stored transitions.

## What's open (carrying forward)

- I met the floor but not the aspiration. The Q network and target network move to next session.

## Anything surprising or worth flagging

- I assumed the buffer would hold 1 million transitions like the original DQN paper. The memory math brought that down. Each state is 84 x 84 x 4 = 28,224 bytes as uint8, and storing state and next_state doubles it to 56,448 bytes per transition. With 16 GB of RAM and roughly 5 GB going to the OS, PyTorch, and the CUDA context, I have about 11 GB, which is around 195,000 transitions. I went with 100,000 to be safe.

---

# Saturday: Q Network

**Date:** 2026-07-11

**Floor:** The day counts as a win if the Q network and target network are both hand built and verified in isolation, then committed. That means a dummy batch of shape (B, 4, 84, 84) on CUDA produces output of shape (B, n_actions), lands on the GPU, runs `loss.backward()` cleanly, and leaves the target network weights unchanged after an optimizer step on the Q network until they are synced. The transpose and division by 255 also live in one consistent place on every path into the network.

**Aspiration:** If time allows, the stretch goal is to build and verify an epsilon greedy action selector. With epsilon = 1, it should behave like uniform random selection, and with epsilon = 0, it should behave like deterministic argmax selection. That would exercise the full action path end to end: stacked frame, transform, network, and action.

---

## What landed today

- I designed, built, and tested the Q network. The tests check its output shape, that it runs on the GPU, that gradients flow, and that its output looks sane. Where the transform happens, where the activations go, and how the action count gets passed in were all choices I made on purpose.

## What's open (carrying forward)

- The target network and the action selector are still open. Both are small. The target network is basically a second QNetwork plus a function that copies the weights over.

## Anything surprising or worth flagging

- Verifying the Q network end to end took a lot less code than I expected, once the architecture and tests were in place.
- I didn't think about the target network until the end of the session, so it moves to next session.

---

# Saturday: Target Network

**Date:** 2026-07-25

**Floor:** The day counts as a win if the target network is hand built and verified: snapshot its weights, take an optimizer step on the Q network, assert the target is unchanged, sync with `load_state_dict`, assert they now match, and confirm target params are never held by the optimizer. The action selector should also be built and verified: epsilon = 1 behaves like uniform random selection, epsilon = 0 behaves like deterministic argmax, and the forward pass runs under `no_grad`.

**Aspiration:** If time allows, the stretch goal is to close the seam between preprocessing and the network: one transform function, shared by the training path and the action selection path, that reconciles the preprocessor's normalized, channel first output with the replay buffer's raw uint8 storage and the network's expected input. Then run one integration smoke test that exercises the full pipeline end to end. `env.reset()` through the preprocessor, through that shared transform, through the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the same transform and network again. No shape, dtype, or device errors anywhere in that chain would count as the stretch goal met.

---

## What landed today

- I designed, built, and tested the target network. The tests confirm the optimizer never touches it, its weights stay frozen through a gradient step on the online network, and it matches the online network after a sync. I sync with a hard copy through `load_state_dict` instead of Polyak averaging. That's what the original Atari DQN paper did, and it's easier to test, since a sync is one event instead of a small drift every step.

## What's open (carrying forward)

- The action selector is still open. It gets its own file, `src/action_selector.py`, and its own test. Q values and epsilon in, an integer action out. Epsilon = 1 should behave like uniform random selection, epsilon = 0 like deterministic argmax, and the forward pass through the network should run under `no_grad`.
- The shared transform function is still open: one function, called by both the training path and the action selection path, that reconciles the preprocessor's normalized `(4, 84, 84)` float32 output with the replay buffer's raw `(84, 84, 4)` uint8 storage and converts it into what the network expects. This doesn't exist on either path yet, it isn't a matter of deduping a copy that's already there.
- The integration smoke test is still open: `env.reset()` through the preprocessor, the shared transform, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the same transform and network again.

## Anything surprising or worth flagging

- I hadn't looked at this code in two weeks, so getting back into `q_network.py` took longer than I expected, and the target network and its tests ran over because of it.

---

# Saturday: Action Selector

**Date:** 2026-08-08

**Floor:** The action selector hand built and verified. `src/action_selector.py` written, kept separate from the network so it only takes in Q values and epsilon, and `tests/test_action_selector.py` covering determinism at epsilon 0, uniform exploration at epsilon 1, correct return type, and valid action range. Committed.

**Aspiration:** If time allows, the stretch goal is the transform layer. Two path specific functions, one for the training path and one for the action selection path, sharing a contract rather than code, that reconcile the preprocessor's normalized 4 x 84 x 84 float32 output with the replay buffer's raw 84 x 84 x 4 uint8 storage and convert it into what the network expects. Neither exists on either path yet, so it is not a matter of combining code that already exists in two places, it is new code. If that lands, the next stretch goal is one integration smoke test that runs the whole pipeline end to end. `env.reset()` through the preprocessor, the transform, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the transform layer and the network again, with no shape, dtype, or device errors anywhere in that chain.

---

## What landed today

- I built `src/action_selector.py`. It takes Q values, epsilon, and the number of actions, and returns a plain Python int, either the argmax action or a random one. I kept it separate from the network on purpose, so whatever calls the network still owns the `no_grad` context, not the selector.
- `tests/test_action_selector.py` passes everything I wanted covered. Epsilon 0 always picks the same argmax action, epsilon 1 explores every action roughly evenly, the return type is a real Python int on both branches, and the action always falls inside the valid range. I caught a real bug while writing these tests, the random branch was returning a numpy int instead of a plain Python int.
- While tracing this, I noticed `architecture.md` had the preprocessor's output and the network's input written down wrong. It described them as 84 x 84 x 4 in uint8, which is actually the replay buffer's storage shape, not what the preprocessor or network use. The real shape is 4 x 84 x 84 in float32. I fixed `architecture.md`.

## What's open (carrying forward)

- The transform layer is still open. Two path specific functions, one for the training path and one for the action selection path, sharing a contract rather than code, that reconcile the preprocessor's normalized 4 x 84 x 84 float32 output with the replay buffer's raw 84 x 84 x 4 uint8 storage and convert it into what the network expects. Neither exists on either path yet, so it isn't a matter of combining code that already exists in two places.
- The integration smoke test is still open. `env.reset()` through the preprocessor, the transform layer, and the network, out to an action, back through `env.step()`, into the replay buffer, then a sampled batch through the transform layer and the network again.

## Anything surprising or worth flagging

- Looking closely at `push` in `replay_buffer.py`, I found there's no conversion logic there at all, just a plain assignment. Right now that fails loudly because the shapes don't match. But if I only fixed the shape and left the scaling alone, it would fail silently instead, since fixing just the shape wouldn't map [0, 1] floats into [0, 255] uint8 storage. Both issues come from the same place. I assumed a conversion existed between the preprocessor and the buffer, and it turns out I never actually wrote one.

---

# Saturday: Transform Layer

**Date:** 2026-08-15

**Floor:** `src/preprocess.py` updated: preprocessing stops after resize, returns uint8, frame stack shifts from axis 0 to the last axis, output shape (84, 84, 4) matching the buffer's storage exactly. `push` needs zero conversion logic. `test_preprocess_dtype`, `test_preprocess_value_range`, and the frame stack shape tests updated to match: uint8, 0 to 255, (84, 84, 4). `architecture.md` corrected in the same sitting to say uint8 HWC, not float32 CHW. The shared transform function built: uint8 HWC in, cast to float32, divide by 255, transpose to NCHW, once, right before the network. A test pushes a known frame through preprocess and the transform and asserts the actual returned values, not just shape and dtype, confirming the numbers coming out are the numbers that should come out. Committed.

**Aspiration:** The full pipeline runs end to end as a smoke test, two separate paths, both with no shape, dtype, or device errors. Acting: `env.reset()` through the preprocessor, the transform, the network, `select_action`, and `env.step()`, on the live batch of one state. Learning: a batch sampled from the buffer, through the transform and the network, no action, no `env.step()`.

---

## What landed today

- I didn't commit any code today. The whole session went to checking and design, and it was still worth it.
- I picked the buffer's storage dtype by measuring, not by habit. Two state sized arrays at capacity 100,000 cost 5.6448 GB under uint8 versus 22.5792 GB under float32, checked against 31.86 GB of system RAM. Float32 technically fits, but it buys nothing, since the emulator only ever produces uint8 pixels in the first place.
- I traced the transform's round trip and found it wasn't actually invertible. `astype(np.uint8)` truncates instead of rounding, so a value that isn't an exact multiple of 1 over 255 comes back slightly wrong after the round trip, silently, with no error thrown.
- Rounding before the cast would have shrunk that error, not removed it. The real fix was noticing that two conversions in the pipeline exist only to undo each other, the preprocessor dividing by 255 and the transform multiplying it back. Deleting both gets rid of the bug instead of just making it smaller.
- I wrote the new design down in enough detail to build straight from it: the preprocessor stops after resize and returns uint8 with no normalization, the buffer takes it raw with no conversion in `push`, and one shared transform sits on the read side, used identically by the training path and the action selection path.

## What's open (carrying forward)

- I didn't meet the floor. Nothing was written today, only checked and designed.
- The preprocessor change itself: stop normalizing, return uint8, and stack the frames on the last axis instead of the first.
- The three preprocessor tests, updated to match the new dtype and shape.
- `architecture.md`, corrected again to reflect the new design.
- The shared transform function, still unwritten.
- The round trip test, still unwritten.
- The end to end smoke test is still the aspiration.

## Anything surprising or worth flagging

- I caught the `push` problem last week by reading the code instead of assuming a conversion existed, and everything today came out of that.
- I spent three and three quarter hours on checking and design, and none on code. That's not a failure by itself, but I want to be honest about why. Did the checks run long because they needed to, or because one more check felt safer than writing something down?

---

# Saturday: Preprocess Pipeline

**Date:** 2026-08-22

**Floor:** `src/preprocess.py`: `preprocess()` stops normalizing and returns uint8. `_get_stacked()` stacks on the last axis. Output `(84, 84, 4)` uint8. Six of seven `test_preprocess.py` tests updated for the new dtype/shape/axis (`test_preprocess_shape` is the one that doesn't move). The shared transform function: uint8 HWC in, cast to float32, divide by 255, transpose to NCHW. The isolated value test on the transform (known input, exact expected output). The round trip test: frame through `preprocess`, through `buffer.push`/`buffer.sample`, through the transform, checked against transforming the original frame directly. `architecture.md` corrected again to say uint8 HWC. All committed and pushed.

**Aspiration:** The full end to end smoke test, two separate paths, both with no shape, dtype, or device errors. Acting: `env.reset()` through the preprocessor, the transform, the network, `select_action`, and `env.step()`, on the live batch of one state. Learning: a batch sampled from the buffer, through the transform and the network, no action, no `env.step()`.

---

## What landed today

- `src/preprocess.py` no longer normalizes. `preprocess()` returns the resized uint8 frame as is, and `_get_stacked()` stacks on the last axis instead of the first, so `reset()`/`step()` return `(84, 84, 4)` uint8, matching the replay buffer's storage exactly.
- I updated six of the seven `test_preprocess.py` tests for the new dtype, shape, and axis. All seven pass against a real Pong frame.
- I built `src/transform.py`: one function, uint8 HWC in, single frame or batch, normalized float32 NCHW out on a given device. It adds the missing batch axis to a single frame itself, so neither caller has to know which shape it started with.
- I built `tests/test_transform.py`: an isolated value test against hand computed expected output, and a round trip test, a frame through `Preprocessor`, through `ReplayBuffer.push`/`sample`, through the transform, checked against transforming the original directly. Both pass.
- I corrected `architecture.md`. The diagram now shows `Transform` in both the collection and training paths, and Preprocessing's output and the network's input match the actual code again.

## What's open (carrying forward)

- The full end to end smoke test is still the aspiration: `env.reset()` through the preprocessor, into the buffer, a sampled batch through the transform and the network, out to an action through `select_action`, back through `env.step()`. No shape, dtype, or device errors anywhere in that chain.

## Anything surprising or worth flagging

- I updated the preprocess tests right after the code change, before running pytest against the unedited version. So I updated them from knowing what the new shape and dtype should be, not from real failure messages.

---

# Saturday: Smoke Test

**Date:** 2026-08-29

**Floor:** Make sure `scripts/visual_check.py` works correctly, and build the end to end smoke test as two separate paths. The acting path: `env.reset()` through the preprocessor, transform, the network, `select_action`, and `env.step()`, using the live batch of one state, asserting shape `(1, n_actions)`, dtype `float32`, and device at the network output, plus a valid action index. The learning path: a batch sampled from the buffer, through the transform and the network, no action selection, no `env.step()`, asserting shape `(batch_size, n_actions)`, dtype `float32`, and device at the network output. No NaNs anywhere in either path.

**Aspiration:** Start the full training loop.

---

## What landed today

- I corrected `scripts/visual_check.py`: no more `*255` overflow, `transform_output.png` now compares against the same frame as `preprocessed_frame.png` instead of a later, mutated one, episode boundaries pair `env.reset()` with `prep.reset()` instead of stepping across them, and the ball candidate printout is cropped to the field so it's not buried under the score strip and the bottom wall. It used to call `preprocess()` directly, so it couldn't see last week's stacking axis change. Now it runs through the buffer, sample, and the shared transform, the same path training will actually use. I reran it and it's still clean.
- `tests/test_preprocess.py` now has two real assertions instead of eyeball checks: `test_newest_frame_at_last_index` and `test_reset_clears_stale_stack`. I also confirmed the frame order a second way, by differencing frames and tracking the ball's raw coordinates, scaled by 84 over 160 to match the preprocessed frame.
- I built `tests/smoke_test.py`: two paths, acting (`env.reset()` through the preprocessor, the transform, the network, `select_action`, `env.step()`) and learning (a batch sampled from the buffer through the transform and the network), both asserting shape, dtype, device, and no NaN on the network output. Ten steps and a batch of eight ran clean.

## What's open (carrying forward)

- The training loop itself: loss, `backward()`, the optimizer step, target network syncing on a schedule, and epsilon decay over time. Nothing built yet, that's today's aspiration. First thing to write next session is the weight update test: snapshot the Q and target weights, one optimizer step on Q with a synthetic batch, assert Q changed and target didn't.
- Action is hardcoded to `0` in `visual_check.py`'s `buffer.push()` call. Needs to be the actual action taken once the training loop exists.
- `smoke_test.py`'s learning path only checks `current_batch` from `buffer.sample()`. The other four fields, action, reward, next state, done, get discarded with `_` and never asserted.
- `run_acting_loop` gets called once by each test function, so the smoke test builds a fresh env and network twice for one run instead of once.

## Anything surprising or worth flagging

- I want to remember the `*255` bug as a category, not just an incident. Multiplying already normalized uint8 data by 255 wraps around instead of erroring, and the result was still a recognizable, plausible looking Pong frame, just inverted like a photo negative. It got past my eyeball checks because it looked fine, and it never crashed, so nothing else caught it either.
- I added a comment in `visual_check.py` explaining why pushing a cross episode `next_state` on a terminal transition is safe: the Bellman target masks the bootstrap term by `(1 - done)`, so that `next_state` never actually gets read. I wrote it so a future refactor doesn't decouple `done` from that assumption without knowing why it mattered.
- The smoke test's wording in this file was wrong in three places, not just today's Floor but the Aspiration lines from two earlier sessions too. All of them described a sampled batch feeding directly into `select_action`, which would crash on `select_action`'s `.item()` call. It stayed uncaught until I had to write the actual code and be specific about which tensor goes where. I reworded all three into two separate paths instead of one chain: the acting path takes the live batch of one state through `select_action`, and the learning path takes the sampled batch through the transform and the network only, `select_action` never sees it.
- I set `EPSILON = 0.0` in the smoke test on purpose, not as a leftover default. A nonzero epsilon would route the action through `select_action`'s random branch, which never reads `q_values`, so the forward pass and the network's actual output would never get exercised. A test set up that way would pass without testing anything.
- I stopped at the aspiration on purpose today instead of drifting into it. I met both floor items, and once I'd confirmed that, I moved the training step to next session by choice, not because I ran out of time.

---

# Saturday: Training Step

**Date:** 2026-09-12

**Floor:** The two open pre flight items closed with actual output, not memory: four ball coordinates across the stack, and right paddle frames at both extremes. If either one fails, that's the day and the training step moves. The VRAM baseline written into this file, plus the one line on score digits as a known choice. The training step written and verified on a fixture where I know the answer by hand: loss finite, Q network weights move, target network weights don't.

**Aspiration:** That training step integrated into an end to end loop that runs a few hundred steps without falling over.

---

## What landed today

- I built `src/train.py`: `train_step` takes the raw batch from `ReplayBuffer.sample()`, calls `transform` itself, and returns the loss as a float.
- I built `tests/test_train.py`: Q params move and target params don't, the loss matches a hand computed value on a fixture where both networks return fixed Q values, and a terminal batch ignores `next_state`. All three pass.
- I reworked `scripts/visual_check.py`: paddle height derived per run from the raw frame, the stack asserted on every step and after every reset, the ball tracked through all 4 slots, and an episode boundary forced at step 10 so that branch finally runs. I reran it and it's clean.
- `scripts/check_environment.py` now reports the baselines: 6144 MiB VRAM total with 5081 free, 21.0 MiB for both networks, 31.86 GiB host RAM, 5.26 GiB for the buffer at capacity 100,000.
- `docs/architecture.md` now records two known choices: no crop before the resize, and the buffer storing every frame twice.

## What's open (carrying forward)

- The loop. Still the aspiration, didn't land. First thing is proving a fixed batch trained repeatedly drives the loss toward zero, since nothing shows `train_step` learns yet.
- `env.action_space` is unseeded, so the last run had both paddles fully visible and neither extreme got checked. Seeding it plus a driven UP and DOWN rollout would put both in every run.
- `FORCED_BOUNDARY_STEP` is scaffolding. Comes out once real episodes terminate inside the loop.
- Action is still hardcoded to `0` in `buffer.push()`. Carried from last session, resolves when the loop supplies a real action.

## Anything surprising or worth flagging

- I widened a failing expectation instead of fixing the derivation. My expected 3 or 4 started flagging legitimate frames at 6 and 7, and I widened the band to 1 through 7 rather than asking why. A band that wide can't fail, so it shows green while testing nothing.
- The motion check used to drop empty slots and difference what was left, so an occluded middle slot made slots 0 and 3 look one frame apart. Today's run had slot 3 empty and reported 2 of 3 pairs instead of hiding it.
- The buffer stack and the same stack through `transform` came back with identical ball coordinates. That's the slot ordering proof, and until today it was me looking at a PNG.

---

# Saturday: Reorient and Train

**Date:** 2026-10-03

**Floor:** Reorient with the codebase and complete the masking test.

**Aspiration:** Begin training the agent.

---

## What landed today

- I walked the pipeline file by file to get back into the codebase.
- I added `test_terminal_target_equals_reward` to `tests/test_train.py`. It uses constant networks so the current Q value is `[-1.0, 5.0]` and the best next state Q value is `10.0`. Every transition is marked done, so the target is the reward alone and the hand computed loss is 4.0.
- I verified the new test by breaking the `targets` line two ways. One multiplied the reward by the future value and gave 17.5. The other zeroed the whole target on terminal transitions, reward included, and gave 2.5. I computed both losses by hand before running, and both failures matched.
- I made the comments and docstrings consistent across all the test files.
- All 27 tests pass.

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

- I added `test_train_step_overfits_fixed_batch` to `tests/test_train.py`. It calls `train_step` 200 times on one seeded batch of four transitions and asserts the final loss falls below 1% of the initial loss. It passes.
- I answered the five training decisions for the real run:
  - **When to train:** every 4 env steps.
  - **When to start:** after 50,000 transitions are in the buffer.
  - **Episode boundary:** reset the env and the frame stack, and keep the buffer, the networks, and the optimizer.
  - **Terminated vs. truncated:** only `terminated` stops bootstrapping. A truncated game was cut off by a time limit and could have kept going, so its next state still has value.
  - **Target sync:** every 10,000 gradient updates.
- I built `train()` in `src/train.py` and `scripts/tiny_train.py`, which calls the same function with tiny settings. One loop, two configs, so the tiny run actually says something about the real one. Actions go through `select_action` instead of inline epsilon greedy logic, so that behavior lives in one tested place.
- I ran the tiny loop on the MacBook Air. It finished, started training after warmup, synced the target on schedule, reset cleanly after an episode ended, and kept the loss finite throughout.

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

1. **Prove the overfit test can catch a broken training step:** take out `optimizer.step()` on purpose, run the test, and confirm it fails for the reason I expect. Then put the update back and confirm the test passes again.
2. **Finalize the exploration schedule and training length for the real run:** decide the starting epsilon, the minimum epsilon, the decay schedule, and the total number of environment steps the agent trains for.
3. **Finalize the action space for the real run:** decide whether the agent uses 3 actions or all 6. Then confirm the environment, the network output size, and the action selection code all match that decision before launch.
4. **Create the complete real run configuration and launch checklist:** put the chosen hyperparameters and run settings in their own config script. Then write down the checks to do before starting the PC run: device, environment, model, replay buffer, output paths, and temperature during warmup.

**Aspiration:** 

1. **Add the minimum logging and checkpointing the first real run needs:** write episode reward, episode number and the other useful training numbers to disk. Save model checkpoints on a fixed schedule, so the run leaves recoverable data and usable milestone models from the very start.
2. **Bring `scripts/visual_check.py` in line with the real training code:** remove the hardcoded action `0`, remove `FORCED_BOUNDARY_STEP`, and seed the action space, so the visual check uses the same action selection and episode boundaries as the training loop. Run the script afterward and confirm it still shows the gameplay I expect.
3. **Launch Run 1 on the PC:** check that the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and starts updating the network without errors. If I get there with logging and checkpointing fully working, let it keep going as the actual first real training run.

---

## What landed today

- I proved the overfit test can fail. The baseline passed at a ratio of 8.446e-05. With `optimizer.step()` commented out, the loss never moved, 2.880e-01 to 2.880e-01, ratio 1.0. With `optimizer.zero_grad()` commented out, the loss went up instead, ratio 3.2. The test passed again after each restore, at 1.744e-07 and 5.432e-04, and `git diff src/train.py` comes back empty.
- I'm keeping the 1% bar on the overfit test. Six healthy runs ranged from 1.744e-07 to 5.432e-04, and the two breaks landed at 1.0 and 3.2. So the bar sits in a gap of almost four orders of magnitude, 18x above the worst healthy run and 100x below the nearest failure. Nothing I can produce lands in that gap.
- I set the exploration schedule and run length:
  - **Epsilon start:** 1.0.
  - **Epsilon end:** 0.01, so the back half of training has very little random action noise, and improvements after 250k mostly come from learning.
  - **Decay steps:** 250,000, so exploration drops off early enough that most of Run 1 tests the learned policy.
  - **Total steps:** 2,000,000, which leaves 1.75M steps after the decay to show whether the policy improves and settles.
- Decay counts from env step 0, not from the end of warmup. I checked this in the code. Epsilon is 0.80 at step 50,000, where training starts, and reaches 0.01 at step 250,000.
- Sticky actions are off for Run 1. `ALE/Pong-v5` defaults to `repeat_action_probability=0.25`, but the DQN papers had none, and the +18 aspiration is tied to that setup. It's passed in as a config value, not hardcoded.
- Action space: all 6. The DQN papers used each game's minimal legal action set, which for Pong is six, and the +18 aspiration depends on matching that. The six collapse into three behaviors in pairs, so I'm keeping the full set to stay comparable, not for extra control.
- The demo's three bars collapse each pair with `max`. The policy is argmax over all six, so the tallest bar is always the action the agent actually picked. Mean or sum could put a bar on top while the agent does something else.
- I rewrote the pre registered prediction so it can fail. A plateau is the trailing mean over the last 100 episodes moving less than 1.0 across 200 episodes. It should cross 0 before step 1.2M, and still being below 0 at 1.2M means the schedule is insufficient.
- The replay buffer stays at 100,000, which is 5.26 GiB of the 18.69 GiB free. 200k would take 10.5 GiB and could start paging.
- I decided the last three hyperparameters. `batch_size` 32 and `gamma` 0.99 both match Nature, and neither depends on the optimizer. `learning_rate` 1e-4 is a deliberate change, since Nature used RMSProp at 0.00025 and learning rates don't carry over to Adam.
- I added `repeat_action_probability` and `seed` as parameters to `train()`. The sticky value gets read back from the emulator at startup instead of echoing the argument, so what prints is what the env is actually running.
- I implemented seeding. `np.random.seed` covers the exploration draw in `select_action` and the batch draw in `ReplayBuffer.sample`. `torch.manual_seed` covers network init, and the env only takes the seed on the first reset.
- I checked the seeding two ways. Two CPU runs at seed 0 came back byte identical, and a run at seed 1 differed in loss, episode length and reward. The matching pair only proved that something fixes the RNG. It took the seed 1 run to prove it's actually the seed parameter.
- I added a startup block that prints the device, seed, sticky actions, action meanings and network output size, plus an assert on the output size from a real forward pass. Floor 3's checks now happen on every run instead of by hand. I also checked the extra forward pass against an earlier CPU run to make sure it doesn't change anything.
- I wrote `scripts/real_train.py`. Its keys match `train()`'s signature and `tiny_train.py` exactly, so the two configs can't drift apart. The derived numbers check out: 487,501 gradient updates, 48 target syncs, epsilon 0.802 at the first update, and 5.26 GiB for the buffer.
- I changed `epsilon_end` in `scripts/tiny_train.py` from 0.1 to 0.01. It was the only value that differed from the real config without being a time or memory number, which made that file's header comment wrong.
- I wrote `docs/launch_checklist.md`. It has three sections, one per stage of the run, and only the checks the code can't do on its own.
- I'm accepting GPU nondeterminism for Run 1. The case study's claim is that DQN learns Pong, not that one exact run replays. So bit for bit reproducibility isn't worth an unknown slowdown on a run that has to finish before Oct 12.
- I tested the `nvidia-smi` logger command before putting it in the checklist. All five query fields are valid, and `-f` writes clean ASCII. At idle the card sits at 40 C and 315 MHz, with 1192 MiB of the 6144 already in use, so the VRAM I actually have at launch is closer to 4.9 GiB than 6.

## What's open (carrying forward)

- The whole aspiration moves to tomorrow: logging, checkpointing, `scripts/visual_check.py`, and the Run 1 launch. That's still inside the Oct 6 to 8 block.
- Three checklist items can't be ticked until logging and checkpointing exist. The decision to put the output path outside OneDrive has nowhere to live until then either.
- There's no evaluation path. The bar is defined over 100 evaluation episodes, but `train()` has no evaluation epsilon and no evaluation loop, so the 1.2M prediction uses training episodes as a stand in.
- I haven't checked which way `RIGHT` moves the paddle. The names are joystick names, so labelling the demo's bars from them could flip Up and Down.
- The determinism throughput comparison, for the Run 2 decision.
- Find one implementation that actually uses Adam at 1e-4, so the writeup can name it instead of just saying it's common.
- Still in `scripts/visual_check.py` from Sept 12: action hardcoded to `0`, `FORCED_BOUNDARY_STEP`, and the unseeded action space.

## Anything surprising or worth flagging

- Sticky actions are on by default in `ALE/Pong-v5`, at 0.25. At the epsilon I'd just picked, the environment would have overridden the policy about 25 times more often than my own exploration does. That would have made my whole argument for 0.01 over 0.1 pointless. I worked out the schedule before checking the environment defaults.
- The six actions aren't six behaviors. My first written reason for keeping all six was that the full action space lets the agent learn a complete control policy, and that isn't true for Pong. I kept the decision but dropped that reason.
- Seeding doesn't make a GPU run reproducible, and the drift shows up in behavior, not just in the numbers. Two GPU runs at seed 0 ended episode 1 at steps 900 and 928. A slightly different weight flips an argmax, which changes the action, which changes the data.
- Two identical runs weren't enough to prove the seeding worked. They showed the run was deterministic, not that the seed was what made it that way. Only changing the seed and watching the output change proved the parameter actually gets used.

---

# Wednesday: Run 1 Launch

**Date:** 2026-10-07

**Floor:**

1. **Add the minimum logging and checkpointing the first real run needs:** write episode reward, episode number and the other useful training numbers to disk. Save model checkpoints on a fixed schedule, so the run leaves recoverable data and usable milestone models from the very start.
2. **Bring `scripts/visual_check.py` in line with the real training code:** remove the hardcoded action `0`, remove `FORCED_BOUNDARY_STEP`, and seed the action space, so the visual check uses the same action selection and episode boundaries as the training loop. Run the script afterward and confirm it still shows the gameplay I expect.
3. **Launch Run 1 on the PC:** check that the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and starts updating the network without errors. If I get there with logging and checkpointing fully working, let it keep going as the actual first real training run.

**Aspiration:**

1. **Use real numbers in the launch checklist.** Fill in `temperature below ___ °C` before launch. Record the warmup steps per second, the projected time for 2,000,000 steps, and the launch commit hash. If Run 1 would run past the diagnosis block, cut the step count or change the run now.
2. **Build the relaunch instrumentation.** Log Q value drift, the actual epsilon, replay dynamics, and episode reward. Check it on a tiny run while Run 1 collects the baseline curve. The instrumentation will need a restart.
3. **Close the evaluation gap.** The bar is average reward above 0 over 100 evaluation episodes, but `train()` has no evaluation loop or evaluation epsilon. Add checkpoint evaluation with 100 episodes at a fixed low epsilon, so the 1.2M prediction uses the real criterion.

---

## What landed today

Floor 1 and Floor 2 are done. Floor 3 moves to tomorrow.

- `episodes.csv` writes one row per finished episode, and each row goes to disk as soon as it's written: `episode`, `env_step`, `reward`, `episode_steps`, `epsilon`, `gradient_updates`, `buffer_size`, `mean_loss`, `wall_clock_s`. `buffer_size` wasn't in the plan. I added it so Floor 3's "fills the replay buffer" can actually be seen, not guessed. `mean_loss` is blank, not 0.0, when no update ran.
- I built `src/checkpoint.py`. A checkpoint holds the Q weights, the target weights, the Adam state, the numpy, torch and CUDA RNG states, and the `env_step`, `episode` and `gradient_updates` counters. Without those counters, a resume would restart the epsilon schedule and the target sync. The buffer is left out, at 5.26 GiB a save. Saves go to a temp name and then get renamed, so dying mid write can't leave a half written newest checkpoint. Step 0 plus every 50,000 steps, so 40 files at 27 MB.
- I built `src/run_paths.py`. There's one `RUNS_ROOT`, at `~/pong-runs`, outside the OneDrive sync root, and both launch scripts import it, so they can't drift. Tiny runs go under `~/pong-runs/tiny`, so they never take a real run number.
- Each run's `run_config.json` holds every config value, plus the device, the GPU name, the sticky actions readback, the action count, `git_dirty` and the full commit hash. `train()` refuses to start if `episodes.csv` already exists. It also refuses on a dirty tree unless `allow_dirty` is passed, since a hash recorded with modified files doesn't describe what actually ran. Untracked files count too, because `src/checkpoint.py` and `src/run_paths.py` were untracked this morning and `train()` imports both.
- `real_train.py` now takes `--total-steps`, `--output-dir` and `--allow-dirty`. That makes the warmup pass just Run 1 with one number changed, not a separate script. Since epsilon decays from step 0, a 70,000 step warmup is literally Run 1's first 70,000 steps.
- `tiny_train.py` went from 1,000 to 1,500 steps, and it now asserts the log has at least one row. Episode 1 ended at 838, 838, 935, 838 and 838 across five runs, so 1,000 only left 65 steps of margin before a run would log nothing and still print `done:`.
- I built `scripts/play_checkpoint.py`. It loads a checkpoint, plays whole episodes, and reports the mean reward, an action histogram, and the counters from the file. It checks the load actually worked instead of assuming it did, and it only prints a verdict at 100 episodes or more.
- `scripts/visual_check.py`: I replaced the hardcoded action `0` with the sampled action and removed `FORCED_BOUNDARY_STEP`. One `SEED = 42` now feeds `np.random.seed`, `env.reset` and `env.action_space.seed`. `BUILD_STEPS` went from 60 to 1,200, based on a measurement instead of a guess: the first episode ends at 1100 at seed 42 and at 959 at seed 7. I renamed the paddle labels to `left (opponent)` and `right (agent)`, since "CPU" got confused with the training device.
- I reran it and it came back clean: transform round trip diff 0, one ball component, left paddle height 7 inside the expected set of 6 or 7, column deltas 2.0, 2.0, 2.0 with no sign changes, and one real episode boundary at step 1100.
- I checked these instead of assuming them: the overwrite refusal raises, a blank `mean_loss` shows up when nothing trained, the final checkpoint fallback produces 0, 400, 800, 900 at `total_steps` 900, a loaded checkpoint returns the counters the run printed, `taskkill /F` left 48 rows intact with `close()` never running and no `.tmp` left behind, and 100 episodes played at 4.05 percent random against the 0.05 target. I also proved both new assertions can fail. 28 of 28 tests pass.

## What's open (carrying forward)

- **All of Floor 3.** The warmup pass, the rate measurement, the projection, and the Run 1 launch.
- Pre launch items I haven't done: the temperature blank in `docs/launch_checklist.md`, pausing Windows Update, and setting sleep on AC to never instead of 1200 seconds. That last one matters, because Windows decides you're idle from user input, not GPU load. The commit is only local.
- No test covers logging or checkpointing. Today's evidence is manual runs, not something the test suite rechecks.
- I didn't start Aspiration 2. Aspiration 3 is partly done: `play_checkpoint.py` is the core of evaluation, but nothing wires an evaluation pass into `train()`, so the 1.2M prediction still uses training episodes as a stand in.
- `train()` runs a forward pass every step, and `select_action` throws it away on the random branch, so almost all of that work is wasted during warmup, when epsilon is near 1.0.
- The driven UP and DOWN rollout in `visual_check.py` is still open from Sept 12, and I still haven't checked which way `RIGHT` moves the paddle.
- Proven by how the code is built, not by a test: a kill landing in the middle of a checkpoint save, and the `True` side of the evaluation verdict, which won't get exercised until a checkpoint actually wins.

## Anything surprising or worth flagging

- Loading a checkpoint is where a real bug showed up, and saving one would never have found it. `map_location` moves every tensor onto the GPU, including the RNG state, and `torch.set_rng_state` rejects anything that isn't a CPU ByteTensor. So a resume would have crashed. The fix is calling `.cpu()` before restoring.
- Seeding the action space alone didn't make `visual_check.py` repeatable. Two seeded runs still disagreed, but only on the two sampled stack sections. That's because `ReplayBuffer.sample` draws with `np.random.choice`, and nothing was seeding `np.random`.
- Seeding that script was a trade off, not a pure win. Unseeded, every run looked at a different frame, so the paddles got checked at lots of heights by accident. Seeded, it's the same frame forever, so a rendering bug at the very bottom would now never show up, where before it might have been caught.
- The failure condition I wrote in advance for the playback test was wrong. It said -21 every episode in minimum steps would mean the network wasn't affecting the action. But the greedy policy picks action 2 on 2,199 of 2,292 steps, which pins the paddle somewhere it never touches the ball. With no contact, the ball's path doesn't depend on the paddle, so every episode is identical. A learned but useless constant policy produces exactly what I'd called failure. The action histogram is what actually tells them apart, so it's printed now.
- The 806 to 816 env steps per second from the kill tests isn't the number the projection needs. `gradient_updates` is 0 on every row, so that's only the fill phase. Projecting 2,000,000 steps from it would give 41 minutes, which would be way off.
- The run is bit for bit deterministic until the first gradient update. Two kill tests had identical episode boundaries through step 41,681, because with no updates the weights never change and both RNG sources are seeded.

---

# Thursday: Warmup and Launch

**Date:** 2026-10-08

**Floor:**

**Launch Run 1 on the PC:** check that the training loop reports `Training device: GPU`, fills the replay buffer, reaches the training phase, and starts updating the network without errors. Before launching, get the checklist numbers from a warmup pass at Run 1 settings: the load temperature, the steps per second after warmup, and the projected time for the full run. If the projected finish doesn't land before Oct 12, when the diagnosis block starts, cut the step count first and restate the pre registered crossing. If I get there with logging and checkpointing fully working, let it keep going as the actual first real training run.

**Aspiration:**

**Close the evaluation gap.** Run the checkpoint evaluation across a whole run, not one checkpoint at a time, and write a row per checkpoint. That way the evaluation curve can be read against the bar, instead of training reward standing in for it. Done means that file exists and Run 1's first checkpoints are in it. Evaluation shares the card with Run 1, so it only runs after I've taken Run 1's rate and temperature readings, and only on the first few checkpoints. Then if things slow down afterwards, I'll know it's the evaluation and not throttling.

---

## What landed today

- I ran the warmup pass at Run 1 settings, 120,000 steps into `run-00-warmup`. It had 131 episodes, 17,501 updates, one target sync at update 10,000, and checkpoints at 0, 50,000, 100,000 and 120,000. `run_config.json` recorded commit `f9ce3b66d9a8c41105676bba87ae8453d4e9cb82` with `git_dirty` false, matching `HEAD`. The 120,000 file is the first time the final checkpoint fallback has fired under the real config.
- `step_00050000.pt` came out at 27 MB, against step 0's 13.5 MB. So Adam state is in it, and the loop really does update before it saves.
- I measured the three checklist numbers. 252.1 steps per second after warmup, measured across 274 seconds from step 50,811 to 119,876, not from adjacent rows. That projects 2.20 hours for 2,000,000 steps, which finishes before Oct 12. So I'm not cutting the step count, and the pre registered 1.2M crossing stands as written. The peak load temperature was 52 °C, against the 88 limit.
- The fill phase ran at 809 steps per second, matching yesterday's 806 and 816. Training is 3.2 times slower, which is why the fill rate was never the number to project from.
- Episode boundaries through step 49,987 matched yesterday's kill tests exactly, then split after the first gradient update. That's the determinism behaving exactly as expected.
- I launched Run 1 at 13:55 into `run-01`, 2,000,000 steps, with commit `afb91fae0b1c1408009cb5385980f432c99f7ec7` recorded and `git_dirty` false. This time the loop based logger writes `gpu_log.csv` live, header included.
- I predicted the end of run reward before reading any rows: positive, around +10. My Oct 6 pre registration has the trailing mean crossing 0 before step 1.2M, which leaves about 800,000 steps of runway at epsilon 0.01. The buffer, at 100,000 against Nature's 1,000,000, is why it could fall short. And 8,000,000 frames against Nature's 50,000,000 is why I'm not predicting the +18 aspiration.
- Run 1 finished in 2.19 hours, against the 2.20 I projected. Both predictions were wrong. The trailing 100 mean never crossed 0, ending at -0.11, and at step 1.2M it was -10.17, so by the terms of the Oct 6 entry the schedule is insufficient. But the curve was still climbing at the last step and never plateaued by my own definition. Episode length more than quadrupled, with no shutouts in the last 100 episodes. So the learning is real, and the run just ran out of steps. Every figure is in [runs/run-01.md](runs/run-01.md).
- I wrote `docs/runs/run-01.md` as the reference for this run, with every figure pulled from `run_config.json`, `episodes.csv` and `gpu_log.csv`. Run 1's story runs into the Oct 12 diagnosis block, so it needs one place to live instead of being spread over three daily entries.
- I committed those three files for both runs under `docs/runs/`, 116 KB of text. Then I wrote `scripts/analyze_run.py` to print every figure from them. It's one command instead of the five chained one liners I started with, so a reader can check the numbers instead of trusting me. The 1.02 GB of checkpoints stays out.

## What's open (carrying forward)

- The aspiration. I didn't build the checkpoint evaluation sweep, so Run 1's result still rests on training episodes, not the 100 episode evaluation the bar is defined over.
- Restating the pre registered crossing. The Oct 6 version got answered and failed, so Run 2 needs a new prediction based on whatever step count it uses.
- Checking buffer RAM in Task Manager against the expected 5.26 GiB. Run 1 has exited, so this carries to the next run.
- Carried forward unchanged: the driven UP and DOWN rollout in `scripts/visual_check.py`, which way `RIGHT` moves the paddle, no test covering logging or checkpointing, and the stale note at `scripts/check_environment.py` line 79.

## Anything surprising or worth flagging

- The GPU is barely working. Across Run 1's 132 minutes it peaked at 42 percent utilization, 51.3 W on a card rated around 160, and 2,266 MiB of 6,144. At 253 steps per second with the card under half loaded, the bottleneck is the CPU side: one Pong instance, the frame preprocessing, and the Python loop running every step. That also explains how the fill phase hit 809 steps per second with no GPU work in it. The warmup understated all of this, peaking at 52 °C, 1,290 MHz and 30 W, because 6 minutes was never long enough to warm up fully.
- `nvidia-smi -f` buffers its output instead of writing each sample. `gpu_log.csv` sat at 0 bytes the whole run and only showed up when I stopped the process. So with that command, I can't do the five minute temperature check while the run is going.
- The checklist item "clock not dropping as temperature rises" would have flagged this perfectly healthy run. The clock swung from 1,290 down to 795 MHz while the temperature stayed in the forties and fifties. That's just the card idling down because it doesn't have enough work. A falling clock only means throttling when the temperature is near the limit.

---

# Friday: Instrumentation and Run 2

**Date:** 2026-10-09

**Floor:**

**Build the relaunch instrumentation and launch Run 2.** Alongside episode reward, log Q value drift on a fixed probe batch, the epsilon actually in use, and replay dynamics. Check it works on a tiny run. Then measure what it costs in throughput with a warmup pass at Run 2 settings, with nothing else on the card so the rate stays clean. The 6.6 hour estimate is just an assumption until that measurement is in. Before launching, write Run 2's pre registration: when the trailing mean should cross zero, where it should end, and what counts as a plateau. Then launch 6,000,000 steps with everything else identical to Run 1, using the first 2,000,000 as a check that it replicates Run 1's curve.

**Aspiration:**

**Close the evaluation gap, carried from yesterday.** Build the sweep around `scripts/play_checkpoint.py` and run it on Run 1 across 11 checkpoints: every 200,000 steps, 100 episodes each, about 50 minutes, one row per checkpoint. That spacing matches the trailing mean table in `docs/runs/run-01.md`, so the evaluation and training curves line up without interpolating. That's what replaces training reward standing in for the bar. Start it on the card while I write the instrumentation, so the two overlap instead of the sweep adding its whole runtime before Run 2. Let it finish before the warmup pass, so that rate stays clean. Not on the CPU, since ALE emulates there and the CPU is already the bottleneck, and not alongside Run 2. If the sweep isn't running within an hour of starting on it, drop it and move to the instrumentation, since the floor needs the rest of the day.

---

## What landed today

The aspiration is done. The floor isn't: I held Run 2 at the 5:30 cutoff on purpose, instead of rushing the launch.

- I built the evaluation sweep. `src/evaluate.py` scores one checkpoint, `scripts/sweep_eval.py` loops over a run and writes a row each, and `scripts/play_checkpoint.py` became a thin command line wrapper around the same `evaluate()`. One evaluation path with three ways in, so the sweep and the single checkpoint view can't drift apart.
- I ran it on Run 1: 11 checkpoints at 100 episodes each, in 50.3 minutes against the 50 I estimated. No checkpoint reaches the bar. The best is -1.63 at step 1,800,000, and the final checkpoint at 2,000,000 is -4.41.
- I scored 1,800,000 and 2,000,000 again at seed 1, on independent episodes. They came back -1.89 and -4.54. So the 2.7 point drop replicates at about 5 combined standard errors, and the means only moved 0.26 and 0.13 between seeds.
- I added `std_reward`, `seed` and `elapsed_s` to the sweep's rows. I also added a header check that refuses to append to a file whose columns don't match, and an `eval_config.json` sidecar that records the commit, the dirty flag and the settings for each run of the sweep. I pulled `src/git_info.py` out of `train.py`, since two places needed it.
- I built the instrumentation. `src/probe.py` makes 500 states from a fixed `PROBE_SEED` of 7. It's independent of the run seed, so two runs measure the same states. Its sha256 goes in `run_config.json`, and a `--probe` flag lets a run reuse an earlier run's file.
- `metrics.csv` writes 21 columns every 10,000 env steps, with a row at step 0 before any update. Fourteen are Q figures: the mean max Q, the mean spread, six per action means and six argmax counts. The rest are epsilon, the buffer size, the two replay fractions and the batch count behind them.
- I checked it on a tiny run, then measured the cost with a 120,000 step warmup at Run 2 settings. The post warmup rate was 256.1 steps per second, against 252.1 without instrumentation, so the cost is smaller than this measurement can pick up. 6,000,000 steps projects to 6.47 hours.
- The replay fraction landed right where the morning's math said it would. I measured 0.0244 of transitions, which is 0.78 per batch of 32, against 0.77 that I'd worked out from Run 1's first 100 episodes before any of this existed.
- `scripts/analyze_run.py`: I relabelled the update count as coming from the last logged episode, since it read like the run total and disagreed with the `done:` line. I also made it handle a missing `gpu_log.csv` instead of crashing.
- `docs/runs/run-01.md` got an Evaluation section with both seeds, the committed `eval.csv` and `eval_seed1.csv`, and the correction that the -0.11 I reported as Run 1's result never actually measured the policy the run finished with.

## What's open (carrying forward)

- **The floor.** Run 2 isn't launched. Everything it needs is done except the pre registration, which goes first thing tomorrow.
- Run 2's pre registration, three items: the crossing, the plateau definition, and whether any checkpoint clears the bar.
- The README doesn't mention `sweep_eval.py`.
- The probe's limitation, that a random policy visits early game states instead of where a trained agent ends up, is only in a code comment. It belongs in `run-02.md`.
- No test covers the sweep, `git_info.py`, the instrumentation, logging or checkpointing. That's five things I've only checked by hand.
- Checking buffer RAM in Task Manager against 5.26 GiB.
- Carried unchanged: the driven UP and DOWN rollout in `scripts/visual_check.py`, and which way `RIGHT` moves the paddle.

## Anything surprising or worth flagging

- The best checkpoint isn't the last one, and that holds at two seeds. Training reported -0.11 at the final step, while the final policy scores -4.41. That's because the trailing mean averages roughly the last 400,000 steps of a policy that keeps changing. While the policy is improving, that makes it look worse than it is. At the end, where it got worse, it makes it look better.
- The Q spread shrinks instead of growing. It's 0.057 at initialisation, then 0.006 by step 60,000, after 2,501 updates. Early training makes the network worse at telling actions apart than random weights were.
- The argmax moves as a block. All 500 probe states share one action, and that action jumps around between rows: NOOP, then FIRE, then RIGHTFIRE, then LEFT. A single concentration number would have read 1.00 on every row and hidden all of that, which is why I replaced it with the six counts.
- The `metrics.csv` rows from step 0 to 40,000 are byte identical. No update happens before 50,000, so that proves the probe stays fixed and isn't resampled.
- Episode boundaries match across both warmups through step 49,987 and split from episode 56. So the instrumentation doesn't use up any of the action RNG, and the split starts exactly where gradient updates start.
- The Oct 6 plateau definition doesn't mean one consistent thing. 200 episodes was 182,000 steps at the start of Run 1 and 828,000 at the end. So the same words are a four times looser test at the end than at the start.
- I predicted the instrumentation would cost under 1 percent, and gave a range of 249 to 252. It did cost under 1 percent, but the rate came back at 256.1, because I treated 252.1 as exact when it moves a couple of percent between runs.

---

# Monday: Run 2 Writeup

**Date:** 2026-10-12

**Floor:**

1. **Copy Run 2's logs into the repo:** the eight files from `~\pong-runs\run-02` into `docs/runs/run-02/`. Leave out `probe.npy` at 13.5 MB, since `run_config.json` records its sha256 and it regenerates from `PROBE_SEED`. Leave the 121 checkpoints at 3.03 GB uncommitted, like Run 1. Right now the run data only exists in one place, outside git and outside the OneDrive sync root.
2. **Write run-02.md's Config and Results:** copy Config from `run_config.json`, then score all three pre registered predictions against both evaluation sweeps. Write the limitations down alongside them: the training run is a single seed even though the evaluation repeats at two, and the probe samples a random policy's early game states instead of where a trained agent ends up.
3. **Find out what the six actions actually do:** drive each one in `scripts/visual_check.py` and record how the agent paddle's row moves, in the columns `PADDLE_COLUMNS` already finds. Then the Results section can describe the argmax counts as paddle direction instead of action numbers. That closes the driven rollout I've been carrying since Oct 3.

**Aspiration:**

1. **Settle whether the +18 aspiration is reachable.** PLANS.md sets the bar at above 0 and the aspiration at +18 or better, and both sweeps so far ran every 200,000 steps. The training trailing mean peaked between two of those points. So score the fifteen checkpoints between 5,000,000 and 6,000,000 that were never scored, at 100 episodes each. That's about 80 minutes at the measured 820 steps per second. Start it on the card before writing, so the two overlap instead of the sweep adding its runtime to the end of the day. If it isn't running within thirty minutes, drop it and spend the day on the floor, since run-02.md is what the Oct 19 case study block depends on.
2. **Pick the four clip checkpoints for the Oct 15 block.** Choose random, first rally, breakthrough and trained from `episodes.csv` and the evaluation files. The trained one should be the actual best policy, not whichever checkpoint the coarse grid happened to land on.

---

## What landed today

The floor has been met, all three items. The aspiration is half done: the +18 question is settled, but I haven't picked the four clip checkpoints.

- I copied Run 2's eight logs into `docs/runs/run-02/`, 296 KB, and checked every file by sha256 against its source. `probe.npy` stayed out at 13.5 MB, and so did the 121 checkpoints at 3.03 GB, same as Run 1.
- I wrote run-02.md, with seven sections that match run-01.md: header, Config, Results, Evaluation, Throughput and hardware, Limitations, and Reproducing the figures.
- I scored the three pre registered predictions. The crossing failed, at 2,715,875 against a 2,300,000 bound. The plateau definition fired falsely. 17 of 31 checkpoints cleared the bar at seed 0 and 18 at seed 1, with the best at +16.97 at step 5,400,000.
- I ran the seed 1 sweep: 31 checkpoints in 190.5 minutes, against 189.6 at seed 0. The seeds agree to 0.29 points on average, and not one of the 31 differs by more than two standard errors. So the dips in the curve are the policy, not the measurement.
- I settled the +18 question in 11 minutes instead of 80, by scoring just 5,450,000 and 5,500,000, the two around the training peak. They came back +16.61 and +16.38. So the 200,000 grid didn't skip over anything better, and I miss PLANS.md's aspiration by 1.03 points.
- I added Check 6 to `visual_check.py`. It drives each action and reads the paddle's row. RIGHT and RIGHTFIRE move it up, LEFT and LEFTFIRE move it down, and NOOP and FIRE don't move it at all. Each pair lands on the exact same row, and four asserts fail if that ever stops being true. That closes the driven rollout I've been carrying since Oct 3.
- I turned sticky actions off in `visual_check.py`, `check_environment.py`, `smoke_test.py` and `random_agent.py`. They were running at the v5 default of 0.25 while training ran at 0.0. The seeded episode now ends at step 946 instead of 1100.
- I gave `visual_check.py` a pass summary that exits with 1 on failure. Checks 3 and 4 report by printing instead of raising, so a bare "all passed" line would have printed right over a MISMATCH.
- I committed the Run 2 warmup's three logs, 9.6 KB, to match `run-00-warmup`. That brought back two figures I'd cut because I couldn't verify them: 256.1 steps per second and the 6.47 hour projection.
- I fixed run-01.md's prose. It gave 809 and 252 as Run 1's fill and post warmup rates, but both of those belong to `run-00-warmup`. Run 1's own are 837.3 and 248.9.
- Both run files told a reader to run `sweep_eval.py <run_dir>`, which nothing in the repo can do, since it reads `<run_dir>/checkpoints/`. Both now name the directory that actually holds them.

## What's open (carrying forward)

- **run-02.md has no Instrumentation section.** `metrics.csv` is committed at 87.5 KB, the probe is described in six places, and not one Q figure is reported. That was the whole point of Run 2.
- The four clip checkpoints for the Oct 15 block. 5,400,000 is the trained milestone, confirmed at both seeds.
- `architecture.md` names 8 of 19 modules and nothing from the training phase, including `train.py`.
- The README doesn't mention `sweep_eval.py`, which is the tool that now produces every headline number.
- No test covers the sweep, `git_info.py`, the instrumentation, logging or checkpointing. That's still five things checked only by hand.
- `check_environment.py` says the optimizer state and the activation peak can't be measured yet. Both are known now: about 12.9 MB from the checkpoint sizes, and 1,047 to 2,287 MiB from a run's GPU log.
- `analyze_run.py` doesn't print the fill phase rate or the step of the best trailing mean, and both run write ups need them.
- Carried unchanged: `run_config.json` doesn't record the torch version or the interpreter.

## Anything surprising or worth flagging

- A fixed seed doesn't reproduce a run on this hardware. Run 1 and Run 2 are byte identical for 54 episodes, first differ at episode 55, where the first gradient update lands, and are 11.32 points apart by step 2,000,000. That's why prediction 1 failed. It also means one training run can't be repeated exactly on this setup.
- For the third time, the best checkpoint isn't the last one. 5,400,000 beats 6,000,000 by 4.37 points, at 13.3 combined standard errors, and by 4.49 at 12.6 at seed 1.
- The gap between training and evaluation replicated: -4.20 and -4.62, against Run 1's -4.30, at reward levels 17 points apart. It's structural, so a headline number has to come from a sweep, not the training log.
- The plateau definition said the curve was stable across a window where evaluation swung 8.80 points. Since it's defined on the training curve, it can't see that, so Run 3's version has to be defined on evaluation.
- Episode length tells you how even the game is, not how good the agent is. The longest evaluation episodes are 5,299 steps at 2,600,000, where the agent scores -0.81, against 2,444 at the best checkpoint. run-01.md uses the rise from 909 to 4,139 as evidence of learning, but it's really evidence of reaching an even game.
- Four figures in my run-02.md draft were wrong, and checking every one against the committed files caught them before I committed. The ones I'd sampled from a 12 row table while the run was still going were the ones that broke. The ones computed over the whole file held up.
- `visual_check.py` had been checking a different game than the one that trains. On Oct 7, Floor 2 lined up its action selection and seeding with the training code, but missed sticky actions, which is the bigger difference.
- RIGHT moves the paddle up. The action names describe the joystick, not the screen, which is why this needed measuring instead of reading.

---

# Tuesday: Instrumentation and Clip Picks

**Date:** 2026-10-13

**Floor:**

1. **Add an Instrumentation section to run-02.md.** PLANS.md's finish line item 4 asks for Q value drift, the epsilon schedule, replay dynamics, wall clock time, VRAM, and GPU temperature. The last three are already in run-02.md's Throughput and hardware section. The first three are logged in `docs/runs/run-02/metrics.csv`, one row every 10,000 env steps. None of it is reported yet. Every figure comes from that file:
   - **Q value drift:** how `q_max_mean` and `q_spread_mean` change across the run. The first is the average highest Q value over 500 fixed probe states. The second is the average gap between the highest and lowest Q.
   - **Epsilon as logged:** the `epsilon` column, including the step where it first hits 0.01.
   - **Replay dynamics:** `batch_reward_fraction` and `batch_done_fraction`, and how both change as rallies get longer. The first is the share of sampled transitions where a point was scored. The second is the share that end an episode.
   - **Paddle direction:** the six `argmax_count` columns, grouped by what each action does on screen. On Oct 12, `visual_check.py` measured it. Actions 2 and 4 move the paddle up, 3 and 5 move it down, and 0 and 1 leave it still.
2. **Test whether Q spread moved before reward did.** The Oct 12 analysis suggested `q_spread_mean` started rising about 450,000 steps before the trailing 100 reward mean in `episodes.csv` did. That was never written down or checked. Before computing anything, write down one rule for "the step where a series starts rising". Apply it to both series, and decide what result would mean the claim is false. Report the result either way.
3. **Check the logger before trusting it.** Confirm three things. `metrics.csv` has 601 data rows, spaced exactly 10,000 steps apart. The six argmax counts sum to 500 on every row. `make_probe()` still regenerates the probe with the sha256 recorded in `run_config.json`. If any of these fail, items 1 and 2 are measuring a broken instrument.
4. **Pick the four clip checkpoints.** The Oct 15 block records a clip at each milestone named in PLANS.md: random, first rally, breakthrough, trained. Random is step 0. Trained is 5,400,000, the best evaluated checkpoint. For the other two, write down what "first rally" and "breakthrough" mean in terms of `episodes.csv` before looking. Then take the nearest saved checkpoint, since saves are every 50,000 steps. Record the four picks and the reasoning in run-02.md.

**Aspiration:**

1. **Get one test clip out of the 5,400,000 checkpoint.** No script in the repo records video yet. `play_checkpoint.py` only plays episodes and prints rewards. Add recording, save one episode as a video, and time it. Then the Oct 15 block can be planned around how long a clip actually takes.
2. **Make `analyze_run.py` print two figures it currently skips.** One is the fill phase rate, the steps per second before the first gradient update. The other is the env step where the trailing 100 mean peaked. Both run files compute these by hand.

---

## What landed today

The floor has been met. Item 2 is reported as an observation, since the rule was chosen after looking at the data. For item 4, I revised the clip rules and committed them before running them against the data, so the rules decided the checkpoints and I didn't pick them by hand. The aspiration is half done.

- I checked the logger before trusting it. `metrics.csv` has 601 rows exactly on the 10,000 step schedule, the argmax counts sum to 500 on every row, and the regenerated probe matches the sha256 in `run_config.json`.
- run-02.md now has a full Instrumentation section: the logger checks, Q value drift, the Q spread onset, paddle direction, epsilon against its schedule, and replay dynamics, including a cross check showing the batch counters agree with `episodes.csv` to within sampling noise.
- Q spread started rising about 520,000 steps before reward, at a 5 percent threshold. It leads at every threshold from 5 to 30 percent, but the "about 450,000" from Oct 12 didn't reproduce exactly.
- I wrote the four clip rules and committed them at 14:25:50. First rally uses episode length, since a rally is the ball coming back and length rises before score does.
- I rescored checkpoints 0 to 1,000,000 into `eval_lengths.csv`, adding `std_episode_steps`, twice. Both runs matched each other and the original `eval.csv` exactly, so evaluation is deterministic.
- The rules picked four checkpoints. Random is 0, the untrained network. First rally is 600,000, the first checkpoint whose episodes run clearly longer than the untrained network's, meaning the agent started returning the ball. First over the bar is 2,800,000, the first checkpoint that beats the built in opponent, with a mean evaluation reward above 0. It replaces "breakthrough" from the floor, which never said what it measured. Trained is 5,400,000, the checkpoint with the highest mean reward.
- Prediction 2's outcome is revised. The plateau rule fired as defined, but the run went on to +19.02, so it wasn't the ceiling I said it would be.
- `analyze_run.py` now prints every figure in run-01.md and run-02.md that comes from the runs' own files, plus `--compare` for figures across runs.
- Tests now cover `sweep_eval.py`, `git_info.py`, `probe.py` and `probe_metrics()`, `train()`'s writing of `episodes.csv` and `metrics.csv`, `checkpoint.py`, and `analyze_run.py`. All tests pass.
- `run_config.json` and each sweep's `*_config.json` now record the Python, torch, CUDA, and cuDNN versions, plus the gymnasium, ale-py, numpy, and OpenCV versions.
- `check_environment.py` now measures Adam state at 12.87 MiB and a training step at 40.3 MiB resident with an 18.1 MiB peak.
- run-01.md and run-02.md now give checkpoint sizes in GiB, and their sweep commands write to a new `--out`, since `eval.csv` and `eval_seed1.csv` predate the `std_episode_steps` column.

## What's open (carrying forward)

- Recording the four clips, in the Oct 15 block. No script records video yet, and each clip has to pass the play twice check from run-02.md's Clip selection before it counts. The test clip from today's aspiration carries forward with it.
- The README is out of date. It doesn't mention Run 2's result, the evaluation sweep (`sweep_eval.py`, built on `evaluate.py`), `analyze_run.py --compare`, or how to run the tests, and it only lists half the files a run writes.
- `architecture.md` names 8 of 19 modules, and nothing from training, evaluation, or instrumentation. The internal modules belong there, not in the README.
- The trained network picks "up" more than "down" on the probe states. From 3,000,000 on it averaged 213.9 up against 170.0 down, where an even split would be about 167 each. That might not be a bias in the policy. The probe states come from random play, so if the ball happens to sit above the agent's paddle in most of them, moving up is just the right answer. Checking it means finding the ball and paddle in each of the 500 probe frames, which `visual_check.py` can already locate, and seeing whether up wins mostly where the ball is above the paddle.

## Anything surprising or worth flagging

- Evaluation is deterministic, even though training isn't. The rescore matched the original `eval.csv` exactly, twice, so the same seed always plays the same 100 episodes. That's what makes the play twice check for clips possible.
- The Oct 12 plateau conclusion was wrong. The rule fired as defined, and what failed was reading a 200,000 step flat stretch as the ceiling. The trailing mean averages about 250,000 steps, so it can't see shorter swings, and Run 3's version needs a longer window, not a move to evaluation.
- Training barely uses the GPU's memory. A training step needs about 58 MiB at its peak, the 40.3 resident plus the 18.1 above it, while `gpu_log.csv` shows 1,047 to 2,287 MiB, and almost all of that is the CUDA context and other programs.

---

# Wednesday: Clip Recording

**Date:** 2026-10-14

**Floor:**

1. **Add recording to `evaluate()`.** An optional argument that keeps the first episode's raw frames, so a clip is played through the same code that scored the sweep.
2. **Write `scripts/record_clip.py`.** It plays the seed 0 first episode twice, refuses to save unless both plays give identical reward and length, then writes an mp4 at 15 fps, scaled up 3 times.
3. **Record all four clips:** 0, 600,000, 2,800,000, and 5,400,000. Note each one's reward, length, and recording time.

**Aspiration:**

1. **Update the README.** Add Run 2's result, the evaluation sweep (`sweep_eval.py`, built on `evaluate.py`), `analyze_run.py --compare`, how to run the tests, and the full list of files a run writes. Done means someone new could go from setup to reproducing Run 2's figures using only the README.

---

## What landed today

The floor has been met. All four clips are recorded, and every one passed the play twice check on the GPU. The aspiration wasn't done.

- I added recording to `evaluate()`. With `record_video=True`, it keeps every raw frame of episode 1, and recording doesn't change the episode it plays.
- `evaluate()` couldn't evaluate step 0 twice in one process. It left torch seeded, so a second call's fresh network matched step 0's weights and the load check stopped it. It now zeroes the network before loading, which is why the Random clip could play twice and agree. `tests/test_evaluate.py` covers both this and recording.
- I reran 5,400,000 for 100 episodes after the changes. It scored +16.97 again, exactly matching the sweep, so scoring is unchanged. It won all 100 games.
- `scripts/record_clip.py` plays the episode twice, saves only if both plays agree, and falls back to CPU if they don't. It writes a 30 second clip as H.264 through `imageio-ffmpeg`, since OpenCV on this machine can't encode H.264, plus a `_q.json` with the six Q values for every frame and a `.json` record.
- I added a clip window rule to run-02.md, the first 30 seconds of each episode, and committed it before cutting. After cutting, I revised it to the last 30 seconds, so every clip ends on its result with the final score on screen, and committed that before cutting again.
- The seed 0 episode at 600,000 never returns the ball, so its clip didn't show a rally. I added a first rally clip rule and committed it before recording. It tries seeds in order, takes the first episode with 3 returns within 30 seconds, and starts the clip 2 seconds before them. `record_clip.py --rally` applies it, and `tests/test_record_clip.py` tests the return counting.
- I recorded the final four clips. Each passed the play twice check on the GPU:
  - **Random, 0:** seed 0, reward -21 (lost 21 to 0), 824 steps, the last 30 seconds, recorded in 3.0 s.
  - **First rally, 600,000:** seed 6, reward -18 (lost 21 to 3), 1,517 steps, the 30 seconds around its first rally, recorded in 20.6 s including the seed search.
  - **First over the bar, 2,800,000:** seed 0, reward +6 (won 21 to 15), 4,796 steps, the last 30 seconds, recorded in 14.2 s.
  - **Trained, 5,400,000:** seed 0, reward +18 (won 21 to 3), 2,386 steps, the last 30 seconds, recorded in 7.3 s.
- Three clips were recorded at `f1ba9d8` and first rally at `4f56a27`, all with a clean working tree. The records include the `imageio-ffmpeg` version, which `git_info.py` now records. `analyze_run.py` prints each clip's seed, start frame, and returns, and run-02.md's Clip selection now covers all three rules and the final clips.
- Every function now has type hints, every test helper has a docstring, 24 comments that only restated the code are gone, and a comment in `train.py` that had been split in two is back in one piece. All 70 tests pass.

## What's open (carrying forward)

- The README is still out of date, since the aspiration wasn't done.
- The case study page has to play the clips muted, inline, and looping, with `<video autoplay muted loop playsinline>`, and draw its Q value bars from each clip's `_q.json`.
- `scripts/watch_checkpoint.py`, a live game window for a trained checkpoint like `random_agent.py`, wasn't started.
- `architecture.md` still names 8 of 19 modules.

## Anything surprising or worth flagging

- The step 0 bug had been there all along and never showed up, because the sweep and the rescore both scored step 0 first in a fresh process. Playing each clip twice was the first time anything evaluated step 0 twice.
- Episode length was a weaker stand in for rallies than I expected. The rule picked 600,000 because its episodes run longer, and returns really do start there, but they're patchy: across seeds 0 to 8, six episodes return the ball at least once and only a few return it several times. The seed 0 episode returns it zero times in 1,128 steps.
- The first rally clip is the first one where the agent scores, taking 3 points before losing 21 to 3.
- H.264 shrank the clips by far more than the cut alone explains. The 30 second clips are 45 to 63 KB each, against 0.9 to 6.3 MB for the full length `mp4v` versions.

---

# Thursday: README and Architecture

**Date:** 2026-10-15

**Floor:**

1. **Update the README.** Add Run 2's result, the evaluation sweep (`sweep_eval.py`, built on `evaluate.py`), `analyze_run.py --compare`, recording clips with `record_clip.py`, how to run the tests, and the full list of files a run writes. Done means someone new could go from setup to reproducing Run 2's figures using only the README.
2. **Bring `architecture.md` up to date.** Its module table names 8 of the 20 modules in `src/` and `scripts/`, and nothing from training, evaluation, instrumentation, or clip recording. Add the other 12, with what each one does and how they connect.

**Aspiration:**

1. **Write `scripts/watch_checkpoint.py`.** A live game window for a trained checkpoint, like `random_agent.py`, so the trained agent can be watched playing without recording a clip.

---

## What landed today

## What's open (carrying forward)

## Anything surprising or worth flagging

---
