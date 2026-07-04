import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.replay_buffer import ReplayBuffer

FRAME_SHAPE = (84, 84, 4)
BATCH_SIZE = 16
SMALL_CAPACITY = 5
N_ACTIONS = 6
NEXT_STATE_OFFSET = 100
UINT8_MAX = 256
LARGE_CAPACITY = 100


def make_transition(i: int) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    """Return a distinguishable transition where every value encodes index i."""
    state = np.full(FRAME_SHAPE, i % UINT8_MAX, dtype=np.uint8)
    next_state = np.full(FRAME_SHAPE, (i + NEXT_STATE_OFFSET) % UINT8_MAX, dtype=np.uint8)
    action = i % N_ACTIONS
    reward = float(i)
    done = i % 2 == 0
    return state, action, reward, next_state, done


def test_push_and_sample_shapes_dtypes() -> None:
    buf = ReplayBuffer(capacity=LARGE_CAPACITY)
    for i in range(BATCH_SIZE * 2):
        buf.push(*make_transition(i))

    states, actions, rewards, next_states, dones = buf.sample(BATCH_SIZE)

    assert states.shape == (BATCH_SIZE, *FRAME_SHAPE),     f"states shape: {states.shape}"
    assert next_states.shape == (BATCH_SIZE, *FRAME_SHAPE), f"next_states shape: {next_states.shape}"
    assert actions.shape == (BATCH_SIZE,),               f"actions shape: {actions.shape}"
    assert rewards.shape == (BATCH_SIZE,),               f"rewards shape: {rewards.shape}"
    assert dones.shape == (BATCH_SIZE,),                 f"dones shape: {dones.shape}"

    assert states.dtype == np.uint8,             f"states dtype: {states.dtype}"
    assert next_states.dtype == np.uint8,        f"next_states dtype: {next_states.dtype}"
    assert actions.dtype == np.int64,            f"actions dtype: {actions.dtype}"
    assert rewards.dtype == np.float32,          f"rewards dtype: {rewards.dtype}"
    assert dones.dtype == np.bool_,              f"dones dtype: {dones.dtype}"

    print("PASS  push and sample shapes and dtypes")


def test_circular_overwrite() -> None:
    overflow = 2
    buf = ReplayBuffer(capacity=SMALL_CAPACITY)
    for i in range(SMALL_CAPACITY + overflow):
        buf.push(*make_transition(i))

    assert len(buf) == SMALL_CAPACITY, f"Expected size {SMALL_CAPACITY}, got {len(buf)}"
    assert buf.position == overflow,   f"Expected position {overflow}, got {buf.position}"

    # The overflow transitions should have wrapped to positions 0 and 1
    assert buf.current_state[0][0][0][0] == SMALL_CAPACITY % UINT8_MAX,          "Position 0 should hold transition 5"
    assert buf.current_state[1][0][0][0] == (SMALL_CAPACITY + 1) % UINT8_MAX,    "Position 1 should hold transition 6"

    print("PASS  circular overwrite")


def test_state_next_state_pairing() -> None:
    buf = ReplayBuffer(capacity=LARGE_CAPACITY)
    for i in range(LARGE_CAPACITY // 2):
        buf.push(*make_transition(i))

    states, _, _, next_states, _ = buf.sample(BATCH_SIZE)

    for j in range(BATCH_SIZE):
        state_val = int(states[j][0][0][0])
        next_val = int(next_states[j][0][0][0])
        assert next_val == (state_val + NEXT_STATE_OFFSET) % UINT8_MAX, (
            f"Pairing broken: state={state_val}, next_state={next_val}"
        )

    print("PASS  state and next_state pairing preserved")


def test_sample_raises_when_insufficient() -> None:
    buf = ReplayBuffer(capacity=LARGE_CAPACITY)
    buf.push(*make_transition(0))
    try:
        buf.sample(BATCH_SIZE)
        assert False, "Expected ValueError"
    except ValueError:
        pass
    print("PASS  sample raises ValueError when batch_size > size")


if __name__ == "__main__":
    test_push_and_sample_shapes_dtypes()
    test_circular_overwrite()
    test_state_next_state_pairing()
    test_sample_raises_when_insufficient()
    print("\nAll tests passed.")
