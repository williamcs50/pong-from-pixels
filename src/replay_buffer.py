import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.current_state = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.action_taken = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_state = np.zeros((capacity, 84, 84, 4), dtype=np.uint8)
        self.done = np.zeros(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self.size
    
    def push(self, current_state: np.ndarray, action_taken: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.current_state[self.position] = current_state
        self.action_taken[self.position] = action_taken
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.done[self.position] = done

        # Advance write pointer
        self.position = (self.position + 1) % self.capacity

        # Grow until capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if batch_size > self.size:
            raise ValueError(f"Error: There is not enough transitions to sample: You have requested {batch_size}, but have {self.size}")

        indices = np.random.choice(self.size, batch_size, replace=False)
        current_batch = self.current_state[indices]
        action_batch = self.action_taken[indices]
        reward_batch = self.reward[indices]
        next_batch = self.next_state[indices]
        done_batch = self.done[indices]

        return current_batch, action_batch, reward_batch, next_batch, done_batch