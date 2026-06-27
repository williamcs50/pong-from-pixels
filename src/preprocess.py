import cv2
import numpy as np
from collections import deque

class Preprocessor:
    def __init__(self, stack_size: int = 4, frame_size: tuple = (84, 84)):
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.frame_stack = deque(maxlen=stack_size)


    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Raw (210,160,3) RGB => (84,84) grayscale uint8"""

        if len(obs.shape) == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        else:
            gray = obs
        
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)

        return resized.astype(np.uint8)
    
    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Call on env.reset(). Fills the stack by repeating the first frame."""
        processed = self.preprocess(obs)

        self.frame_stack.clear()

        for _ in range(self.stack_size):
            self.frame_stack.append(processed)

        return self._get_stacked()


    def step(self, obs: np.ndarray) -> np.ndarray:
        """Call after every env.step()"""
        processed = self.preprocess(obs)

        self.frame_stack.append(processed)

        return self._get_stacked()


    def _get_stacked(self) -> np.ndarray:
        """Return current stack as (84, 84, 4) uint8."""
        return np.stack(list(self.frame_stack), axis=-1)

 