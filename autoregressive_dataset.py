from typing import Tuple
import numpy as np

import torch
from torch.utils.data import Dataset

class AutoregressiveDataset(Dataset):
    def __init__(self, window_size:int, sequence:list[float]) -> None:
        super().__init__()
        assert len(sequence) > window_size
        self._sequence = np.array(sequence, dtype=np.float32)
        self._window_size = window_size

    def __len__(self):
        return len(self._sequence) - self._window_size - 1

    def __getitem__(self, index) -> Tuple[np.ndarray, float]:
        return self._sequence[index:index+self._window_size], self._sequence[index+self._window_size+1]