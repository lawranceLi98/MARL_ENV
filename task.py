from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class UserTask:
    parallelism: int
    prompt: str
    arrival_time: float
    # Real time tracking for real backend
    real_arrival_time: Optional[float] = None

    def state(self) -> np.ndarray:
        # Shape (3,), only parallelism is well-defined per doc; the rest left as zeros
        # return np.array([float(self.parallelism)/8, self.arrival_time/300, 0.0], dtype=np.float32)
        return np.array([float(self.parallelism)/8, self.arrival_time/300], dtype=np.float32)


@dataclass
class SDTask:
    assigned_node_indices: List[int]
    steps: int
    reload_required: bool
    prompt: str
    arrival_time: float
    completion_time: Optional[float] = None
    # Heterogeneous computing: minimum compute power among assigned nodes
    min_compute_power: float = 1.0
    # Real time tracking for real backend
    real_arrival_time: Optional[float] = None
    real_completion_time: Optional[float] = None


@dataclass
class CloudTask:
    steps: int
    prompt: str
    arrival_time: float
    submit_time: float
    completion_time: float


class Queue:
    def __init__(self) -> None:
        self._items: List[UserTask] = []

    def push(self, task: UserTask) -> None:
        self._items.append(task)

    def pop(self, index: int) -> Optional[UserTask]:
        if index < 0 or index >= len(self._items):
            return None
        return self._items.pop(index)

    def peek(self, index: int) -> Optional[UserTask]:
        if index < 0 or index >= len(self._items):
            return None
        return self._items[index]

    def __len__(self) -> int:  # noqa: D401
        return len(self._items)

    def as_state(self, max_visible: int, task_state_dim: int = 2) -> np.ndarray:
        # Shape (l, task_state_dim)
        l = max_visible
        d = int(task_state_dim)
        state = np.zeros((l, d), dtype=np.float32)
        upto = min(l, len(self._items))
        for i in range(upto):
            raw = self._items[i].state()
            copy_len = min(d, int(raw.shape[0]))
            state[i, :copy_len] = raw[:copy_len]
        return state
