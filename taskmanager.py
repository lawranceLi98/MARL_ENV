from __future__ import annotations

import random
from typing import List

import numpy as np

from .task import UserTask


class TaskGenerator:
    """
    一个可复现的任务生成器，使用一个外部传入的NumPy随机数生成器(RNG)。
    """

    def __init__(self, prompts: List[str], parallel_choices: List[int], rng: np.random.Generator) -> None:
        """
        初始化 TaskGenerator。

        Args:
            prompts (List[str]): 可供选择的提示词列表。
            parallel_choices (List[int]): 可供选择的并行度列表。
            rng (np.random.Generator): 用于所有随机操作的 NumPy 随机数生成器实例。
        """
        self._prompts = prompts
        self._parallel_choices = parallel_choices
        self.rng = rng  # 存储传入的rng对象

    def reset(self, rng: np.random.Generator) -> None:
        """
        重置生成器的随机状态，通过接收一个新的rng对象实现。
        当外部环境重置并产生新的rng时，应调用此方法。
        """
        self.rng = rng

    def sample(self, arrival_time: float) -> UserTask:
        """使用 self.rng 从prompts和并行度选项中采样。"""
        # 使用 self.rng.choice 替代 random.choice
        prompt = self.rng.choice(self._prompts)
        parallel = self.rng.choice(self._parallel_choices)
        return UserTask(parallelism=parallel, prompt=prompt, arrival_time=arrival_time)

    def poisson_arrivals(self, lam_per_sec: float, dt_seconds: float, now: float) -> List[UserTask]:
        """使用 self.rng 生成符合泊松分布的到达任务。"""
        expected = lam_per_sec * dt_seconds

        # 使用 self.rng.poisson 替代 np.random.poisson
        num = self.rng.poisson(expected)

        arrivals: List[UserTask] = []
        if num <= 0:
            return arrivals

        # 使用 self.rng.uniform 替代 np.random.uniform
        offsets = self.rng.uniform(0.0, dt_seconds, size=num)
        
        for off in offsets:
            arrivals.append(self.sample(arrival_time=now + float(off)))
        return arrivals


class TaskManager:
    # In simulation: only tracks and provides user task objects
    def __init__(self, generator: TaskGenerator) -> None:
        self._generator = generator

    def initial_tasks(self) -> List[UserTask]:
        # With Poisson process, we start with empty queue
        return []


