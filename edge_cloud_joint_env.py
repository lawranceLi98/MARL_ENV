from __future__ import annotations

from collections import deque
import random
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional dependency
    gym = None
    spaces = None  # type: ignore

GymBase = gym.Env if gym is not None else object

from .config import EnvConfig, get_config
from .node import Node
from .task import CloudTask, Queue, SDTask
from .taskmanager import TaskGenerator, TaskManager
from .node_manager import NodeManager
from config import get_runtime_config


class EdgeCloudJointEnv(GymBase):
    """Joint edge-cloud scheduling env for MAPPO-style training.

    Action (joint, continuous):
      - shape: (2 * num_nodes,)
      - first half: local bid per node
      - second half: step preference per node

    Observation:
      - same structural form as EdgeEnv:
        nodes_state (num_nodes, node_state_dim), queue_state (l, task_state_dim)
      - or flattened when concat_state=True.
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: Optional[int] = None) -> None:
        self.config: EnvConfig = config or get_config()
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

        self.num_nodes = int(self.config.num_nodes)
        self.max_visible_tasks = int(self.config.max_visible_tasks)
        self.node_state_dim = int(self.config.node_state_dim)
        self.task_state_dim = int(self.config.task_state_dim)
        self.dt = float(self.config.dt_seconds)
        self.max_sim_seconds = float(self.config.max_sim_seconds)

        runtime_config = get_runtime_config()
        self.node_manager = NodeManager(
            config=self.config,
            runtime_config=runtime_config,
            mode=runtime_config.mode,
        )

        self.generator = TaskGenerator(self.config.prompt_pool, self.config.parallel_choices, rng=self.rng)
        self.task_manager = TaskManager(self.generator)

        self.now_seconds: float = 0.0
        self.queue: Queue = Queue()
        self._running_local_tasks: List[SDTask] = []
        self._running_cloud_tasks: List[CloudTask] = []
        self._cloud_available_at: float = 0.0

        self._num_generated_tasks: int = 0
        self._num_completed_tasks: int = 0
        self._completed_latency_sum: float = 0.0
        self._last_completed_latencies: List[float] = []
        self._last_quality: float = 0.0

        self._qos_violation_count: int = 0
        self._local_dispatch_count: int = 0
        self._cloud_dispatch_count: int = 0
        self._privacy_violation_count: int = 0
        self._latency_history: Deque[float] = deque(maxlen=int(self.config.risk_history_window))

        if spaces is not None:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.num_nodes,),
                dtype=np.float32,
            )

            node_low_vec = np.zeros((self.node_state_dim,), dtype=np.float32)
            node_high_vec = np.full((self.node_state_dim,), np.inf, dtype=np.float32)
            node_low = np.tile(node_low_vec, (self.num_nodes, 1))
            node_high = np.tile(node_high_vec, (self.num_nodes, 1))

            queue_low_vec = np.zeros((self.task_state_dim,), dtype=np.float32)
            queue_high_vec = np.full((self.task_state_dim,), np.inf, dtype=np.float32)
            queue_low = np.tile(queue_low_vec, (self.max_visible_tasks, 1))
            queue_high = np.tile(queue_high_vec, (self.max_visible_tasks, 1))

            if self.config.concat_state:
                obs_low = np.concatenate([node_low.flatten(), queue_low.flatten()]).astype(np.float32)
                obs_high = np.concatenate([node_high.flatten(), queue_high.flatten()]).astype(np.float32)
                self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            else:
                self.observation_space = spaces.Tuple(
                    (
                        spaces.Box(
                            low=node_low,
                            high=node_high,
                            shape=(self.num_nodes, self.node_state_dim),
                            dtype=np.float32,
                        ),
                        spaces.Box(
                            low=queue_low,
                            high=queue_high,
                            shape=(self.max_visible_tasks, self.task_state_dim),
                            dtype=np.float32,
                        ),
                    )
                )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            if gym is not None:
                super().reset(seed=seed)
            self.rng = np.random.default_rng(seed)
            random.seed(seed)

        self.now_seconds = 0.0
        self.generator.reset(rng=self.rng)
        self.node_manager.reset()

        self.queue = Queue()
        self._running_local_tasks = []
        self._running_cloud_tasks = []
        self._cloud_available_at = 0.0

        self._num_generated_tasks = 0
        self._num_completed_tasks = 0
        self._completed_latency_sum = 0.0
        self._last_completed_latencies = []
        self._last_quality = 0.0
        self._qos_violation_count = 0
        self._local_dispatch_count = 0
        self._cloud_dispatch_count = 0
        self._privacy_violation_count = 0
        self._latency_history.clear()

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 2 * self.num_nodes:
            action = np.pad(action, (0, 2 * self.num_nodes - action.shape[0]), mode="constant")
        elif action.shape[0] > 2 * self.num_nodes:
            action = action[: 2 * self.num_nodes]

        local_bids = action[: self.num_nodes]
        step_prefs = action[self.num_nodes :]

        reward = 0.0
        if len(self.queue) > 0:
            user_task = self.queue.peek(0)
            if user_task is not None:
                required = int(user_task.parallelism)
                idle = self.node_manager.get_idle_node_indices()
                ranked_idle = sorted(idle, key=lambda i: float(local_bids[i]), reverse=True)
                threshold = float(self.config.local_bid_threshold)
                local_candidates = [i for i in ranked_idle if float(local_bids[i]) > threshold]

                if required > 0 and len(local_candidates) >= required:
                    chosen = local_candidates[:required]
                    step_val = self._map_step(float(np.mean(step_prefs[chosen])))
                    local_task = self._build_local_task_from_queue_head(chosen, step_val)
                    if local_task is not None:
                        self._apply_local_schedule(local_task)
                        self._local_dispatch_count += 1
                        reward = self._compute_local_reward(local_task)
                else:
                    cloud_task = self._route_head_task_to_cloud(step_prefs=step_prefs, idle_indices=idle)
                    if cloud_task is not None:
                        self._cloud_dispatch_count += 1
                        reward = self._compute_cloud_reward(cloud_task)

        # Advance time and arrivals once per decision step.
        self._advance_time_and_arrivals(self.dt)

        terminated = self._all_tasks_completed()
        truncated = self.now_seconds >= self.max_sim_seconds
        obs = self._get_obs()
        info = self._build_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ===== Core helpers =====
    def _get_obs(self) -> Any:
        nodes_state = self.node_manager.get_states()
        queue_state = self.queue.as_state(self.max_visible_tasks, task_state_dim=self.task_state_dim).astype(np.float32)
        if self.config.concat_state:
            return np.concatenate([nodes_state.flatten(), queue_state.flatten()]).astype(np.float32)
        return nodes_state, queue_state

    def _map_step(self, pref: float) -> int:
        max_step = int(self.config.max_step)
        scaled = (np.clip(pref, -1.0, 1.0) + 1.0) * 0.5
        return int(np.round(1 + scaled * (max_step - 1)))

    def _build_local_task_from_queue_head(self, chosen: List[int], steps: int) -> Optional[SDTask]:
        user_task = self.queue.pop(0)
        if user_task is None:
            return None
        reload_required = not self.node_manager.is_exact_group(chosen)
        min_compute_power = min(self.config.compute_power[i] for i in chosen) if chosen else 1.0
        return SDTask(
            assigned_node_indices=chosen,
            steps=int(steps),
            reload_required=reload_required,
            prompt=user_task.prompt,
            arrival_time=user_task.arrival_time,
            min_compute_power=min_compute_power,
            real_arrival_time=user_task.real_arrival_time,
        )

    def _apply_local_schedule(self, local_task: SDTask) -> None:
        chosen = local_task.assigned_node_indices
        self.node_manager.assign_task(local_task)
        if local_task.reload_required:
            self.node_manager.unload_model_groups(chosen)
            new_group_id = self.node_manager.get_max_group_id() + 1
            self.node_manager.load_model_group(chosen, new_group_id)
        self.node_manager.renumber_groups()
        self._running_local_tasks.append(local_task)

    def _route_head_task_to_cloud(self, step_prefs: np.ndarray, idle_indices: List[int]) -> Optional[CloudTask]:
        if not bool(self.config.cloud_enabled):
            return None
        user_task = self.queue.pop(0)
        if user_task is None:
            return None

        if idle_indices:
            step_pref = float(np.mean(step_prefs[idle_indices]))
        else:
            step_pref = float(np.mean(step_prefs))
        steps = self._map_step(step_pref)

        service_seconds = (
            float(self.config.cloud_base_latency_seconds)
            + float(self.config.cloud_step_time_coeff) * float(steps)
        )
        start_time = max(self.now_seconds, self._cloud_available_at)
        completion_time = start_time + service_seconds
        self._cloud_available_at = completion_time

        cloud_task = CloudTask(
            steps=int(steps),
            prompt=user_task.prompt,
            arrival_time=float(user_task.arrival_time),
            submit_time=float(self.now_seconds),
            completion_time=float(completion_time),
        )
        self._running_cloud_tasks.append(cloud_task)
        return cloud_task

    def _advance_time(self, dt: float) -> None:
        self.now_seconds = float(self.now_seconds + dt)
        self.node_manager.update(dt=dt, now=self.now_seconds)
        self._last_completed_latencies = []
        self._check_local_completion()
        self._check_cloud_completion()
        self.node_manager.renumber_groups()

    def _advance_time_and_arrivals(self, dt: float) -> None:
        lam = self.config.poisson_lambda_per_sec
        arrivals = self.task_manager._generator.poisson_arrivals(
            lam_per_sec=lam,
            dt_seconds=dt,
            now=self.now_seconds,
        )
        for t in arrivals:
            if self._num_generated_tasks >= self.config.total_tasks:
                break
            self.queue.push(t)
            self._num_generated_tasks += 1
        self._advance_time(dt)

    def _check_local_completion(self) -> None:
        if not self._running_local_tasks:
            return
        still_running: List[SDTask] = []
        for task in self._running_local_tasks:
            done = True
            for idx in task.assigned_node_indices:
                if self.node_manager.get_node_remaining_time(idx) > 0.5:
                    done = False
                    break
            if done:
                task.completion_time = self.now_seconds
                self._record_completion(float(task.completion_time - task.arrival_time))
            else:
                still_running.append(task)
        self._running_local_tasks = still_running

    def _check_cloud_completion(self) -> None:
        if not self._running_cloud_tasks:
            return
        still_running: List[CloudTask] = []
        for task in self._running_cloud_tasks:
            if task.completion_time <= self.now_seconds + 1e-9:
                self._record_completion(float(task.completion_time - task.arrival_time))
            else:
                still_running.append(task)
        self._running_cloud_tasks = still_running

    def _record_completion(self, latency: float) -> None:
        latency = max(0.0, float(latency))
        self._num_completed_tasks += 1
        self._completed_latency_sum += latency
        self._last_completed_latencies.append(latency)
        self._latency_history.append(latency)
        if latency > float(self.config.qos_latency_threshold):
            self._qos_violation_count += 1

    def _all_tasks_completed(self) -> bool:
        if self._num_generated_tasks < self.config.total_tasks:
            return False
        if len(self.queue) > 0:
            return False
        if len(self._running_local_tasks) > 0 or len(self._running_cloud_tasks) > 0:
            return False
        for i in range(self.num_nodes):
            if self.node_manager.get_node_remaining_time(i) > 0.0:
                return False
        return True

    # ===== Reward =====
    def _reward_from_components(self, t1: float, t2_avg: float, q_raw: float) -> float:
        cost_t1 = np.tanh(float(t1) / float(self.config.reward_t1_scale))
        cost_t2 = np.tanh(float(t2_avg) / float(self.config.reward_t2_scale))
        quality = np.tanh(float(q_raw) / float(self.config.reward_q_scale))
        self._last_quality = float(quality)
        total_cost = float(self.config.reward_t1_weight) * cost_t1 + float(self.config.reward_t2_weight) * cost_t2
        reward = -total_cost + float(self.config.reward_q_weight) * quality + float(self.config.reward_bais)
        return float(max(0.0, reward))

    def _queue_wait_mean(self) -> float:
        waits: List[float] = []
        for i in range(len(self.queue)):
            ut = self.queue.peek(i)
            if ut is not None:
                waits.append(max(0.0, float(self.now_seconds - ut.arrival_time)))
        return float(np.mean(waits)) if waits else 0.0

    def _compute_local_reward(self, task: SDTask) -> float:
        job_seconds = 0.0
        for idx in task.assigned_node_indices:
            job_seconds = max(job_seconds, float(self.node_manager.get_node_remaining_time(idx)))
        bandwidth_values = [float(self.config.bandwidth_by_state.get("good", 100.0)) for _ in task.assigned_node_indices]
        avg_bw = float(np.mean(bandwidth_values)) if bandwidth_values else 0.0
        transfer_seconds = float(self.config.image_size_mb) / max(1e-9, avg_bw)
        t1 = job_seconds + transfer_seconds
        t2 = self._queue_wait_mean()
        return self._reward_from_components(t1=t1, t2_avg=t2, q_raw=float(task.steps))

    def _compute_cloud_reward(self, task: CloudTask) -> float:
        t1 = max(0.0, float(task.completion_time - self.now_seconds))
        t2 = self._queue_wait_mean()
        q_raw = float(task.steps) * float(self.config.cloud_quality_multiplier)
        return self._reward_from_components(t1=t1, t2_avg=t2, q_raw=q_raw)

    # ===== Info / Metrics =====
    def _risk_metrics(self) -> Tuple[float, float, float]:
        if len(self._latency_history) == 0:
            return 0.0, 0.0, 0.0
        arr = np.asarray(self._latency_history, dtype=np.float32)
        p95 = float(np.percentile(arr, 95))
        alpha = float(np.clip(self.config.cvar_alpha, 0.5, 0.999))
        var = float(np.percentile(arr, alpha * 100.0))
        tail = arr[arr >= var]
        cvar = float(np.mean(tail)) if tail.size > 0 else var
        qos_violation_rate = float(self._qos_violation_count) / float(max(1, self._num_completed_tasks))
        return p95, cvar, qos_violation_rate

    def _build_info(self) -> Dict[str, Any]:
        cnt = int(len(self._last_completed_latencies))
        latency_sum = float(sum(self._last_completed_latencies)) if cnt > 0 else 0.0
        p95, cvar, qos_violation_rate = self._risk_metrics()
        routed_total = self._local_dispatch_count + self._cloud_dispatch_count
        route_local_ratio = float(self._local_dispatch_count) / float(max(1, routed_total))
        return {
            "completed_count": cnt,
            "completed_latency_sum": latency_sum,
            "quality": float(self._last_quality),
            "p95": p95,
            "cvar": cvar,
            "qos_violation": qos_violation_rate,
            "privacy_violation": float(self._privacy_violation_count),
            "route_local_ratio": route_local_ratio,
            # compatibility aliases for old/new metric names
            "latency_p95_proxy": p95,
            "cvar_cost": cvar,
        }


def make_joint_env(config: Optional[EnvConfig] = None, seed: Optional[int] = None):
    def _factory():
        return EdgeCloudJointEnv(config=config, seed=seed)

    try:  # pragma: no cover - optional dependency
        from tianshou.env import DummyVectorEnv

        return DummyVectorEnv([_factory])
    except Exception:
        return _factory()
