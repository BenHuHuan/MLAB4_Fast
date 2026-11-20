"""Background CPU async reset-state ring pool + branchless GPU prefix-sum consumer.

Eliminates per-step ``reset_buf.nonzero()`` GPU<->CPU synchronization by pre-sampling
reset states on idle same-NUMA CPU P-core(s) into a pinned-memory staging buffer and
asynchronously DMA'ing them into a power-of-2 GPU ring buffer overlapped behind
``PPO.update()``. Inside ``EnvFullStepGraph``, GPU resets consume compact contiguous
slots via branchless ``torch.cumsum`` + bitmask wrap (zero host sync, CUDA-Graph safe).
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import numpy as np
import torch

from mjlab.utils.cpu_runtime import get_worker_reset_pool_cpu_ids, pin_current_thread


def next_pow2(n: int) -> int:
  n = max(1, int(n))
  return 1 << (n - 1).bit_length()


def choose_ring_pool_size(
  num_envs: int,
  num_steps_per_env: int,
  *,
  bytes_per_slot: int = 768,
  max_vram_bytes: int = 128 * 1024 * 1024,
  min_slots: int = 1024,
) -> int:
  """Choose power-of-2 pool size >= ``num_envs * num_steps_per_env`` within VRAM budget.

  For 4,096 envs x 24 steps = 98,304 transitions at ~612 B/slot, ``next_pow2(98304)``
  is ``131,072`` (~80 MiB VRAM), mathematically eliminating mid-iteration wraparound
  even under pathological 100%-every-step termination.
  """
  target = next_pow2(max(min_slots, int(num_envs) * max(1, int(num_steps_per_env))))
  cap_by_mem = max(min_slots, max_vram_bytes // max(1, int(bytes_per_slot)))
  # Largest power of 2 <= cap_by_mem
  max_pow2 = 1 << (int(cap_by_mem).bit_length() - 1)
  return max(min_slots, min(target, max_pow2))


@dataclass(frozen=True)
class FieldSpec:
  """Shape (excluding leading slot dim) and torch dtype for one SoA pool field."""

  shape: tuple[int, ...]
  dtype: torch.dtype


class AsyncResetPool:
  """Power-of-2 GPU ring buffer refilled asynchronously by a pinned CPU worker thread.

  Producer-consumer protocol:
  1. ``prefill_all()`` synchronously fills all ``P`` slots on startup.
  2. During rollout (inside ``EnvFullStepGraph``), ``consume(mask)`` pops ``K`` slots
     per step via branchless ``cumsum`` + ``& (P - 1)`` with ZERO host syncs.
  3. At iteration end (single D2H flush), caller reads ``consumed_iter`` (asserting
     ``consumed <= P``), resets ``consumed_iter.zero_()``, and calls
     ``kick_async_refill(consumed, producer_state)`` right before ``PPO.update()``.
  4. At the start of the next rollout, caller calls ``wait_refill_on_stream(stream)``,
     which joins the already-completed CPU future and inserts a zero-CPU-stall GPU
     ``stream.wait_event(copy_done_event)``.
  """

  def __init__(
    self,
    *,
    pool_size: int,
    field_specs: dict[str, FieldSpec],
    device: str | torch.device,
    seed: int = 0,
    rank: int = 0,
    producer_fn: Callable[[int, np.random.Generator, Any, dict[str, np.ndarray]], None],
    worker_cpu_ids: Sequence[int] | None = None,
  ) -> None:
    if pool_size <= 0 or (pool_size & (pool_size - 1)) != 0:
      raise ValueError(f"pool_size must be a positive power of 2, got {pool_size}")
    self.pool_size = int(pool_size)
    self._pool_mask = self.pool_size - 1
    self.device = torch.device(device)
    self._is_cuda = self.device.type == "cuda"
    self.field_specs = dict(field_specs)
    self._producer_fn = producer_fn

    self._rng = np.random.Generator(
      np.random.PCG64(int(seed) + 1009 * int(rank) + 7919)
    )

    # Allocate GPU ring tensors and pinned host staging buffers.
    self._gpu_pool: dict[str, torch.Tensor] = {}
    self._host_staging: dict[str, torch.Tensor] = {}
    self._host_staging_np: dict[str, np.ndarray] = {}
    for name, spec in self.field_specs.items():
      full_shape = (self.pool_size, *spec.shape)
      self._gpu_pool[name] = torch.empty(
        full_shape, dtype=spec.dtype, device=self.device
      )
      host_t = torch.empty(
        full_shape, dtype=spec.dtype, device="cpu", pin_memory=self._is_cuda
      )
      self._host_staging[name] = host_t
      self._host_staging_np[name] = host_t.numpy()

    # GPU-side ring cursors (mutated purely on GPU inside CUDA Graph).
    self._head = torch.zeros((), dtype=torch.int64, device=self.device)
    self._consumed_iter = torch.zeros((), dtype=torch.int64, device=self.device)
    self._tail_cpu: int = 0  # Next slot index to refill on CPU.
    self._total_consumed: int = 0
    self._total_refills: int = 0

    self._copy_stream = torch.cuda.Stream(device=self.device) if self._is_cuda else None
    self._copy_done_event = (
      torch.cuda.Event(enable_timing=False) if self._is_cuda else None
    )

    cpus = (
      tuple(int(c) for c in worker_cpu_ids)
      if worker_cpu_ids is not None
      else get_worker_reset_pool_cpu_ids()
    )

    def _init_worker() -> None:
      if cpus:
        pin_current_thread(cpus, is_worker=True)

    self._executor = ThreadPoolExecutor(
      max_workers=1,
      thread_name_prefix="mlab4_reset_pool",
      initializer=_init_worker if cpus else None,
    )
    self._pending_future: Future[None] | None = None
    self._closed = False
    import atexit as _atexit
    import weakref as _weakref

    _self_ref = _weakref.ref(self)

    def _atexit_close() -> None:
      obj = _self_ref()
      if obj is not None and not obj._closed:
        try:
          obj.close()
        except Exception:
          pass

    _atexit.register(_atexit_close)

  @property
  def head_tensor(self) -> torch.Tensor:
    return self._head

  @property
  def consumed_iter_tensor(self) -> torch.Tensor:
    return self._consumed_iter

  @property
  def refill_count(self) -> int:
    return self._total_refills

  def consume(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Branchless CUDA-Graph-safe slot consumer (ZERO host syncs).

    Every lane ``i`` with ``mask[i] == True`` receives a unique consecutive slot from
    the ring; lanes with ``mask[i] == False`` index a valid slot ``(head - 1) & mask``
    whose values are subsequently masked out by ``torch.where``.
    """
    r_int = mask.to(device=self.device, dtype=torch.int64)
    offsets = torch.cumsum(r_int, dim=0) - 1
    slots = (self._head + offsets) & self._pool_mask
    cnt = r_int.sum()
    self._head.add_(cnt).bitwise_and_(self._pool_mask)
    self._consumed_iter.add_(cnt)
    return {k: buf.index_select(0, slots) for k, buf in self._gpu_pool.items()}

  def _produce_and_dma(self, count: int, start_slot: int, producer_state: Any) -> None:
    if count <= 0:
      return
    count = min(int(count), self.pool_size)
    # Prepare contiguous numpy views [0:count] of pinned staging tensors.
    views = {k: arr[:count] for k, arr in self._host_staging_np.items()}
    self._producer_fn(count, self._rng, producer_state, views)

    start = start_slot & self._pool_mask
    first_n = min(count, self.pool_size - start)
    second_n = count - first_n

    if self._is_cuda:
      assert self._copy_stream is not None and self._copy_done_event is not None
      with torch.cuda.stream(self._copy_stream):
        for k, gpu_buf in self._gpu_pool.items():
          host_buf = self._host_staging[k]
          gpu_buf[start : start + first_n].copy_(host_buf[:first_n], non_blocking=True)
          if second_n > 0:
            gpu_buf[:second_n].copy_(
              host_buf[first_n : first_n + second_n], non_blocking=True
            )
        self._copy_done_event.record(self._copy_stream)
    else:
      for k, gpu_buf in self._gpu_pool.items():
        host_buf = self._host_staging[k]
        gpu_buf[start : start + first_n].copy_(host_buf[:first_n])
        if second_n > 0:
          gpu_buf[:second_n].copy_(host_buf[first_n : first_n + second_n])

  def prefill_all(self, producer_state: Any = None) -> None:
    """Synchronously fill all ``P`` slots on startup (before first rollout)."""
    self.wait_refill_on_stream(None)
    self._head.zero_()
    self._consumed_iter.zero_()
    self._tail_cpu = 0
    self._produce_and_dma(self.pool_size, 0, producer_state)
    if self._is_cuda and self._copy_done_event is not None:
      self._copy_done_event.synchronize()

  def kick_async_refill(self, consumed: int, producer_state: Any = None) -> None:
    """Enqueue async CPU refill + H2D DMA for the ``consumed`` slots used last iteration."""
    if self._closed:
      return
    if self._pending_future is not None:
      self._pending_future.result()
      self._pending_future = None
    consumed = int(consumed)
    if consumed <= 0:
      return
    if consumed > self.pool_size:
      raise RuntimeError(
        f"AsyncResetPool wraparound detected: consumed={consumed} > pool_size={self.pool_size}. "
        "Increase pool_size or chunk rollout steps."
      )
    start_slot = self._tail_cpu
    self._tail_cpu = (self._tail_cpu + consumed) & self._pool_mask
    self._total_consumed += consumed
    self._total_refills += 1
    self._pending_future = self._executor.submit(
      self._produce_and_dma, consumed, start_slot, producer_state
    )

  def wait_refill_on_stream(self, stream: torch.cuda.Stream | None = None) -> None:
    """Join completed CPU refill and order GPU stream after async DMA (zero CPU GPU-wait)."""
    if self._pending_future is not None:
      self._pending_future.result()
      self._pending_future = None
      if self._is_cuda and self._copy_done_event is not None:
        target = (
          stream if stream is not None else torch.cuda.current_stream(self.device)
        )
        target.wait_event(cast(Any, self._copy_done_event))

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self._pending_future is not None:
        self._pending_future.result()
        self._pending_future = None
    finally:
      self._executor.shutdown(wait=True)


def _quat_from_euler_xyz_np(
  roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
) -> np.ndarray:
  cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
  cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
  cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  return np.stack((w, x, y, z), axis=-1).astype(np.float32, copy=False)


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return np.stack(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    axis=-1,
  ).astype(np.float32, copy=False)


def build_mjlab_motion_reset_pool(
  motion_cmd: Any,
  *,
  num_steps_per_env: int = 24,
  seed: int = 0,
  rank: int = 0,
) -> AsyncResetPool:
  """Build an ``AsyncResetPool`` for ``mjlab.tasks.tracking.mdp.MotionCommand``."""
  motion = motion_cmd.motion
  cfg = motion_cmd.cfg
  jpos_np = motion.joint_pos.detach().cpu().numpy().astype(np.float32, copy=False)
  jvel_np = motion.joint_vel.detach().cpu().numpy().astype(np.float32, copy=False)
  rpos_np = (
    motion.body_pos_w[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
  )
  rquat_np = (
    motion.body_quat_w[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
  )
  rlin_np = (
    motion.body_lin_vel_w[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
  )
  rang_np = (
    motion.body_ang_vel_w[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
  )
  total_steps = int(motion.time_step_total)
  bin_count = int(motion_cmd.bin_count)
  kernel_np = motion_cmd.kernel.detach().cpu().numpy().astype(np.float64, copy=False)
  kernel_size = int(cfg.adaptive_kernel_size)
  uniform_ratio = float(cfg.adaptive_uniform_ratio)
  sampling_mode = str(cfg.sampling_mode)

  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  pose_lo = np.array(
    [float(cfg.pose_range.get(k, (0.0, 0.0))[0]) for k in keys], dtype=np.float32
  )
  pose_hi = np.array(
    [float(cfg.pose_range.get(k, (0.0, 0.0))[1]) for k in keys], dtype=np.float32
  )
  vel_lo = np.array(
    [float(cfg.velocity_range.get(k, (0.0, 0.0))[0]) for k in keys], dtype=np.float32
  )
  vel_hi = np.array(
    [float(cfg.velocity_range.get(k, (0.0, 0.0))[1]) for k in keys], dtype=np.float32
  )
  j_lo, j_hi = float(cfg.joint_position_range[0]), float(cfg.joint_position_range[1])
  n_joints = int(jpos_np.shape[1])

  def _produce(
    count: int,
    rng: np.random.Generator,
    bin_failed_np: np.ndarray | None,
    out: dict[str, np.ndarray],
  ) -> None:
    if sampling_mode == "start":
      steps = np.zeros(count, dtype=np.int64)
    elif sampling_mode == "uniform":
      steps = rng.integers(0, max(total_steps, 1), size=count, dtype=np.int64)
    else:
      bf = (
        np.asarray(bin_failed_np, dtype=np.float64)
        if bin_failed_np is not None
        else np.zeros(bin_count, dtype=np.float64)
      )
      probs = bf + uniform_ratio / float(max(bin_count, 1))
      if kernel_size > 1:
        padded = np.pad(probs, (0, kernel_size - 1), mode="edge")
        probs = np.correlate(padded, kernel_np, mode="valid")
      s = float(np.sum(probs))
      probs_f64 = (
        np.asarray(probs, dtype=np.float64) / s
        if s > 0.0
        else np.full(bin_count, 1.0 / max(bin_count, 1), dtype=np.float64)
      )
      sampled_bins = rng.choice(bin_count, size=count, replace=True, p=probs_f64)
      frac = rng.uniform(0.0, 1.0, size=count)
      steps = np.clip(
        ((sampled_bins + frac) / max(bin_count, 1) * max(total_steps - 1, 0)).astype(
          np.int64
        ),
        0,
        max(total_steps - 1, 0),
      )

    p_noise = rng.uniform(pose_lo, pose_hi, size=(count, 6)).astype(np.float32)
    v_noise = rng.uniform(vel_lo, vel_hi, size=(count, 6)).astype(np.float32)
    j_noise = rng.uniform(j_lo, j_hi, size=(count, n_joints)).astype(np.float32)

    out["time_steps"][:] = steps
    out["root_pos_local"][:] = rpos_np[steps] + p_noise[:, :3]
    ori_delta = _quat_from_euler_xyz_np(p_noise[:, 3], p_noise[:, 4], p_noise[:, 5])
    out["root_quat"][:] = _quat_mul_np(ori_delta, rquat_np[steps])
    out["root_lin_vel"][:] = rlin_np[steps] + v_noise[:, :3]
    out["root_ang_vel"][:] = rang_np[steps] + v_noise[:, 3:]
    out["joint_pos"][:] = jpos_np[steps] + j_noise
    out["joint_vel"][:] = jvel_np[steps]

  specs = {
    "time_steps": FieldSpec((), torch.int64),
    "root_pos_local": FieldSpec((3,), torch.float32),
    "root_quat": FieldSpec((4,), torch.float32),
    "root_lin_vel": FieldSpec((3,), torch.float32),
    "root_ang_vel": FieldSpec((3,), torch.float32),
    "joint_pos": FieldSpec((n_joints,), torch.float32),
    "joint_vel": FieldSpec((n_joints,), torch.float32),
  }
  bytes_per_slot = 8 + (3 + 4 + 3 + 3 + 2 * n_joints) * 4
  pool_size = choose_ring_pool_size(
    motion_cmd.num_envs, num_steps_per_env, bytes_per_slot=bytes_per_slot
  )
  pool = AsyncResetPool(
    pool_size=pool_size,
    field_specs=specs,
    device=motion_cmd.device,
    seed=seed,
    rank=rank,
    producer_fn=_produce,
  )
  pool.prefill_all(motion_cmd.bin_failed_count.detach().cpu().numpy())
  return pool
