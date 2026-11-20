"""Two-tier Full-Step Mega CUDA Graph runtime for lossless IEEE-FP32 RL rollouts.

Tier 1 (``EnvFullStepGraph``):
  Captures the ENTIRE ``env.step()`` lifecycle into ONE ``torch.cuda.CUDAGraph`` +
  external ``wp.ScopedCapture``:
    process_action -> decimation x (PD + write + mjwarp.step + scene.update)
    -> overflow/nefc torch._assert_async (BEFORE reset_data!)
    -> termination -> reward -> branchless interval/step events
    -> branchless masked reset (mjwarp.reset_data(wp.bool) + AsyncResetPool cumsum)
    -> single forward_kinematics_and_sensors() -> command compute -> ShiftRegisterBuffer obs.

Tier 2 (``RolloutStepMegaGraph``):
  Captures 1 full rollout step (Actor/Critic forward + randn_like sampling +
  ``EnvFullStepGraph._body()`` + optional in-place obs_normalizer + transition staging +
  GPU nonfinite latch) with strict parameter AND buffer ``data_ptr()`` signature checks.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Sequence

import torch
import warp as wp

from mjlab.actuator import IdealPdActuator
from mjlab.envs.reset_pool import AsyncResetPool
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sensor.sensor import Sensor


class EnvFullStepGraph:
  """Tier 1 runner-agnostic full-step CUDA Graph for ``ManagerBasedRlEnv`` and subclasses."""

  def __init__(
    self,
    env: Any,
    *,
    reset_pools: Sequence[tuple[AsyncResetPool, Callable[[], Any] | None]] = (),
    default_horizon: int = 24,
  ) -> None:
    self.env = env
    self.graph: torch.cuda.CUDAGraph | None = None
    self.signature: tuple[Any, ...] | None = None
    self.warm_steps: int = 0
    self.captures: int = 0
    self.replays: int = 0

    # Use a PyTorch-owned NON-BLOCKING stream registered with Warp so CUDA 13.x
    # never triggers CUDA_ERROR_STREAM_CAPTURE_IMPLICIT (906) from a blocking stream.
    self.stream = torch.cuda.Stream(device=env.device)
    self.warp_stream = wp.Stream(
      device=env.sim.wp_device, cuda_stream=self.stream.cuda_stream
    )

    # Upgrade history & delay buffers to CUDA-Graph-safe chronological ShiftRegisterBuffer.
    env.observation_manager.enable_shift_register_buffers()
    # Freeze Python list[int] SceneEntityCfg index lists to GPU tensors / zero-copy slices
    # so reward/termination/obs/event indexing never triggers pageable H2D copies during capture.
    self._freeze_scene_entity_cfg_ids(env)

    self._graph_action_in = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim),
      dtype=torch.float32,
      device=env.device,
    )
    self._zero_dt = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    self._step_dt_buf = torch.full(
      (env.num_envs,), float(env.step_dt), dtype=torch.float32, device=env.device
    )

    # Metric layout: [Episode_Reward/* (mean), Episode_Termination/* (count),
    #                 Episode_Metrics/* (mean), Metrics/* (mean)]
    self._rew_keys: list[str] = list(env.reward_manager._episode_sums.keys())
    self._term_keys: list[str] = list(env.termination_manager._term_dones.keys())
    self._em_keys: list[str] = list(
      getattr(env.metrics_manager, "_episode_sums", {}).keys()
    )
    self._cmd_keys: list[str] = []
    for name, term in env.command_manager._terms.items():
      for mk in term.metrics.keys():
        self._cmd_keys.append(f"Metrics/{name}/{mk}")

    self._metric_names: list[str] = (
      [f"Episode_Reward/{k}" for k in self._rew_keys]
      + [f"Episode_Termination/{k}" for k in self._term_keys]
      + [f"Episode_Metrics/{k}" for k in self._em_keys]
      + list(self._cmd_keys)
    )
    # True -> divide sum by reset_count[t]; False -> raw count (Episode_Termination/*)
    self._metric_is_mean: list[bool] = (
      [True] * len(self._rew_keys)
      + [False] * len(self._term_keys)
      + [True] * len(self._em_keys)
      + [True] * len(self._cmd_keys)
    )
    self._num_metrics = len(self._metric_names)

    self._step_metric_sums = torch.zeros(
      max(self._num_metrics, 1), dtype=torch.float32, device=env.device
    )
    self._step_reset_count = torch.zeros((), dtype=torch.float32, device=env.device)

    self._horizon = max(1, int(default_horizon))
    self._iter_metric_sums = torch.zeros(
      (self._horizon, max(self._num_metrics, 1)),
      dtype=torch.float32,
      device=env.device,
    )
    self._iter_reset_counts = torch.zeros(
      self._horizon, dtype=torch.float32, device=env.device
    )
    self._iter_step_cursor: int = 0
    self._in_staged_iteration: bool = False
    self.last_nonzero_reset_steps: int = 0

    self._reset_pools: list[tuple[AsyncResetPool, Callable[[], Any] | None]] = list(
      reset_pools
    )
    self._empty_extras_log: dict[str, Any] = {}

  @staticmethod
  def _freeze_scene_entity_cfg_ids(env: Any) -> None:
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    def _convert_ids(val: Any) -> Any:
      if isinstance(val, list) and val and all(isinstance(x, int) for x in val):
        if val == list(range(val[0], val[0] + len(val))):
          return slice(val[0], val[0] + len(val))
        return torch.tensor(val, dtype=torch.long, device=env.device)
      return val

    def _visit(obj: Any) -> None:
      if isinstance(obj, SceneEntityCfg):
        for attr in ("body_ids", "joint_ids", "site_ids", "geom_ids", "actuator_ids"):
          if hasattr(obj, attr):
            setattr(obj, attr, _convert_ids(getattr(obj, attr)))
      elif isinstance(obj, dict):
        for v in obj.values():
          _visit(v)
      elif isinstance(obj, (list, tuple)):
        for v in obj:
          _visit(v)

    for term_cfg in getattr(env.reward_manager, "_term_cfgs", ()):
      _visit(term_cfg.params)
    for term_cfg in getattr(env.termination_manager, "_term_cfgs", ()):
      _visit(term_cfg.params)
    for term_cfg in getattr(env.metrics_manager, "_term_cfgs", ()):
      _visit(term_cfg.params)
    for cfgs in getattr(env.observation_manager, "_group_obs_term_cfgs", {}).values():
      for term_cfg in cfgs:
        _visit(term_cfg.params)
    for mode in ("reset", "interval", "step"):
      for term_cfg in getattr(env.event_manager, "_mode_term_cfgs", {}).get(mode, ()):
        _visit(term_cfg.params)

  @staticmethod
  def is_supported(env: Any) -> tuple[bool, str]:
    """Static eligibility guard for Tier 1 ``EnvFullStepGraph``."""
    if not str(env.device).startswith("cuda") or not torch.cuda.is_available():
      return False, "requires CUDA device"
    if not getattr(env.sim, "use_cuda_graph", False):
      return False, "sim.use_cuda_graph is disabled"
    if getattr(env.sim.cfg, "deterministic", None) in ("atomics", "all"):
      return (
        False,
        f"deterministic={env.sim.cfg.deterministic} incompatible with outer graph",
      )
    if getattr(env.sim.nan_guard, "enabled", False):
      return False, "nan_guard is enabled"
    if not getattr(env.cfg, "auto_reset", True):
      return False, "auto_reset is False"
    if getattr(env, "_twosim_hybrid_active", False):
      return False, "2SIM_TRAIN hybrid cosim is active"
    if getattr(env.recorder_manager, "active_terms", None):
      return False, "recorder_manager has active recorders"
    sensor_ctx = getattr(env.sim, "_sensor_context", None)
    if sensor_ctx is not None and getattr(sensor_ctx, "has_cameras", False):
      return False, "sim._sensor_context has active cameras"
    if getattr(env.metrics_manager, "_substep_term_indices", None):
      return False, "substep metrics configured"
    if getattr(env.curriculum_manager, "active_terms", None):
      for name in env.curriculum_manager.active_terms:
        cfg = env.curriculum_manager.get_term_cfg(name)
        if not (
          hasattr(cfg.func, "compute_masked")
          or getattr(cfg.func, "_iter_level_curriculum", False)
        ):
          return (
            False,
            f"curriculum term '{name}' lacks compute_masked() or _iter_level_curriculum",
          )

    from mjlab.actuator.builtin_actuator import (
      BuiltinMotorActuator,
      BuiltinMuscleActuator,
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
    )
    from mjlab.sensor.raycast_sensor import RayCastSensor

    allowed_actuators = (
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
      BuiltinMotorActuator,
      BuiltinMuscleActuator,
      IdealPdActuator,
    )
    for entity in env.scene.entities.values():
      for act in getattr(entity, "actuators", ()):
        if type(act) not in allowed_actuators or getattr(act, "has_delay", False):
          return False, f"actuator {type(act).__name__} unsupported or has delay"

    for sensor in env.scene.sensors.values():
      if isinstance(sensor, ContactSensor):
        if type(sensor).update is not ContactSensor.update:
          return False, "custom ContactSensor.update override"
      elif isinstance(sensor, RayCastSensor):
        continue
      elif type(sensor).update is not Sensor.update:
        return False, f"stateful sensor {type(sensor).__name__}"

    for term in env.action_manager._terms.values():
      if getattr(getattr(term, "cfg", None), "delay_max_ticks", 0) != 0:
        return False, "action term has non-zero transport delay"
      if not hasattr(term, "reset_masked"):
        return False, f"action term {type(term).__name__} lacks reset_masked()"

    ok, reason = env.event_manager.is_masked_graph_compatible()
    if not ok:
      return False, reason
    ok, reason = env.command_manager.is_masked_graph_compatible()
    if not ok:
      return False, reason
    ok, reason = env.observation_manager.is_masked_graph_compatible()
    if not ok:
      return False, reason

    return True, "ok"

  def add_reset_pool(
    self,
    pool: AsyncResetPool,
    state_snapshot_fn: Callable[[], Any] | None = None,
  ) -> None:
    self._reset_pools.append((pool, state_snapshot_fn))

  def begin_iteration(self, num_steps: int) -> None:
    """Prepare staging buffers and order GPU stream after completed async CPU refills."""
    num_steps = max(1, int(num_steps))
    if num_steps > self._horizon:
      self._horizon = num_steps
      self._iter_metric_sums = torch.zeros(
        (self._horizon, max(self._num_metrics, 1)),
        dtype=torch.float32,
        device=self.env.device,
      )
      self._iter_reset_counts = torch.zeros(
        self._horizon, dtype=torch.float32, device=self.env.device
      )
    self._iter_step_cursor = 0
    self._in_staged_iteration = True
    # Advance step-counter-based curricula (e.g. commands_vel) once per iteration outside graph.
    self._iter_curriculum_logs: dict[str, float] = {}
    if getattr(self.env.curriculum_manager, "active_terms", None):
      for name in self.env.curriculum_manager.active_terms:
        cfg = self.env.curriculum_manager.get_term_cfg(name)
        if getattr(cfg.func, "_iter_level_curriculum", False):
          out = cfg.func(self.env, None, **cfg.params)
          if isinstance(out, dict):
            for k, v in out.items():
              self._iter_curriculum_logs[f"Curriculum/{name}/{k}"] = float(
                v.item() if torch.is_tensor(v) else v
              )
          elif out is not None:
            self._iter_curriculum_logs[f"Curriculum/{name}"] = float(
              out.item() if torch.is_tensor(out) else out
            )
    for pool, _ in self._reset_pools:
      pool.wait_refill_on_stream(self.stream)

  def end_iteration(self) -> list[dict[str, float]]:
    """ONE single iteration-boundary D2H flush for zero-dilution ep_infos + async pool refills."""
    steps = min(self._iter_step_cursor, self._horizon)
    self._in_staged_iteration = False
    self._iter_step_cursor = 0

    ep_infos: list[dict[str, float]] = []
    if steps == 0 and not self._reset_pools:
      return ep_infos

    # Pack counts + metric sums + pool consumed counters into ONE coalesced D2H readback.
    chunks: list[torch.Tensor] = []
    if steps > 0:
      chunks.append(self._iter_reset_counts[:steps].to(dtype=torch.float64))
      if self._num_metrics > 0:
        chunks.append(
          self._iter_metric_sums[:steps, : self._num_metrics]
          .reshape(-1)
          .to(dtype=torch.float64)
        )
    for pool, _ in self._reset_pools:
      chunks.append(pool.consumed_iter_tensor.to(dtype=torch.float64).reshape(1))

    if not chunks:
      return ep_infos

    packed = torch.cat(chunks, dim=0).cpu().tolist()
    offset = 0
    self.last_nonzero_reset_steps = 0
    if steps > 0:
      counts = packed[offset : offset + steps]
      offset += steps
      if self._num_metrics > 0:
        m_flat = packed[offset : offset + steps * self._num_metrics]
        offset += steps * self._num_metrics
        accum = [0.0] * self._num_metrics
        nonzero_steps = 0
        for t in range(steps):
          cnt = counts[t]
          # Strictly skip steps with 0 resets to prevent 0.0-dilution of Episode_Reward/* logs!
          if cnt <= 0.0:
            continue
          nonzero_steps += 1
          base = t * self._num_metrics
          for m_idx in range(self._num_metrics):
            val = float(m_flat[base + m_idx])
            accum[m_idx] += val / cnt if self._metric_is_mean[m_idx] else val
        self.last_nonzero_reset_steps = nonzero_steps
        if nonzero_steps > 0:
          # Pre-average across active reset steps into ONE dict so rsl_rl Logger.log()
          # avoids O(H * M) per-key GPU tensor syncs (~350 -> ~19 syncs/iter).
          inv_steps = 1.0 / float(nonzero_steps)
          ep_infos.append(
            {
              name: accum[m_idx] * inv_steps
              for m_idx, name in enumerate(self._metric_names)
            }
          )

    # Reset GPU consumed counters and kick async CPU refills overlapped behind PPO.update().
    for p_idx, (pool, snap_fn) in enumerate(self._reset_pools):
      consumed = int(round(packed[offset + p_idx]))
      pool.consumed_iter_tensor.zero_()
      if consumed > 0:
        state_snap = snap_fn() if snap_fn is not None else None
        pool.kick_async_refill(consumed, state_snap)

    # Collect iteration-level curriculum telemetry (e.g. commands_vel + terrain_levels_vel).
    curr_logs = dict(getattr(self, "_iter_curriculum_logs", {}))
    if getattr(self.env.curriculum_manager, "active_terms", None):
      for name in self.env.curriculum_manager.active_terms:
        cfg = self.env.curriculum_manager.get_term_cfg(name)
        log_fn = getattr(cfg.func, "_iter_level_log", None)
        if log_fn is not None:
          out = log_fn(self.env, **cfg.params)
          if isinstance(out, dict):
            for k, v in out.items():
              curr_logs[f"Curriculum/{name}/{k}"] = float(
                v.item() if torch.is_tensor(v) else v
              )
    if curr_logs:
      if ep_infos:
        ep_infos[-1].update(curr_logs)
      else:
        ep_infos.append(curr_logs)

    return ep_infos

  def _update_scene(self) -> None:
    env = self.env
    for entity in env.scene.entities.values():
      entity.update(env.physics_dt)
    for sensor in env.scene.sensors.values():
      sensor.update(env.physics_dt)

  def _body(self) -> None:
    """Single full-step body captured directly inside CUDA Graph (zero host syncs)."""
    env = self.env
    post_check = getattr(env.sim, "_post_decimation_check", None)
    step_fn = (
      getattr(env.sim, "_unwrapped_step", env.sim.step)
      if post_check is not None
      else env.sim.step
    )

    # 1. Process action in-place from static input buffer.
    env.action_manager.process_action(self._graph_action_in)

    # 2. Decimation physics substeps.
    for _ in range(env.cfg.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      step_fn()
      self._update_scene()
      env.metrics_manager.compute_substep()

    # 3. Overflow / constraint capacity check BEFORE mjwarp.reset_data clears d.overflow & d.nefc!
    if post_check is not None:
      post_check()

    # 4. Episode counter + termination + reward.
    env.episode_length_buf.add_(1)
    reset_mask = env.termination_manager.compute()
    env.reset_buf = reset_mask
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs

    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
    env.metrics_manager.compute()
    pre_reset_hook = getattr(env, "_mega_pre_reset_hook", None)
    if pre_reset_hook is not None:
      pre_reset_hook()

    # 5. Branchless step/interval disturbance events (e.g., push_robot).
    if "step" in env.event_manager.available_modes:
      env.event_manager.apply_step_masked(dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
      env.event_manager.apply_interval_masked(dt=env.step_dt)

    # 6. Branchless masked reset + metric accumulation into static buffers.
    self._step_reset_count.copy_(reset_mask.sum(dtype=torch.float32))
    metric_parts: list[torch.Tensor] = []

    if env.curriculum_manager.active_terms:
      for name in env.curriculum_manager.active_terms:
        cfg = env.curriculum_manager.get_term_cfg(name)
        if hasattr(cfg.func, "compute_masked"):
          cfg.func.compute_masked(env, reset_mask, **cfg.params)

    env.sim.reset_mask(reset_mask)
    env.scene.reset_masked(reset_mask)
    if "reset" in env.event_manager.available_modes:
      env.event_manager.apply_reset_masked(reset_mask)
    env.observation_manager.reset_masked(reset_mask)
    env.action_manager.reset_masked(reset_mask)

    _, rew_sums = env.reward_manager.reset_masked(reset_mask)
    if rew_sums is not None:
      metric_parts.append(rew_sums)

    _, term_counts = env.termination_manager.reset_masked(reset_mask)
    if term_counts is not None:
      metric_parts.append(term_counts)

    if self._em_keys and hasattr(env.metrics_manager, "reset_masked"):
      _, em_sums = env.metrics_manager.reset_masked(reset_mask)
      if em_sums is not None:
        metric_parts.append(em_sums)

    _, cmd_sums_list = env.command_manager.reset_masked(reset_mask)
    metric_parts.extend(cmd_sums_list)

    if metric_parts:
      torch.cat(metric_parts, dim=0, out=self._step_metric_sums[: self._num_metrics])

    env.event_manager.reset_masked(reset_mask)
    env.episode_length_buf.masked_fill_(reset_mask, 0)
    if hasattr(env, "_manual_reset_pending"):
      env._manual_reset_pending.masked_fill_(reset_mask, False)

    # Fold any command wraparound resamples BEFORE the single post-step forward.
    for term in env.command_manager._terms.values():
      wrap_fn = getattr(term, "pre_forward_wraparound_masked", None)
      if wrap_fn is not None:
        wrap_mask = wrap_fn(reset_mask)
        term._resample_command_masked(wrap_mask)

    post_reset_hook = getattr(env, "_mega_post_reset_hook", None)
    if post_reset_hook is not None:
      post_reset_hook(reset_mask)

    env.scene.write_data_to_sim()

    # 7. Single post-step kinematics/sensor forward.
    if getattr(env, "_use_kinematic_post_step_forward", False):
      env.sim.forward_kinematics_and_sensors()
    else:
      forward_fn = getattr(env.sim, "_unwrapped_forward", env.sim.forward)
      forward_fn()

    # 8. Branchless command update (dt=0 for reset envs, step_dt for active envs).
    torch.where(reset_mask, self._zero_dt, self._step_dt_buf, out=env._command_dt)
    env.command_manager.compute_masked(dt=env._command_dt)

    # 9. Raycast sensing (pure GPU + Warp when no cameras) + ShiftRegisterBuffer obs.
    env.sim.sense()
    sdk_hook = getattr(env, "_mega_sdk_bank_obs_overrides", None)
    sdk_overrides = sdk_hook(reset_mask) if sdk_hook is not None else None
    env.obs_buf = env.observation_manager.compute_masked(
      reset_mask=reset_mask,
      sdk_bank_overrides=sdk_overrides,
    )

  def _build_signature(self) -> tuple[Any, ...]:
    env = self.env
    histories = tuple(
      buf.data_ptr()
      for sensor in env.scene.sensors.values()
      if isinstance(sensor, ContactSensor) and sensor._history_state is not None
      for buf in sensor._history_state.values()
    )
    post_check = getattr(env.sim, "_post_decimation_check", None)
    cmd_ranges = tuple(
      (
        name,
        getattr(getattr(term.cfg, "ranges", None), "lin_vel_x", None),
        getattr(getattr(term.cfg, "ranges", None), "lin_vel_y", None),
        getattr(getattr(term.cfg, "ranges", None), "ang_vel_z", None),
      )
      for name, term in env.command_manager._terms.items()
    )
    return (
      id(env.sim.wp_model),
      id(env.sim.wp_data),
      id(env.sim.step_graph),
      getattr(env.sim.step, "__func__", env.sim.step),
      getattr(post_check, "__func__", post_check),
      env.cfg.decimation,
      bool(getattr(env, "_use_kinematic_post_step_forward", False)),
      histories,
      cmd_ranges,
    )

  def step(self, action: torch.Tensor):
    """Execute one full environment step via captured CUDA Graph (or 2-step warmup)."""
    env = self.env
    sig = self._build_signature()
    if self.signature != sig:
      if self.graph is not None:
        self.stream.synchronize()
      self.graph = None
      self.warm_steps = 0
      self.signature = sig

    current = torch.cuda.current_stream(env.device)
    self.stream.wait_stream(current)

    if self.warm_steps < 2:
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      try:
        env.sim._inline_warp_kernels = True
        with (
          torch.inference_mode(False),
          torch.no_grad(),
          torch.cuda.stream(self.stream),
          wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False),
        ):
          self._graph_action_in.copy_(action.to(device=env.device, dtype=torch.float32))
          self._body()
      finally:
        env.sim._inline_warp_kernels = prev_inline
        current.wait_stream(self.stream)
      self.warm_steps += 1
    else:
      use_cuda_graph = env.sim.use_cuda_graph
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      was_enabled = gc.isenabled()
      try:
        with (
          torch.inference_mode(False),
          torch.no_grad(),
          torch.cuda.stream(self.stream),
          wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False),
        ):
          self._graph_action_in.copy_(action.to(device=env.device, dtype=torch.float32))
          if self.graph is None:
            # Nested wp.capture_launch cannot be captured inside outer graph.
            env.sim.use_cuda_graph = False
            env.sim._inline_warp_kernels = True
            gc.disable()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
              self.graph,
              stream=self.stream,
              capture_error_mode="thread_local",
            ):
              with wp.ScopedCapture(
                stream=self.warp_stream,
                external=True,
                capture_mode=wp.CaptureMode.THREAD_LOCAL,
              ):
                self._body()
            self.captures += 1
          self.graph.replay()
          self.replays += 1
      finally:
        env.sim.use_cuda_graph = use_cuda_graph
        env.sim._inline_warp_kernels = prev_inline
        if was_enabled:
          gc.enable()
        current.wait_stream(self.stream)

    # Host bookkeeping (0 CUDA syncs).
    env._sim_step_counter += env.cfg.decimation
    env.common_step_counter += 1
    for sensor in env.scene.sensors.values():
      sensor._invalidate_cache()

    if self._in_staged_iteration and self._iter_step_cursor < self._horizon:
      idx = self._iter_step_cursor
      if self._num_metrics > 0:
        self._iter_metric_sums[idx, : self._num_metrics].copy_(
          self._step_metric_sums[: self._num_metrics]
        )
      self._iter_reset_counts[idx].copy_(self._step_reset_count)
      self._iter_step_cursor += 1
      env.extras["log"] = self._empty_extras_log
    else:
      env.extras["log"] = self._empty_extras_log

    # Return cloned non-aliased step outputs so callers storing pre-step obs_t
    # (e.g. PPO.act(obs_t) before env.step(act) -> PPO.process_env_step) never
    # have obs_t clobbered in-place by the CUDA Graph's static observation buffers!
    obs_out: dict[str, Any] = {}
    for k, v in env.obs_buf.items():
      if isinstance(v, dict):
        obs_out[k] = {
          sk: sv.clone() if torch.is_tensor(sv) else sv for sk, sv in v.items()
        }
      elif torch.is_tensor(v):
        obs_out[k] = v.clone()
      else:
        obs_out[k] = v

    return (
      obs_out,
      env.reward_buf.clone(),
      env.reset_terminated.clone(),
      env.reset_time_outs.clone(),
      env.extras,
    )

  def close(self) -> None:
    for pool, _ in self._reset_pools:
      pool.close()
    self._reset_pools.clear()


class RolloutStepMegaGraph:
  """Tier 2 1-step Mega CUDA Graph fusing Actor/Critic + EnvFullStepGraph + Normalizer.

  Validates ALL model parameter AND buffer ``data_ptr()``s (including normalizer
  ``_mean/_var/_std``) at the start of every rollout iteration and automatically
  re-captures if any address changed.
  """

  def __init__(
    self,
    env_graph: EnvFullStepGraph,
    modules: Sequence[torch.nn.Module],
    step_body_fn: Callable[[], None],
  ) -> None:
    self.env_graph = env_graph
    self.modules = tuple(modules)
    self.step_body_fn = step_body_fn
    self.graph: torch.cuda.CUDAGraph | None = None
    self.signature: tuple[Any, ...] | None = None
    self.warm_steps: int = 0
    self.captures: int = 0
    self.replays: int = 0

  def _pointer_signature(self) -> tuple[Any, ...]:
    ptrs: list[int] = []
    for mod in self.modules:
      for p in mod.parameters():
        ptrs.append(p.data_ptr())
      for b in mod.buffers():
        ptrs.append(b.data_ptr())
    return (self.env_graph._build_signature(), tuple(ptrs))

  def verify_or_invalidate(self) -> None:
    sig = self._pointer_signature()
    if self.signature != sig:
      if self.graph is not None:
        self.env_graph.stream.synchronize()
      self.graph = None
      self.warm_steps = 0
      self.signature = sig

  def run_step(self) -> None:
    env = self.env_graph.env
    stream = self.env_graph.stream
    warp_stream = self.env_graph.warp_stream
    current = torch.cuda.current_stream(env.device)
    stream.wait_stream(current)

    if self.warm_steps < 2:
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      try:
        env.sim._inline_warp_kernels = True
        with (
          torch.inference_mode(False),
          torch.no_grad(),
          torch.cuda.stream(stream),
          wp.ScopedStream(warp_stream, sync_enter=False, sync_exit=False),
        ):
          self.step_body_fn()
      finally:
        env.sim._inline_warp_kernels = prev_inline
        current.wait_stream(stream)
      self.warm_steps += 1
    else:
      use_cuda_graph = env.sim.use_cuda_graph
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      was_enabled = gc.isenabled()
      try:
        with (
          torch.inference_mode(False),
          torch.no_grad(),
          torch.cuda.stream(stream),
          wp.ScopedStream(warp_stream, sync_enter=False, sync_exit=False),
        ):
          if self.graph is None:
            env.sim.use_cuda_graph = False
            env.sim._inline_warp_kernels = True
            gc.disable()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
              self.graph,
              stream=stream,
              capture_error_mode="thread_local",
            ):
              with wp.ScopedCapture(
                stream=warp_stream,
                external=True,
                capture_mode=wp.CaptureMode.THREAD_LOCAL,
              ):
                self.step_body_fn()
            self.captures += 1
          self.graph.replay()
          self.replays += 1
      finally:
        env.sim.use_cuda_graph = use_cuda_graph
        env.sim._inline_warp_kernels = prev_inline
        if was_enabled:
          gc.enable()
        current.wait_stream(stream)

    env._sim_step_counter += env.cfg.decimation
    env.common_step_counter += 1
    for sensor in env.scene.sensors.values():
      sensor._invalidate_cache()
    eg = self.env_graph
    if eg._in_staged_iteration and eg._iter_step_cursor < eg._horizon:
      idx = eg._iter_step_cursor
      if eg._num_metrics > 0:
        eg._iter_metric_sums[idx, : eg._num_metrics].copy_(
          eg._step_metric_sums[: eg._num_metrics]
        )
      eg._iter_reset_counts[idx].copy_(eg._step_reset_count)
      eg._iter_step_cursor += 1
