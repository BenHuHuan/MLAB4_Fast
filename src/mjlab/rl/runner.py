import os
import time
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.async_io import (
  AsyncCheckpointWorker,
  BatchedAsyncLogWriter,
  consume_prewarmed_async_worker,
)
from mjlab.utils.cpu_runtime import (
  apply_cpu_execution_plan,
  get_worker_io_cpu_ids,
  partition_server_cpus,
)

_STANDARD_PPO_CLASS_NAMES = frozenset(
  {
    "PPO",
    "rsl_rl.algorithms:PPO",
    "rsl_rl.algorithms.ppo:PPO",
    "rsl_rl.algorithms.ppo.PPO",
  }
)


def _is_plain_tb_writer(writer: object) -> bool:
  """Return True only for pure TensorBoard SummaryWriter instances."""
  if not isinstance(writer, SummaryWriter) or isinstance(writer, BatchedAsyncLogWriter):
    return False
  name = type(writer).__name__
  return (
    "Wandb" not in name and "Neptune" not in name and not hasattr(writer, "save_model")
  )


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)

    self._strict_rng_parity: bool = bool(train_cfg.get("strict_rng_parity", False))
    dev_type = torch.device(device).type
    use_zero_sync_ppo = bool(
      train_cfg.get("zero_sync_ppo", True)
      and not self._strict_rng_parity
      and dev_type == "cuda"
      and torch.cuda.is_available()
    )
    alg_cfg = train_cfg.get("algorithm")
    if (
      use_zero_sync_ppo
      and isinstance(alg_cfg, dict)
      and alg_cfg.get("class_name") in _STANDARD_PPO_CLASS_NAMES
    ):
      alg_cfg["class_name"] = "mjlab.rl.zero_sync_ppo:ZeroSyncPPO"

    super().__init__(env, train_cfg, log_dir, device)
    self._async_io_worker: AsyncCheckpointWorker | None = None
    self._cpu_execution_plan = None
    self._batched_logging_enabled: bool = False
    self._zero_sync_rollout: bool = bool(
      train_cfg.get("zero_sync_rollout", True)
      and not self._strict_rng_parity
      and dev_type == "cuda"
      and torch.cuda.is_available()
    )
    self._mega_graph_mode: str = str(
      train_cfg.get(
        "mega_graph_mode",
        getattr(getattr(env, "unwrapped", env), "cfg", None)
        and getattr(getattr(env, "unwrapped", env).cfg, "_mega_graph_mode", "off")
        or "off",
      )
    )
    self._rollout_mega_graph = None

  def _install_batched_logger_hooks(self, io_cpu_ids: tuple[int, ...]) -> bool:
    """Wrap TensorBoard SummaryWriter with single-Event protobuf coalescing."""
    logger = getattr(self, "logger", None)
    if logger is not None and not getattr(logger, "_mlab4_batched_hooked", False):
      orig_init = logger.init_logging_writer
      orig_log = logger.log
      orig_stop = getattr(logger, "stop_logging_writer", None)

      def _wrapped_init() -> None:
        orig_init()
        w = getattr(logger, "writer", None)
        if _is_plain_tb_writer(w):
          logger.writer = BatchedAsyncLogWriter(w, worker_cpu_ids=io_cpu_ids)

      def _wrapped_log(*args, **kwargs) -> None:
        self._drain_async_model_uploads()
        w = getattr(logger, "writer", None)
        if isinstance(w, BatchedAsyncLogWriter):
          with w.step_batch():
            orig_log(*args, **kwargs)
        else:
          orig_log(*args, **kwargs)

      def _wrapped_stop() -> None:
        self.flush_async_io()
        logger._mlab4_writer_stopped = True
        if orig_stop is not None:
          orig_stop()

      logger.init_logging_writer = _wrapped_init
      logger.log = _wrapped_log
      if orig_stop is not None:
        logger.stop_logging_writer = _wrapped_stop
      logger._mlab4_batched_hooked = True
      w_now = getattr(logger, "writer", None)
      if _is_plain_tb_writer(w_now):
        logger.writer = BatchedAsyncLogWriter(w_now, worker_cpu_ids=io_cpu_ids)
      return True

    writer = getattr(self, "writer", None)
    if _is_plain_tb_writer(writer):
      self.writer = BatchedAsyncLogWriter(writer, worker_cpu_ids=io_cpu_ids)
      return True
    return isinstance(writer, BatchedAsyncLogWriter)

  def _drain_async_model_uploads(self) -> None:
    logger = getattr(self, "logger", None)
    if (
      self._async_io_worker is None
      or not self.cfg.get("upload_model", False)
      or (logger is not None and getattr(logger, "_mlab4_writer_stopped", False))
    ):
      return
    saves, _ = self._async_io_worker.pop_completed_artifacts()
    for ckpt_path, it in saves:
      if ckpt_path:
        self.logger.save_model(str(ckpt_path), int(it))

  def enable_cpu_optimizations(
    self,
    *,
    async_checkpoint: bool = True,
    confinement: bool = True,
    driver_physical_cores: int | None = None,
    batched_logging: bool = True,
  ) -> dict:
    """Enable NUMA/CCD/P-core CPU isolation, save_v2 async checkpointing, and batched logging."""
    plan = partition_server_cpus(
      world_size=getattr(self, "gpu_world_size", 1),
      rank=getattr(self, "gpu_global_rank", 0),
      local_rank=getattr(self, "gpu_local_rank", 0),
      local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
      device=self.device,
      driver_physical_cores=driver_physical_cores,
    )
    self._cpu_execution_plan = plan
    pinned = 0
    if confinement:
      conf_info = apply_cpu_execution_plan(plan, primary_driver_only=True)
      raw_pinned = conf_info.get("pinned_existing_threads", 0)
      pinned = raw_pinned if isinstance(raw_pinned, int) else 0
    if async_checkpoint and self._async_io_worker is None:
      self._async_io_worker = consume_prewarmed_async_worker(
        worker_cpu_ids=plan.worker_io_cpu_ids
      ) or AsyncCheckpointWorker(worker_cpu_ids=plan.worker_io_cpu_ids)
    io_cpus = plan.worker_io_cpu_ids or get_worker_io_cpu_ids()
    if batched_logging:
      self._batched_logging_enabled = self._install_batched_logger_hooks(io_cpus)
    return {
      **plan.to_dict(),
      "pinned_existing_threads": pinned,
      "async_checkpoint": self._async_io_worker is not None,
      "batched_async_logging": self._batched_logging_enabled,
    }

  def _active_writers(self) -> list:
    out = []
    w1 = getattr(self, "writer", None)
    if w1 is not None:
      out.append(w1)
    logger = getattr(self, "logger", None)
    w2 = getattr(logger, "writer", None) if logger is not None else None
    if w2 is not None and w2 is not w1:
      out.append(w2)
    return out

  def flush_async_io(self) -> None:
    if self._async_io_worker is not None:
      self._async_io_worker.flush()
      self._drain_async_model_uploads()
    for w in self._active_writers():
      if hasattr(w, "flush"):
        w.flush()

  def close_cpu_workers(self) -> None:
    err = None
    if self._async_io_worker is not None:
      try:
        self._async_io_worker.flush()
        self._drain_async_model_uploads()
      except Exception as exc:
        err = exc
    for w in self._active_writers():
      if hasattr(w, "close"):
        try:
          w.close()
        except Exception as exc:
          if err is None:
            err = exc
    if self._async_io_worker is not None:
      try:
        self._async_io_worker.close()
      except Exception as exc:
        if err is None:
          err = exc
      finally:
        self._async_io_worker = None
    if err is not None:
      raise err

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Zero-sync rollout + learning loop eliminating per-step check_nan and logger.nonzero() syncs."""
    if not self._zero_sync_rollout:
      return super().learn(
        num_learning_iterations=num_learning_iterations,
        init_at_random_ep_len=init_at_random_ep_len,
      )

    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    num_steps = int(self.cfg["num_steps_per_env"])
    num_envs = int(self.env.num_envs)
    has_rnd = bool(self.cfg["algorithm"].get("rnd_cfg"))
    check_nan_enabled = bool(self.cfg.get("check_for_nan", True))
    unwrapped = getattr(self.env, "unwrapped", self.env)

    # Preallocate zero-sync per-step rollout staging buffers on GPU.
    step_done_buf = torch.empty(
      (num_steps, num_envs), dtype=torch.bool, device=self.device
    )
    step_rew_buf = torch.empty(
      (num_steps, num_envs), dtype=torch.float32, device=self.device
    )
    step_len_buf = torch.empty(
      (num_steps, num_envs), dtype=torch.float32, device=self.device
    )
    step_erew_buf = (
      torch.empty((num_steps, num_envs), dtype=torch.float32, device=self.device)
      if has_rnd
      else None
    )
    step_irew_buf = (
      torch.empty((num_steps, num_envs), dtype=torch.float32, device=self.device)
      if has_rnd
      else None
    )
    rollout_nonfinite = torch.zeros((), dtype=torch.bool, device=self.device)

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations

    # Optional Tier 2 RolloutStepMegaGraph (fuses Actor/Critic + EnvFullStepGraph).
    rollout_mega_state = None
    env_graph = getattr(unwrapped, "_mega_step_graph", None)
    can_tier2 = (
      self._mega_graph_mode == "rollout"
      and env_graph is not None
      and getattr(self.env, "env", None) is unwrapped
      and not has_rnd
      and not getattr(self.alg, "symmetry", None)
      and not getattr(self.alg.actor, "is_recurrent", False)
      and not getattr(self.alg.critic, "is_recurrent", False)
      and type(getattr(self.alg.actor, "distribution", None)).__name__
      == "GaussianDistribution"
      and hasattr(self.alg.actor, "get_latent")
      and hasattr(self.alg.actor, "mlp")
      and torch.device(self.device) == torch.device(unwrapped.device)
    )
    if can_tier2:
      from tensordict import TensorDict

      from mjlab.envs.mega_step_graph import RolloutStepMegaGraph

      # Bind static obs buffers from current env.obs_buf without advancing history or noise.
      static_obs_dict = unwrapped.observation_manager.init_static_obs_buffer(
        unwrapped.obs_buf
      )
      obs = TensorDict(static_obs_dict, batch_size=[num_envs])
      actor = self.alg.actor
      critic = self.alg.critic
      clip_act = getattr(self.env, "clip_actions", None)
      is_finite_horizon = bool(getattr(unwrapped.cfg, "is_finite_horizon", False))

      # Probe distribution shapes once cleanly (use no_grad rather than inference_mode
      # so staging tensors remain normal non-inference tensors).
      with torch.no_grad():
        _lat0 = actor.get_latent(obs, None, None)
        _mlp0 = actor.mlp(_lat0)
        actor.distribution.update(_mlp0)
        _dparams0 = tuple(p.detach().clone() for p in actor.output_distribution_params)
        _val0 = critic(obs).detach().clone()
        _act0 = torch.empty_like(actor.output_mean)
        _lp0 = actor.get_output_log_prob(_act0).detach().clone()

      stg_pre_obs = obs.clone()
      stg_actions = torch.empty_like(_act0)
      stg_log_prob = torch.empty_like(_lp0)
      stg_values = torch.empty_like(_val0)
      stg_dist_params = tuple(torch.empty_like(p) for p in _dparams0)
      stg_rewards = torch.empty(num_envs, dtype=torch.float32, device=self.device)
      stg_dones_bool = torch.empty(num_envs, dtype=torch.bool, device=self.device)
      stg_dones_long = torch.empty(num_envs, dtype=torch.long, device=self.device)
      stg_ep_rew = torch.empty(num_envs, dtype=torch.float32, device=self.device)
      stg_ep_len = torch.empty(num_envs, dtype=torch.float32, device=self.device)
      gamma = float(self.alg.gamma)

      def _tier2_step_body() -> None:
        # 1. Snapshot pre-step obs before EnvFullStepGraph mutates static_obs_dict in-place.
        for k in stg_pre_obs.keys():
          stg_pre_obs[k].copy_(obs[k])
        # 2. Actor + Critic forward & CUDA-Graph-safe Philox sampling.
        lat = actor.get_latent(obs, None, None)
        mlp_out = actor.mlp(lat)
        actor.distribution.update(mlp_out)
        act_sample = (
          actor.output_mean + torch.randn_like(actor.output_mean) * actor.output_std
        )
        stg_actions.copy_(act_sample)
        stg_log_prob.copy_(actor.get_output_log_prob(stg_actions))
        stg_values.copy_(critic(obs))
        for dst_p, src_p in zip(
          stg_dist_params, actor.output_distribution_params, strict=True
        ):
          dst_p.copy_(src_p)
        # 3. Feed action into EnvFullStepGraph and run full physics + masked reset + obs.
        if clip_act is not None:
          torch.clamp(stg_actions, -clip_act, clip_act, out=env_graph._graph_action_in)
        else:
          env_graph._graph_action_in.copy_(stg_actions)
        env_graph._body()
        # 4. Compute dones, bootstrapped rewards, nonfinite latch, and episode sums.
        torch.logical_or(
          unwrapped.reset_terminated, unwrapped.reset_time_outs, out=stg_dones_bool
        )
        stg_dones_long.copy_(stg_dones_bool)
        stg_rewards.copy_(unwrapped.reward_buf)
        if not is_finite_horizon:
          stg_rewards.add_(
            stg_values.squeeze(-1) * unwrapped.reset_time_outs.to(dtype=torch.float32),
            alpha=gamma,
          )
        if check_nan_enabled:
          bad = ~(
            torch.isfinite(unwrapped.reward_buf).all()
            & torch.isfinite(stg_dones_long).all()
          )
          for v in obs.values():
            if torch.is_tensor(v):
              bad = bad | (~torch.isfinite(v).all())
          rollout_nonfinite.logical_or_(bad)
        if self.logger.writer is not None:
          self.logger.cur_reward_sum.add_(unwrapped.reward_buf)
          self.logger.cur_episode_length.add_(1.0)
          stg_ep_rew.copy_(self.logger.cur_reward_sum)
          stg_ep_len.copy_(self.logger.cur_episode_length)
          self.logger.cur_reward_sum.masked_fill_(stg_dones_bool, 0.0)
          self.logger.cur_episode_length.masked_fill_(stg_dones_bool, 0.0)

      tier2_graph = RolloutStepMegaGraph(
        env_graph=env_graph,
        modules=(
          getattr(self.alg, "_raw_actor", self.alg.actor),
          getattr(self.alg, "_raw_critic", self.alg.critic),
        ),
        step_body_fn=_tier2_step_body,
      )
      self._rollout_mega_graph = tier2_graph
      rollout_mega_state = (
        tier2_graph,
        stg_pre_obs,
        stg_actions,
        stg_log_prob,
        stg_values,
        stg_dist_params,
        stg_rewards,
        stg_dones_bool,
        stg_dones_long,
        stg_ep_rew,
        stg_ep_len,
      )

    for it in range(start_it, total_it):
      start = time.time()
      rollout_nonfinite.zero_()
      if hasattr(unwrapped, "begin_rollout_iteration"):
        unwrapped.begin_rollout_iteration(num_steps)

      with torch.inference_mode():
        if rollout_mega_state is not None:
          (
            tier2_graph,
            stg_pre_obs,
            stg_actions,
            stg_log_prob,
            stg_values,
            stg_dist_params,
            stg_rewards,
            stg_dones_bool,
            stg_dones_long,
            stg_ep_rew,
            stg_ep_len,
          ) = rollout_mega_state
          tier2_graph.verify_or_invalidate()
          st = self.alg.storage
          for step_idx in range(num_steps):
            tier2_graph.run_step()
            # Copy static staging outputs into RolloutStorage[step_idx] (0 host syncs).
            self.alg.transition.hidden_states = (None, None)
            self.alg.transition.observations = stg_pre_obs
            self.alg.transition.actions = stg_actions
            self.alg.transition.values = stg_values
            self.alg.transition.actions_log_prob = stg_log_prob
            self.alg.transition.distribution_params = stg_dist_params
            self.alg.transition.rewards = stg_rewards
            self.alg.transition.dones = stg_dones_long
            st.add_transition(self.alg.transition)
            self.alg.transition.clear()
            if self.logger.writer is not None:
              step_done_buf[step_idx].copy_(stg_dones_bool)
              step_rew_buf[step_idx].copy_(stg_ep_rew)
              step_len_buf[step_idx].copy_(stg_ep_len)
        else:
          for step_idx in range(num_steps):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            obs, rewards, dones = (
              obs.to(self.device),
              rewards.to(self.device),
              dones.to(self.device),
            )
            if check_nan_enabled:
              bad = ~(torch.isfinite(rewards).all() & torch.isfinite(dones).all())
              for v in obs.values():
                if torch.is_tensor(v):
                  bad = bad | (~torch.isfinite(v).all())
              rollout_nonfinite.logical_or_(bad)

            self.alg.process_env_step(obs, rewards, dones, extras)
            intrinsic_rewards = self.alg.intrinsic_rewards if has_rnd else None

            if self.logger.writer is not None:
              if not getattr(unwrapped, "_staged_episode_logs_active", False):
                if "episode" in extras:
                  self.logger.ep_extras.append(extras["episode"])
                elif "log" in extras:
                  self.logger.ep_extras.append(extras["log"])
              if has_rnd and intrinsic_rewards is not None:
                self.logger.cur_ereward_sum.add_(rewards)
                self.logger.cur_ireward_sum.add_(intrinsic_rewards)
                self.logger.cur_reward_sum.add_(rewards + intrinsic_rewards)
              else:
                self.logger.cur_reward_sum.add_(rewards)
              self.logger.cur_episode_length.add_(1.0)
              done_mask = dones.reshape(-1) > 0
              step_done_buf[step_idx].copy_(done_mask)
              step_rew_buf[step_idx].copy_(self.logger.cur_reward_sum)
              step_len_buf[step_idx].copy_(self.logger.cur_episode_length)
              self.logger.cur_reward_sum.masked_fill_(done_mask, 0.0)
              self.logger.cur_episode_length.masked_fill_(done_mask, 0.0)
              if has_rnd and step_erew_buf is not None and step_irew_buf is not None:
                step_erew_buf[step_idx].copy_(self.logger.cur_ereward_sum)
                step_irew_buf[step_idx].copy_(self.logger.cur_ireward_sum)
                self.logger.cur_ereward_sum.masked_fill_(done_mask, 0.0)
                self.logger.cur_ireward_sum.masked_fill_(done_mask, 0.0)

        # Single rollout-end flush: nonfinite check + completed episode rewards/lengths + staged logs.
        if hasattr(unwrapped, "end_rollout_iteration"):
          flushed_infos = unwrapped.end_rollout_iteration()
          if self.logger.writer is not None and flushed_infos:
            self.logger.ep_extras.extend(flushed_infos)

        if check_nan_enabled and bool(rollout_nonfinite.item()):
          raise ValueError(
            f"NaN/Inf detected during rollout at iteration {it} (deferred zero-sync guard)."
          )

        if self.logger.writer is not None:
          max_ep = self.logger.rewbuffer.maxlen
          if has_rnd and step_erew_buf is not None and step_irew_buf is not None:
            packed_t = torch.stack(
              (step_rew_buf, step_len_buf, step_erew_buf, step_irew_buf), dim=0
            )[:, step_done_buf]
            if max_ep is not None and packed_t.shape[1] > max_ep:
              packed_t = packed_t[:, -max_ep:]
            if packed_t.shape[1] > 0:
              packed = packed_t.cpu().tolist()
              self.logger.rewbuffer.extend(packed[0])
              self.logger.lenbuffer.extend(packed[1])
              self.logger.erewbuffer.extend(packed[2])
              self.logger.irewbuffer.extend(packed[3])
          else:
            packed_t = torch.stack((step_rew_buf, step_len_buf), dim=0)[
              :, step_done_buf
            ]
            if max_ep is not None and packed_t.shape[1] > max_ep:
              packed_t = packed_t[:, -max_ep:]
            if packed_t.shape[1] > 0:
              packed = packed_t.cpu().tolist()
              self.logger.rewbuffer.extend(packed[0])
              self.logger.lenbuffer.extend(packed[1])

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if has_rnd else None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    if self._async_io_worker is not None:
      self._drain_async_model_uploads()
      self._async_io_worker.submit_save(
        saved_dict,
        path,
        export_onnx=False,
      )
      self._drain_async_model_uploads()
    else:
      torch.save(saved_dict, path)
      if self.cfg["upload_model"]:
        self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
