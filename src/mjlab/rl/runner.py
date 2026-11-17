import os
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
    super().__init__(env, train_cfg, log_dir, device)
    self._async_io_worker: AsyncCheckpointWorker | None = None
    self._cpu_execution_plan = None
    self._batched_logging_enabled: bool = False

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
      conf_info = apply_cpu_execution_plan(plan)
      pinned = int(conf_info.get("pinned_existing_threads", 0))
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
