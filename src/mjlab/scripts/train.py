"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from mjlab._backend_bootstrap import (
  MujocoBackendChoice,
  format_physics_backend_banner,
  get_physics_backend_info,
  normalize_mj4_shorthand_argv,
)
from mjlab.utils.cpu_runtime import (
  apply_cpu_execution_plan,
  bootstrap_cpu_env_before_torch,
  partition_server_cpus,
)
from mjlab.utils.rollout_optimizations import (
  configure_fp32_exact,
  configure_fp32_fast,
  resolve_fp32_execution_mode,
)


def _argv_requests_multi_gpu(argv: Sequence[str]) -> bool:
  """Stdlib-only pre-parse check to avoid pinning a multi-GPU torchrunx parent launcher."""
  if int(os.environ.get("WORLD_SIZE", "1") or "1") > 1:
    return False
  tokens = list(argv)
  for idx, tok in enumerate(tokens):
    if tok == "--gpu-ids":
      vals: list[str] = []
      j = idx + 1
      while j < len(tokens) and not tokens[j].startswith("-"):
        vals.append(tokens[j])
        j += 1
      if len(vals) > 1 or (len(vals) == 1 and vals[0].lower() == "all"):
        return True
    elif tok.startswith("--gpu-ids="):
      val = tok.split("=", 1)[1].strip()
      if (
        val.lower() == "all" or len([x for x in val.replace(",", " ").split() if x]) > 1
      ):
        return True
  return False


def _argv_disables_cpu_optimizations(argv: Sequence[str]) -> bool:
  """Stdlib-only check for ``--no-cpu-optimizations`` or ``--cpu-optimizations False``."""
  tokens = list(argv)
  for idx, tok in enumerate(tokens):
    low = tok.lower()
    if low in (
      "--no-cpu-optimizations",
      "--cpu-optimizations=false",
      "--cpu-optimizations=0",
    ):
      return True
    if low == "--cpu-optimizations" and idx + 1 < len(tokens):
      if tokens[idx + 1].lower() in ("false", "0", "no"):
        return True
  return False


# Run pre-import CPU/OpenMP bootstrap BEFORE importing torch-dependent modules.
if "torch" not in sys.modules and not _argv_disables_cpu_optimizations(sys.argv[1:]):
  bootstrap_cpu_env_before_torch(
    is_multi_gpu_launcher=_argv_requests_multi_gpu(sys.argv[1:])
  )

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.rl import (  # noqa: E402
  MjlabOnPolicyRunner,
  RslRlBaseRunnerCfg,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.registry import load_runner_cls  # noqa: E402
from mjlab.tasks.tracking.mdp import MotionCommandCfg  # noqa: E402
from mjlab.utils.async_io import prewarm_async_checkpoint_worker  # noqa: E402
from mjlab.utils.os import (  # noqa: E402
  dump_yaml,
  get_checkpoint_path,
  get_wandb_checkpoint_path,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wandb import add_wandb_tags  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402


def _ensure_runtime_types() -> None:
  """Compatibility no-op (types already bound after pre-import CPU bootstrap)."""
  return None


def _ensure_runtime_helpers() -> None:
  """Compatibility no-op (helpers already bound after pre-import CPU bootstrap)."""
  return None


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  mujoco_backend: MujocoBackendChoice = "4.0"
  """Physics backend selector (defaults to '4.0' in MLAB4_Fast; 'auto', '3.11', '4.0-git' also supported)."""
  fp32_exact: bool = True
  """Lossless IEEE 754 FP32 acceleration (#2A CPU/OpenMP + #2B ZeroSyncPPO + #1A Tier 1 EnvFullStepGraph + AsyncResetPool)."""
  fp32_fast: bool = False
  """Pure 32-bit IEEE 754 FP32 max-throughput mode (adds #1B Tier 2 RolloutStepMegaGraph fusing Actor/Critic + Env)."""
  strict_rng_parity: bool = False
  """Disable async reset pool, mega CUDA graph, and ZeroSyncPPO for legacy bitwise PRNG regression tests."""
  mega_graph_mode: Literal["auto", "off", "env", "rollout"] = "auto"
  """Full-step CUDA graph tier: 'auto' (env for fp32_exact, rollout for fp32_fast), 'off', 'env', or 'rollout'."""
  reset_pool_mode: Literal["off", "async_ring"] = "async_ring"
  """Background CPU reset-state ring pool mode under fp32_exact / fp32_fast."""
  cpu_optimizations: bool = True
  """NUMA/P-core CPU isolation + save_v2 async checkpointing + batched logging."""
  cpu_driver_cores: int | None = None
  """Optional explicit count of physical cores reserved for the CUDA/Warp driver thread."""

  @staticmethod
  def from_task(task_id: str) -> TrainConfig:
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  if cfg.cpu_optimizations:
    bootstrap_cpu_env_before_torch(
      driver_physical_cores=cfg.cpu_driver_cores,
      is_multi_gpu_launcher=False,
    )

  _ensure_runtime_helpers()

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + rank

  resolved = resolve_fp32_execution_mode(
    fp32_exact=cfg.fp32_exact,
    fp32_fast=cfg.fp32_fast,
    strict_rng_parity=cfg.strict_rng_parity,
    mega_graph_mode=cfg.mega_graph_mode,
    reset_pool_mode=cfg.reset_pool_mode,
  )
  configure_torch_backends(allow_tf32=resolved["allow_tf32"])
  if resolved["ieee_enabled"]:
    if resolved["fp32_fast"]:
      configure_fp32_fast(
        cfg.env,
        enabled=True,
        strict_rng_parity=resolved["strict_rng_parity"],
        mega_graph_mode=resolved["mega_graph_mode"],
        reset_pool_mode=resolved["reset_pool_mode"],
      )
    else:
      configure_fp32_exact(cfg.env, enabled=True)
      cfg.env._strict_rng_parity = resolved["strict_rng_parity"]  # type: ignore[attr-defined]
      cfg.env._mega_graph_mode = resolved["mega_graph_mode"]  # type: ignore[attr-defined]
      cfg.env._reset_pool_mode = resolved["reset_pool_mode"]  # type: ignore[attr-defined]

  if cfg.cpu_optimizations:
    plan = partition_server_cpus(
      world_size=int(os.environ.get("WORLD_SIZE", "1")),
      rank=rank,
      local_rank=int(os.environ.get("LOCAL_RANK", "0")),
      local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
      device=device,
      driver_physical_cores=cfg.cpu_driver_cores,
    )
    try:
      apply_cpu_execution_plan(plan, primary_driver_only=True)
    except TypeError:
      apply_cpu_execution_plan(plan)
    if rank == 0:
      prewarm_async_checkpoint_worker(worker_cpu_ids=plan.worker_io_cpu_ids)

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
  if rank == 0:
    print(format_physics_backend_banner(cfg.mujoco_backend))

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if resolved["ieee_enabled"] and hasattr(env, "enable_fp32_step_optimizations"):
    opt_rep = env.enable_fp32_step_optimizations()
    if rank == 0:
      print(
        f"[INFO] {resolved['mode']} step optimizations: {opt_rep} "
        f"(mega_graph_mode={resolved['mega_graph_mode']}, "
        f"reset_pool_mode={resolved['reset_pool_mode']}, "
        f"strict_rng_parity={resolved['strict_rng_parity']})"
      )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)
  agent_cfg["fp32_mode"] = resolved["mode"]
  agent_cfg["fp32_fast"] = resolved["fp32_fast"]
  agent_cfg["strict_rng_parity"] = resolved["strict_rng_parity"]
  agent_cfg["zero_sync_ppo"] = resolved["zero_sync_ppo"]
  agent_cfg["zero_sync_rollout"] = resolved["zero_sync_rollout"]
  agent_cfg["mega_graph_mode"] = resolved["mega_graph_mode"]

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    backend_info = get_physics_backend_info(cfg.mujoco_backend)
    (log_dir / "params" / "physics_backend.json").write_text(
      json.dumps(backend_info, indent=2) + "\n"
    )

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)
  if cfg.cpu_optimizations and hasattr(runner, "enable_cpu_optimizations"):
    runner.enable_cpu_optimizations(
      async_checkpoint=True,
      confinement=True,
      driver_physical_cores=cfg.cpu_driver_cores,
    )

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  try:
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
  finally:
    if hasattr(runner, "close_cpu_workers"):
      runner.close_cpu_workers()
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  from mjlab.utils.gpu import select_gpus

  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
      + (
        "MUJOCO*",
        "MJLAB_*",
        "OMP_*",
        "GOMP_*",
        "MKL_*",
        "OPENBLAS_*",
        "NUMEXPR_*",
      ),
    ).run(run_train, task_id, args, log_dir)


def main():
  import tyro

  import mjlab
  from mjlab.scripts._cli import maybe_print_top_level_help

  maybe_print_top_level_help("train")

  # Pre-import CPU/OpenMP bootstrap BEFORE importing torch / mjlab.tasks.
  if not _argv_disables_cpu_optimizations(sys.argv[1:]):
    bootstrap_cpu_env_before_torch(
      is_multi_gpu_launcher=_argv_requests_multi_gpu(sys.argv[1:])
    )
  _ensure_runtime_types()

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    args=normalize_mj4_shorthand_argv(),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
