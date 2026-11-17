"""Smoke and unit tests for MLAB4_Fast (MuJoCo 4.0 + permanent fp32_exact + CPU/AsyncIO)."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import torch

from mjlab.scripts.train import TrainConfig
from mjlab.utils import cpu_runtime as cr
from mjlab.utils.async_io import (
  AsyncBatchedScalarWriter,
  AsyncCheckpointWorker,
  BatchedAsyncLogWriter,
  consume_prewarmed_async_worker,
  prewarm_async_checkpoint_worker,
)
from mjlab.utils.rollout_optimizations import FP32_EXACT_ENABLED, configure_fp32_exact


def test_mlab4_fast_defaults_permanently_enabled():
  assert FP32_EXACT_ENABLED is True
  cfg = TrainConfig(env=SimpleNamespace(), agent=SimpleNamespace())
  assert cfg.mujoco_backend == "4.0"
  assert cfg.fp32_exact is True
  assert cfg.cpu_optimizations is True


def test_configure_fp32_exact_sets_flags():
  cmd = SimpleNamespace(batch_reset_metrics=False, cache_reference_kernels=False)
  env_cfg = SimpleNamespace(commands={"motion": cmd})
  res = configure_fp32_exact(env_cfg, enabled=True)
  assert res["fp32_exact"] is True
  assert env_cfg._rollout_optimization_mode == "fp32_exact"
  assert env_cfg._use_substep_cuda_graph is True
  assert env_cfg._use_zero_sync_optimizations is True
  assert cmd.batch_reset_metrics is True
  assert cmd.cache_reference_kernels is True


def test_cpu_runtime_pcore_filtering():
  hybrid = cr.CpuTopology(
    allowed_cpus=tuple(range(9)),
    physical_groups=(
      cr.CpuCoreGroup(
        package_id=0, core_id=0, numa_node=0, max_freq_khz=5200000, cpu_ids=(0, 1)
      ),
      cr.CpuCoreGroup(
        package_id=0, core_id=1, numa_node=0, max_freq_khz=5200000, cpu_ids=(2, 3)
      ),
      cr.CpuCoreGroup(
        package_id=0, core_id=2, numa_node=0, max_freq_khz=3800000, cpu_ids=(4,)
      ),
      cr.CpuCoreGroup(
        package_id=0, core_id=3, numa_node=0, max_freq_khz=3800000, cpu_ids=(5,)
      ),
    ),
    numa_nodes=(0,),
  )
  assert cr.get_performance_cpu_ids(
    (0, 1, 2, 3, 4, 5), one_per_core=True, topology=hybrid
  ) == (0, 2)


def test_cpu_runtime_multi_ccd_isolation():
  ccd_groups = tuple(
    cr.CpuCoreGroup(
      package_id=0,
      core_id=c,
      numa_node=0,
      llc_id=c // 6,
      max_freq_khz=5100000,
      cpu_ids=(c, c + 24),
    )
    for c in range(24)
  )
  topo = cr.CpuTopology(
    allowed_cpus=tuple(range(48)),
    physical_groups=ccd_groups,
    numa_nodes=(0,),
    llc_ids=(0, 1, 2, 3),
  )
  plan = cr.partition_server_cpus(
    world_size=1, rank=0, topology=topo, driver_physical_cores=2
  )
  assert plan.driver_cpu_ids == (0, 1, 24, 25)
  assert set(plan.driver_cpu_ids).isdisjoint(plan.worker_cpu_ids)
  # Non-driver CCDs (CCD 1..3 = cores 6..23) are prioritized ahead of CCD 0 siblings.
  assert plan.worker_compute_cpu_ids[:18] == tuple(range(6, 24))


def test_prewarm_and_single_event_step_batch(tmp_path: Path):
  from torch.utils.tensorboard import SummaryWriter

  prewarm_async_checkpoint_worker()
  worker = consume_prewarmed_async_worker()
  assert worker is not None
  try:
    worker.submit_save(
      {"iter": 7, "w": torch.ones((16, 16))},
      tmp_path / "prewarm_7.pt",
    )
    worker.flush()
    saves, _ = worker.pop_completed_artifacts()
    assert any(it == 7 and ckpt.endswith("prewarm_7.pt") for ckpt, it in saves)
  finally:
    worker.close()

  tb_dir = tmp_path / "tb_coalesced"
  sw = SummaryWriter(log_dir=str(tb_dir))
  bw = BatchedAsyncLogWriter(sw)
  try:
    with bw.step_batch():
      for k in range(12):
        bw.add_scalar(f"Loss/m_{k}", float(k), 1)
    bw.flush()
  finally:
    bw.close()
  event_files = list(tb_dir.glob("events.out.tfevents.*"))
  assert len(event_files) == 1 and event_files[0].stat().st_size > 0


def test_async_checkpoint_ping_pong_and_batched_writer(tmp_path: Path):
  worker = AsyncCheckpointWorker()
  try:
    t0 = time.perf_counter()
    for i in range(3):
      worker.submit_save(
        {"iter": i, "w": torch.full((32, 32), float(i + 1))}, tmp_path / f"m_{i}.pt"
      )
      if i == 0:
        first_ms = (time.perf_counter() - t0) * 1000.0
    worker.flush()
  finally:
    worker.close()

  assert first_ms < 500.0
  for i in range(3):
    loaded = torch.load(tmp_path / f"m_{i}.pt", map_location="cpu", weights_only=True)
    assert loaded["iter"] == i

  recorded: list[tuple[str, float, int]] = []

  class _FakeWriter:
    def add_scalar(self, tag: str, value: float, step: int) -> None:
      recorded.append((tag, float(value), int(step)))

    def flush(self) -> None:
      pass

    def close(self) -> None:
      pass

  w = AsyncBatchedScalarWriter(_FakeWriter(), batch_size=4, flush_interval_s=10.0)
  try:
    for i in range(5):
      w.add_scalar("Train/loss", float(i), i)
    w.flush()
    assert len(recorded) == 5
  finally:
    w.close()


def test_is_plain_tb_writer_skips_wandb_and_flushes_before_stop(tmp_path: Path):
  from torch.utils.tensorboard import SummaryWriter

  from mjlab.rl.runner import MjlabOnPolicyRunner, _is_plain_tb_writer

  class _FakeWandbWriter(SummaryWriter):
    def save_model(self, model_path: str, iter: int) -> None:
      pass

    def stop(self) -> None:
      pass

  tb = SummaryWriter(log_dir=str(tmp_path / "tb"))
  wb = _FakeWandbWriter(log_dir=str(tmp_path / "wb"))
  try:
    assert _is_plain_tb_writer(tb) is True
    assert _is_plain_tb_writer(wb) is False
  finally:
    tb.close()
    wb.close()

  events: list[str] = []

  class _FakeLogger:
    def __init__(self, writer):
      self.writer = writer

    def init_logging_writer(self) -> None:
      pass

    def log(self, *args, **kwargs) -> None:
      pass

    def stop_logging_writer(self) -> None:
      events.append("stop")

    def save_model(self, path: str, it: int) -> None:
      events.append(f"upload:{it}")

  dummy_runner = SimpleNamespace(
    logger=_FakeLogger(wb),
    cfg={"upload_model": True},
    _async_io_worker=SimpleNamespace(
      flush=lambda: events.append("flush"),
      pop_completed_artifacts=lambda: ([("/tmp/m_1.pt", 1)], []),
    ),
  )
  dummy_runner._active_writers = lambda: []
  dummy_runner._drain_async_model_uploads = (
    lambda: MjlabOnPolicyRunner._drain_async_model_uploads(dummy_runner)  # type: ignore[arg-type]
  )
  dummy_runner.flush_async_io = lambda: MjlabOnPolicyRunner.flush_async_io(
    dummy_runner  # type: ignore[arg-type]
  )
  assert (
    MjlabOnPolicyRunner._install_batched_logger_hooks(
      dummy_runner,  # type: ignore[arg-type]
      (),
    )
    is True
  )
  assert dummy_runner.logger.writer is wb

  dummy_runner.logger.stop_logging_writer()
  assert events == ["flush", "upload:1", "stop"]
  assert getattr(dummy_runner.logger, "_mlab4_writer_stopped", False) is True
  # Late drain after stop_logging_writer must not call save_model again.
  dummy_runner._drain_async_model_uploads()
  assert events == ["flush", "upload:1", "stop"]


def test_matches_requested_backend_distinct_symlinked_venvs(tmp_path: Path):
  import sys

  from mjlab._backend_bootstrap import _matches_requested_backend

  git_bin = tmp_path / ".venv-mj4-git" / "bin"
  git_bin.mkdir(parents=True)
  git_py = git_bin / "python"
  git_py.symlink_to(sys.executable)

  # Even when git_py points to the same underlying python binary, a different
  # venv bin directory must trigger re-exec rather than returning True.
  assert (
    _matches_requested_backend(
      "4.0-git",
      "4.0",
      environ={"MJLAB_MUJOCO4_GIT_PYTHON": str(git_py)},
    )
    is False
  )


def test_run_train_enables_step_opts_and_respects_cpu_opt_false(
  monkeypatch, tmp_path: Path
):
  from dataclasses import dataclass, field

  import mjlab.scripts.train as train_mod

  calls: dict[str, object] = {
    "cpu_plan": 0,
    "prewarm": 0,
    "step_opts": 0,
    "runner_cpu_opts": 0,
  }

  @dataclass
  class _FakeEnvCfg:
    seed: int = 42
    commands: dict = field(default_factory=dict)

  @dataclass
  class _FakeAgentCfg:
    seed: int = 42
    resume: bool = False
    clip_actions: float | None = None
    wandb_tags: tuple = ()
    max_iterations: int = 1

  class _FakeEnv:
    def __init__(self, cfg, device, render_mode=None):
      pass

    def enable_fp32_step_optimizations(self):
      calls["step_opts"] = int(calls["step_opts"]) + 1
      return {}

    def close(self):
      pass

  class _FakeRunner:
    def __init__(self, env, agent_cfg, log_dir, device, **kwargs):
      pass

    def enable_cpu_optimizations(self, **kwargs):
      calls["runner_cpu_opts"] = int(calls["runner_cpu_opts"]) + 1

    def add_git_repo_to_log(self, path):
      pass

    def learn(self, **kwargs):
      pass

    def close_cpu_workers(self):
      pass

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train_mod, "configure_torch_backends", lambda **kw: None)
  monkeypatch.setattr(train_mod, "configure_fp32_exact", lambda cfg, enabled=True: {})
  monkeypatch.setattr(
    train_mod,
    "partition_server_cpus",
    lambda **kw: (
      calls.__setitem__("cpu_plan", int(calls["cpu_plan"]) + 1)
      or SimpleNamespace(worker_io_cpu_ids=())
    ),
  )
  monkeypatch.setattr(train_mod, "apply_cpu_execution_plan", lambda plan: {})
  monkeypatch.setattr(
    train_mod,
    "prewarm_async_checkpoint_worker",
    lambda **kw: calls.__setitem__("prewarm", int(calls["prewarm"]) + 1),
  )
  monkeypatch.setattr(train_mod, "ManagerBasedRlEnv", _FakeEnv)
  monkeypatch.setattr(
    train_mod, "RslRlVecEnvWrapper", lambda env, clip_actions=None: env
  )
  monkeypatch.setattr(train_mod, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train_mod, "dump_yaml", lambda *a, **kw: None)
  monkeypatch.setattr(train_mod, "get_physics_backend_info", lambda b: {})
  monkeypatch.setattr(train_mod, "add_wandb_tags", lambda tags: None)

  cfg = train_mod.TrainConfig(
    env=_FakeEnvCfg(),  # type: ignore[arg-type]
    agent=_FakeAgentCfg(),  # type: ignore[arg-type]
    fp32_exact=True,
    cpu_optimizations=False,
  )
  run_dir = tmp_path / "run_off"
  (run_dir / "params").mkdir(parents=True)
  train_mod.run_train("Mjlab-Velocity-Flat-Unitree-G1", cfg, run_dir)
  assert calls["step_opts"] == 1
  assert calls["cpu_plan"] == 0
  assert calls["prewarm"] == 0
  assert calls["runner_cpu_opts"] == 0
