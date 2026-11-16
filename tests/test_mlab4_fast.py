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
      cr.CpuCoreGroup(package_id=0, core_id=0, numa_node=0, max_freq_khz=5200000, cpu_ids=(0, 1)),
      cr.CpuCoreGroup(package_id=0, core_id=1, numa_node=0, max_freq_khz=5200000, cpu_ids=(2, 3)),
      cr.CpuCoreGroup(package_id=0, core_id=2, numa_node=0, max_freq_khz=3800000, cpu_ids=(4,)),
      cr.CpuCoreGroup(package_id=0, core_id=3, numa_node=0, max_freq_khz=3800000, cpu_ids=(5,)),
    ),
    numa_nodes=(0,),
  )
  assert cr.get_performance_cpu_ids((0, 1, 2, 3, 4, 5), one_per_core=True, topology=hybrid) == (0, 2)


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
  plan = cr.partition_server_cpus(world_size=1, rank=0, topology=topo, driver_physical_cores=2)
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
      worker.submit_save({"iter": i, "w": torch.full((32, 32), float(i + 1))}, tmp_path / f"m_{i}.pt")
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
