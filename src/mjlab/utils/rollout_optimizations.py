"""Permanently-enabled FP32-exact execution & zero-sync rollout optimization helpers."""

from __future__ import annotations

from typing import Any

FP32_EXACT_ENABLED: bool = True
EXECUTION_VERSION: str = "mlab4_fast_fp32_exact_v1"


def configure_fp32_exact(cfg: Any, *, enabled: bool = True) -> dict[str, Any]:
  """Configure FP32-exact lossless rollout/physics acceleration on an env or command config."""
  mode = "fp32_exact" if enabled else "off"
  cfg._rollout_optimization_mode = mode
  cfg._use_substep_cuda_graph = bool(enabled)
  cfg._use_zero_sync_optimizations = bool(enabled)
  cfg._use_fp32_fast_optimizations = False
  cfg._use_max_optimizations = False

  commands = getattr(cfg, "commands", None)
  if isinstance(commands, dict):
    for cmd_cfg in commands.values():
      if cmd_cfg is None:
        continue
      cmd_cfg._rollout_optimization_mode = mode
      cmd_cfg._use_substep_cuda_graph = bool(enabled)
      cmd_cfg._use_zero_sync_optimizations = bool(enabled)
      if hasattr(cmd_cfg, "batch_reset_metrics"):
        cmd_cfg.batch_reset_metrics = bool(enabled)
      if hasattr(cmd_cfg, "cache_reference_kernels"):
        cmd_cfg.cache_reference_kernels = bool(enabled)

  return {
    "version": EXECUTION_VERSION,
    "mode": mode,
    "fp32_exact": bool(enabled),
    "substep_cuda_graph": bool(enabled),
    "zero_sync_optimizations": bool(enabled),
  }
