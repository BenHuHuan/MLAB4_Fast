"""Permanently-enabled FP32-exact & FP32-fast lossless execution optimization helpers.

Both ``fp32_exact`` and ``fp32_fast`` enforce strict 23-bit IEEE 754 FP32
(``allow_tf32=False``, zero BF16/TF32 precision loss):

- ``fp32_exact`` (default):
  ``#2A`` (pre-import CPU P-core/OpenMP/NUMA isolation + async I/O)
  + ``#2B`` (``ZeroSyncPPO`` exact IEEE float64 GPU adaptive KL LR + native
    fused Adam ``found_inf`` guard + in-place normalizer)
  + ``#1A`` (Tier 1 ``EnvFullStepGraph`` + ``ShiftRegisterBuffer``
    + ``AsyncResetPool``).
- ``fp32_fast``:
  Everything in ``fp32_exact`` plus ``#1B`` (Tier 2 ``RolloutStepMegaGraph``
  fusing Actor/Critic + ``EnvFullStepGraph`` + transition staging into one
  CUDA Graph per rollout step, still in strict pure 32-bit IEEE FP32).
- ``strict_rng_parity=True``:
  Opt-out regression switch keeping strict IEEE FP32 + Phase-0 substep graph
  while disabling ``AsyncResetPool``, Tier 1/2 Mega-Graphs, and ``ZeroSyncPPO``
  for legacy bitwise PRNG comparisons.
"""

from __future__ import annotations

from typing import Any

FP32_EXACT_ENABLED: bool = True
EXECUTION_VERSION: str = "mlab4_fast_fp32_v2"


def _apply_mode_flags(
  cfg: Any,
  *,
  mode: str,
  enabled: bool,
  fp32_fast: bool,
  strict_rng_parity: bool,
  mega_graph_mode: str,
  reset_pool_mode: str,
) -> dict[str, Any]:
  eff_mega = "off" if (not enabled or strict_rng_parity) else str(mega_graph_mode)
  eff_pool = "off" if (not enabled or strict_rng_parity) else str(reset_pool_mode)
  cfg._rollout_optimization_mode = mode
  cfg._use_substep_cuda_graph = bool(enabled)
  cfg._use_zero_sync_optimizations = bool(enabled)
  cfg._use_fp32_fast_optimizations = bool(
    enabled and fp32_fast and not strict_rng_parity
  )
  cfg._use_max_optimizations = False
  cfg._strict_rng_parity = bool(strict_rng_parity)
  cfg._mega_graph_mode = eff_mega
  cfg._reset_pool_mode = eff_pool

  commands = getattr(cfg, "commands", None)
  if isinstance(commands, dict):
    for cmd_cfg in commands.values():
      if cmd_cfg is None:
        continue
      cmd_cfg._rollout_optimization_mode = mode
      cmd_cfg._use_substep_cuda_graph = bool(enabled)
      cmd_cfg._use_zero_sync_optimizations = bool(enabled)
      cmd_cfg._use_fp32_fast_optimizations = bool(
        enabled and fp32_fast and not strict_rng_parity
      )
      cmd_cfg._strict_rng_parity = bool(strict_rng_parity)
      if hasattr(cmd_cfg, "batch_reset_metrics"):
        cmd_cfg.batch_reset_metrics = bool(enabled)
      if hasattr(cmd_cfg, "cache_reference_kernels"):
        cmd_cfg.cache_reference_kernels = bool(enabled)

  return {
    "version": EXECUTION_VERSION,
    "mode": mode,
    "fp32_exact": bool(enabled),
    "fp32_fast": bool(enabled and fp32_fast and not strict_rng_parity),
    "strict_rng_parity": bool(strict_rng_parity),
    "substep_cuda_graph": bool(enabled),
    "zero_sync_optimizations": bool(enabled),
    "mega_graph_mode": eff_mega,
    "reset_pool_mode": eff_pool,
  }


def configure_fp32_exact(
  cfg: Any,
  *,
  enabled: bool = True,
  strict_rng_parity: bool = False,
  mega_graph_mode: str = "env",
  reset_pool_mode: str = "auto",
) -> dict[str, Any]:
  """Configure FP32-exact lossless acceleration (#2A + #2B + #1A Tier 1 env graph)."""
  return _apply_mode_flags(
    cfg,
    mode="fp32_exact" if enabled else "off",
    enabled=enabled,
    fp32_fast=False,
    strict_rng_parity=strict_rng_parity,
    mega_graph_mode=mega_graph_mode,
    reset_pool_mode=reset_pool_mode,
  )


def configure_fp32_fast(
  cfg: Any,
  *,
  enabled: bool = True,
  strict_rng_parity: bool = False,
  mega_graph_mode: str = "rollout",
  reset_pool_mode: str = "auto",
) -> dict[str, Any]:
  """Configure pure-FP32 max-throughput acceleration (#2A + #2B + #1A + #1B Tier 2 rollout mega-graph)."""
  return _apply_mode_flags(
    cfg,
    mode="fp32_fast" if enabled else "off",
    enabled=enabled,
    fp32_fast=True,
    strict_rng_parity=strict_rng_parity,
    mega_graph_mode=mega_graph_mode,
    reset_pool_mode=reset_pool_mode,
  )


def resolve_fp32_execution_mode(
  *,
  fp32_exact: bool = True,
  fp32_fast: bool = False,
  strict_rng_parity: bool = False,
  mega_graph_mode: str = "auto",
  reset_pool_mode: str = "auto",
) -> dict[str, Any]:
  """Resolve effective IEEE FP32 execution settings from high-level CLI flags."""
  ieee_enabled = bool(fp32_exact or fp32_fast)
  fast_enabled = bool(fp32_fast and not strict_rng_parity)
  if not ieee_enabled or strict_rng_parity:
    eff_mega = "off"
    eff_pool = "off"
  else:
    if mega_graph_mode == "auto":
      eff_mega = "rollout" if fast_enabled else "env"
    else:
      eff_mega = str(mega_graph_mode)
    eff_pool = str(reset_pool_mode)

  mode = "off" if not ieee_enabled else ("fp32_fast" if fast_enabled else "fp32_exact")
  return {
    "mode": mode,
    "ieee_enabled": ieee_enabled,
    "allow_tf32": not ieee_enabled,
    "fp32_fast": fast_enabled,
    "strict_rng_parity": bool(strict_rng_parity),
    "zero_sync_ppo": bool(ieee_enabled and not strict_rng_parity),
    "zero_sync_rollout": bool(ieee_enabled and not strict_rng_parity),
    "mega_graph_mode": eff_mega,
    "reset_pool_mode": eff_pool,
  }
