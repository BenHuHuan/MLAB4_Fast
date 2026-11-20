"""Zero-sync strict-IEEE-FP32 PPO subclass for rsl-rl-lib v5 (no site-packages edits).

Eliminates all per-minibatch GPU<->CPU synchronizations inside ``PPO.update()``:
1. GPU-resident float64->float32 exact-schedule adaptive KL learning rate with
   zero-sync 8-byte GPU ``dist.broadcast`` in multi-GPU DDP (preventing 1-ULP LR drift).
2. Native PyTorch fused Adam ``found_inf`` fail-closed nonfinite gradient/loss latching
   (leaves weights, moments, and step counters bit-identically untouched on NaN/Inf).
3. Skips wasted ``GaussianDistribution.sample()`` during policy update forward passes.
4. Replaces ``EmpiricalNormalization._std = torch.sqrt(self._var)`` with in-place
   ``torch.sqrt(norm._var, out=norm._std)`` and cached GPU ``count_x`` so tensor
   addresses stay stable for CUDA Graph replay and incur zero pageable H2D copies.
5. Coalesces all metrics, ``nonfinite`` flag, and exact float64 learning rate into ONE
   single float64 D2H readback at the end of ``update()``.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


def export_compatible_adam_state_dict(
  optimizer: torch.optim.Optimizer,
  *,
  learning_rate: float | None = None,
) -> dict[str, Any]:
  """Export an optimizer state_dict compatible with standard unfused rsl-rl Adam checkpoints."""
  raw = optimizer.state_dict()
  has_fused_or_tensor_lr = any(
    g.get("fused") or torch.is_tensor(g.get("lr")) for g in raw.get("param_groups", [])
  )
  if not has_fused_or_tensor_lr:
    return raw

  state_out: dict[Any, Any] = {}
  for key, val in raw["state"].items():
    row = dict(val)
    if "step" in row and torch.is_tensor(row["step"]):
      row["step"] = row["step"].detach().to(device="cpu", dtype=torch.float32)
    state_out[key] = row

  groups_out: list[dict[str, Any]] = []
  for g in raw["param_groups"]:
    g_copy = dict(g)
    if torch.is_tensor(g_copy.get("lr")):
      g_copy["lr"] = (
        float(learning_rate)
        if learning_rate is not None
        else float(g_copy["lr"].detach().item())
      )
    elif learning_rate is not None:
      g_copy["lr"] = float(learning_rate)
    g_copy["fused"] = None
    groups_out.append(g_copy)
  return {"state": state_out, "param_groups": groups_out}


def restore_adam_state_dict_preserving_fused(
  optimizer: torch.optim.Optimizer,
  state_dict: dict[str, Any],
) -> None:
  """Load an Adam state_dict while preserving fused/foreach/capturable and step tensor device."""
  desired = [
    {
      "fused": g.get("fused", None),
      "foreach": g.get("foreach", None),
      "capturable": g.get("capturable", False),
    }
    for g in optimizer.param_groups
  ]
  optimizer.load_state_dict(state_dict)
  for group, cfg in zip(optimizer.param_groups, desired, strict=True):
    if "fused" in group or cfg["fused"] is not None:
      group["fused"] = cfg["fused"]
    if cfg["fused"]:
      group["foreach"] = None if cfg["foreach"] is None else False
    elif "foreach" in group or cfg["foreach"] is not None:
      group["foreach"] = cfg["foreach"]
    if "capturable" in group or cfg["capturable"]:
      group["capturable"] = cfg["capturable"]
    need_device_step = bool(group.get("fused") or group.get("capturable"))
    for p in group["params"]:
      p_state = optimizer.state.get(p)
      if p_state and "step" in p_state:
        target_dev = p.device if need_device_step else torch.device("cpu")
        if torch.is_tensor(p_state["step"]):
          p_state["step"] = p_state["step"].to(device=target_dev, dtype=torch.float32)
        elif need_device_step:
          p_state["step"] = torch.tensor(
            float(p_state["step"]), device=target_dev, dtype=torch.float32
          )


def update_empirical_normalizer_inplace(
  norm: EmpiricalNormalization,
  x: torch.Tensor,
  count_cache: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] | None = None,
) -> None:
  """Zero-sync in-place moment update preserving ``norm._std.data_ptr()`` for CUDA Graphs."""
  if not norm.training:
    return

  dev = norm.count.device
  dtype_c = norm.count.dtype
  batch_n = int(x.shape[0])
  if count_cache is not None:
    key = (batch_n, dtype_c, dev)
    count_x_base = count_cache.get(key)
    if count_x_base is None:
      count_x_base = torch.tensor(batch_n, dtype=dtype_c, device=dev)
      count_cache[key] = count_x_base
    count_x = (
      count_x_base.clone() if torch.distributed.is_initialized() else count_x_base
    )
  else:
    count_x = torch.tensor(batch_n, dtype=dtype_c, device=dev)

  var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
  mean_x = torch.mean(x, dim=0, keepdim=True)

  if torch.distributed.is_initialized():
    local_mean_x = mean_x
    torch.distributed.all_reduce(count_x)
    mean_sum_x = mean_x * batch_n
    torch.distributed.all_reduce(mean_sum_x)
    mean_x = mean_sum_x / count_x
    var_sum_x = batch_n * (var_x + (local_mean_x - mean_x).square())
    torch.distributed.all_reduce(var_sum_x)
    var_x = var_sum_x / count_x

  if norm.until is not None:
    active_i = (norm.count < norm.until).to(dtype=dtype_c)
    active_f = active_i.to(dtype=norm._mean.dtype)
    eff_count_x = count_x * active_i
    new_count = norm.count + eff_count_x
    denom = new_count.clamp_min(1)
    rate = (eff_count_x / denom).to(dtype=norm._mean.dtype) * active_f
    norm.count.copy_(new_count)
  else:
    norm.count.add_(count_x)
    rate = (count_x / norm.count).to(dtype=norm._mean.dtype)

  delta_mean = mean_x - norm._mean
  norm._mean.add_(rate * delta_mean)
  norm._var.add_(rate * (var_x - norm._var + delta_mean * (mean_x - norm._mean)))
  torch.sqrt(norm._var, out=norm._std)


class ZeroSyncPPO(PPO):
  """Lossless IEEE-FP32 zero-sync PPO update built on rsl-rl-lib v5 ``PPO``."""

  def __init__(
    self,
    *args: Any,
    max_learning_rate: float = 1e-2,
    strict_rng_parity: bool = False,
    zero_sync_update: bool = True,
    **kwargs: Any,
  ) -> None:
    self._optimizer_name = str(kwargs.get("optimizer", "adam")).lower()
    self.max_learning_rate = float(max_learning_rate)
    self.strict_rng_parity = bool(strict_rng_parity)
    super().__init__(*args, **kwargs)

    dev = torch.device(self.device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()
    self.zero_sync_update = bool(
      zero_sync_update
      and not self.strict_rng_parity
      and is_cuda
      and self._optimizer_name == "adam"
    )
    self._norm_count_cache: dict[
      tuple[int, torch.dtype, torch.device], torch.Tensor
    ] = {}
    self._lr_fp64: torch.Tensor | None = None
    self._lr_gpu: torch.Tensor | None = None

    if self.zero_sync_update:
      self._lr_fp64 = torch.tensor(
        float(self.learning_rate), device=dev, dtype=torch.float64
      )
      self._lr_factor_fp64 = torch.tensor(1.5, device=dev, dtype=torch.float64)
      self._lr_gpu = torch.tensor(
        float(self.learning_rate), device=dev, dtype=torch.float32
      )
      self.optimizer = torch.optim.Adam(
        chain(self.actor.parameters(), self.critic.parameters()),
        lr=self._lr_gpu,
        fused=True,
      )
      self.optimizer.found_inf = torch.zeros((), device=dev, dtype=torch.float32)  # type: ignore[attr-defined]
      if self.rnd is not None and isinstance(self.rnd.optimizer, torch.optim.Adam):
        rnd_lr = float(self.rnd.optimizer.param_groups[0]["lr"])
        self.rnd.optimizer = torch.optim.Adam(
          self.rnd.predictor.parameters(),
          lr=rnd_lr,
          fused=True,
        )
        self.rnd.optimizer.found_inf = torch.zeros((), device=dev, dtype=torch.float32)  # type: ignore[attr-defined]

  def _update_actor_distribution_no_sample(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None,
    hidden_state: Any,
  ) -> None:
    """Refresh actor distribution parameters for PPO update without wasted RNG sampling."""
    actor = self.actor
    is_compiled = type(actor).__name__ == "OptimizedModule"
    if (
      not is_compiled
      and getattr(actor, "distribution", None) is not None
      and hasattr(actor, "get_latent")
      and hasattr(actor, "mlp")
    ):
      obs_in = (
        unpad_trajectories(obs, masks)
        if masks is not None and not actor.is_recurrent
        else obs
      )
      latent = actor.get_latent(obs_in, masks, hidden_state)
      mlp_output = actor.mlp(latent)
      actor.distribution.update(mlp_output)
    else:
      actor(
        obs,
        masks=masks,
        hidden_state=hidden_state,
        stochastic_output=True,
      )

  def _update_model_normalization_inplace(
    self, model: nn.Module, obs: TensorDict
  ) -> None:
    if not getattr(model, "obs_normalization", False):
      return
    norm = getattr(model, "obs_normalizer", None)
    obs_groups = getattr(model, "obs_groups", None)
    if isinstance(norm, EmpiricalNormalization) and obs_groups is not None:
      obs_list = [obs[g] for g in obs_groups]
      mlp_obs = obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)
      update_empirical_normalizer_inplace(norm, mlp_obs, self._norm_count_cache)
    elif hasattr(model, "update_normalization"):
      model.update_normalization(obs)  # type: ignore[operator]

  def update(self) -> dict[str, float]:
    """Run zero-sync PPO update epochs with ONE coalesced D2H readback at completion."""
    if not self.zero_sync_update:
      return super().update()

    assert self._lr_fp64 is not None and self._lr_gpu is not None
    dev = torch.device(self.device)
    has_rnd = self.rnd is not None
    has_sym = self.symmetry is not None
    n_metrics = 3 + int(has_rnd) + int(has_sym)
    metrics_accum = torch.zeros(n_metrics, device=dev, dtype=torch.float32)
    update_nonfinite = torch.zeros((), device=dev, dtype=torch.bool)
    rnd_nonfinite = torch.zeros((), device=dev, dtype=torch.bool) if has_rnd else None

    # Sync GPU LR tensors with self.learning_rate in case external code modified it.
    self._lr_fp64.fill_(float(self.learning_rate))
    self._lr_gpu.copy_(self._lr_fp64.to(torch.float32))
    for group in self.optimizer.param_groups:
      group["lr"] = self._lr_gpu

    actor_params = list(self.actor.parameters())
    critic_params = list(self.critic.parameters())

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )  # type: ignore[operator]

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      with torch.amp.autocast(  # type: ignore[attr-defined]
        device_type=dev.type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self._update_actor_distribution_no_sample(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
        values = self.critic(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[1],
        )
        distribution_params = tuple(
          p[:original_batch_size] for p in self.actor.output_distribution_params
        )
        entropy = self.actor.output_entropy[:original_batch_size]

        # GPU-resident exact float64->float32 adaptive KL schedule (zero host sync).
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              batch.old_distribution_params, distribution_params
            )  # type: ignore[arg-type]
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
              torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
              kl_mean.div_(self.gpu_world_size)
            if not self.is_multi_gpu or self.gpu_global_rank == 0:
              hi = kl_mean > (self.desired_kl * 2.0)
              lo = (kl_mean < (self.desired_kl * 0.5)) & (kl_mean > 0.0)
              next_lr = torch.where(
                hi,
                (self._lr_fp64 / self._lr_factor_fp64).clamp_min(1e-5),
                torch.where(
                  lo,
                  (self._lr_fp64 * self._lr_factor_fp64).clamp_max(
                    self.max_learning_rate
                  ),
                  self._lr_fp64,
                ),
              )
              self._lr_fp64.copy_(next_lr)
            if self.is_multi_gpu:
              # Zero-sync 8-byte GPU broadcast eliminates cross-rank 1-ULP LR divergence.
              torch.distributed.broadcast(self._lr_fp64, src=0)
            self._lr_gpu.copy_(self._lr_fp64.to(torch.float32))

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore[arg-type]
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        entropy_mean = entropy.mean()
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_mean
        )

        rnd_loss = (
          self.rnd.compute_loss(batch.observations[:original_batch_size])
          if self.rnd is not None
          else None
        )
        symmetry_loss = None
        if self.symmetry is not None:
          symmetry_loss = self.symmetry.compute_loss(
            self.actor, batch, original_batch_size
          )
          if self.symmetry.use_mirror_loss:
            loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if self.rnd is not None and rnd_loss is not None:
        self.rnd.optimizer.zero_grad(set_to_none=True)
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      norm_a = nn.utils.clip_grad_norm_(
        actor_params, self.max_grad_norm, error_if_nonfinite=False
      )
      norm_c = nn.utils.clip_grad_norm_(
        critic_params, self.max_grad_norm, error_if_nonfinite=False
      )
      step_bad = ~(
        torch.isfinite(loss.detach()) & torch.isfinite(norm_a) & torch.isfinite(norm_c)
      )
      update_nonfinite.logical_or_(step_bad)
      self.optimizer.found_inf.copy_(update_nonfinite.to(torch.float32))  # type: ignore[attr-defined]
      self.optimizer.step()

      if self.rnd is not None and rnd_loss is not None:
        assert rnd_nonfinite is not None
        rnd_nonfinite.logical_or_(~torch.isfinite(rnd_loss.detach()))
        if hasattr(self.rnd.optimizer, "found_inf"):
          self.rnd.optimizer.found_inf.copy_(rnd_nonfinite.to(torch.float32))  # type: ignore[attr-defined]
        self.rnd.optimizer.step()

      # Accumulate metrics purely on GPU (zero .item() calls).
      metrics_accum[0].add_(value_loss.detach())
      metrics_accum[1].add_(surrogate_loss.detach())
      metrics_accum[2].add_(entropy_mean.detach())
      idx = 3
      if has_rnd and rnd_loss is not None:
        metrics_accum[idx].add_(rnd_loss.detach())
        idx += 1
      if has_sym and symmetry_loss is not None:
        metrics_accum[idx].add_(symmetry_loss.detach())

    # Update normalizers in-place (preserves _std.data_ptr(), zero pageable H2D).
    obs = self.storage.observations.flatten(0, 1)
    self._update_model_normalization_inplace(self._raw_actor, obs)
    self._update_model_normalization_inplace(self._raw_critic, obs)
    if self.rnd is not None:
      self.rnd.update_normalization(obs)  # type: ignore[arg-type]

    num_updates = self.num_learning_epochs * self.num_mini_batches
    means_gpu = (metrics_accum / float(num_updates)).to(torch.float64)
    any_bad = (
      update_nonfinite | rnd_nonfinite
      if rnd_nonfinite is not None
      else update_nonfinite
    )

    # ONE single float64 D2H readback for the entire PPO.update() call.
    packed = (
      torch.cat(
        (
          means_gpu,
          any_bad.to(torch.float64).reshape(1),
          self._lr_fp64.reshape(1),
        ),
        dim=0,
      )
      .cpu()
      .tolist()
    )

    self.storage.clear()

    if packed[n_metrics] > 0.0:
      raise FloatingPointError(
        "Nonfinite PPO loss or gradient detected during ZeroSyncPPO.update() "
        "(optimizer update safely skipped via fused found_inf guard)."
      )

    self.learning_rate = float(packed[n_metrics + 1])

    loss_dict: dict[str, float] = {
      "value": float(packed[0]),
      "surrogate": float(packed[1]),
      "entropy": float(packed[2]),
    }
    idx = 3
    if has_rnd:
      loss_dict["rnd"] = float(packed[idx])
      idx += 1
    if has_sym:
      loss_dict["symmetry"] = float(packed[idx])
    return loss_dict

  def save(self) -> dict[str, Any]:
    """Serialize models and unfused-compatible optimizer state."""
    saved_dict: dict[str, Any] = {
      "actor_state_dict": self._raw_actor.state_dict(),
      "critic_state_dict": self._raw_critic.state_dict(),
      "optimizer_state_dict": export_compatible_adam_state_dict(
        self.optimizer, learning_rate=self.learning_rate
      ),
    }
    if self.rnd is not None:
      saved_dict["rnd_state_dict"] = self.rnd.state_dict()
      saved_dict["rnd_optimizer_state_dict"] = export_compatible_adam_state_dict(
        self.rnd.optimizer
      )
    return saved_dict

  def load(
    self,
    loaded_dict: dict[str, Any],
    load_cfg: dict[str, Any] | None,
    strict: bool,
  ) -> bool:
    """Load checkpoint and restore GPU-LR / fused Adam / found_inf bindings."""
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
        "rnd": True,
      }

    if load_cfg.get("actor"):
      self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if load_cfg.get("critic"):
      self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
    if load_cfg.get("optimizer"):
      restore_adam_state_dict_preserving_fused(
        self.optimizer, loaded_dict["optimizer_state_dict"]
      )
      loaded_lr = self.optimizer.param_groups[0]["lr"]
      self.learning_rate = float(
        loaded_lr.item() if torch.is_tensor(loaded_lr) else loaded_lr
      )
      if (
        self.zero_sync_update and self._lr_fp64 is not None and self._lr_gpu is not None
      ):
        self._lr_fp64.fill_(self.learning_rate)
        self._lr_gpu.copy_(self._lr_fp64.to(torch.float32))
        for group in self.optimizer.param_groups:
          group["lr"] = self._lr_gpu
        if not hasattr(self.optimizer, "found_inf"):
          self.optimizer.found_inf = torch.zeros(  # type: ignore[attr-defined]
            (), device=self.device, dtype=torch.float32
          )
    if load_cfg.get("rnd") and self.rnd is not None:
      self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
      restore_adam_state_dict_preserving_fused(
        self.rnd.optimizer, loaded_dict["rnd_optimizer_state_dict"]
      )
      if self.zero_sync_update and not hasattr(self.rnd.optimizer, "found_inf"):
        self.rnd.optimizer.found_inf = torch.zeros(  # type: ignore[attr-defined]
          (), device=self.device, dtype=torch.float32
        )
    return bool(load_cfg.get("iteration", False))
