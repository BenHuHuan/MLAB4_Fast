"""Tests for Phase 1 (#2A CPU pre-import bootstrap + #2B ZeroSyncPPO)."""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.rl.zero_sync_ppo import (
  ZeroSyncPPO,
  update_empirical_normalizer_inplace,
)
from mjlab.utils import cpu_runtime as cr


def _make_dummy_obs(num_envs: int, obs_dim: int, device: str) -> TensorDict:
  return TensorDict(
    {"actor": torch.randn(num_envs, obs_dim, device=device)},
    batch_size=[num_envs],
    device=device,
  )


def _build_ppo_pair(
  *,
  num_envs: int = 32,
  num_steps: int = 8,
  obs_dim: int = 24,
  act_dim: int = 8,
  device: str = "cuda",
  obs_norm: bool = True,
):
  obs = _make_dummy_obs(num_envs, obs_dim, device)
  obs_groups = {"actor": ["actor"], "critic": ["actor"]}
  dist_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }

  torch.manual_seed(12345)
  actor_ref = MLPModel(
    obs,
    obs_groups,
    "actor",
    act_dim,
    hidden_dims=(64, 64),
    obs_normalization=obs_norm,
    distribution_cfg=dist_cfg,
  ).to(device)
  critic_ref = MLPModel(
    obs,
    obs_groups,
    "critic",
    1,
    hidden_dims=(64, 64),
    obs_normalization=obs_norm,
  ).to(device)
  actor_zs = copy.deepcopy(actor_ref).to(device)
  critic_zs = copy.deepcopy(critic_ref).to(device)

  storage_ref = RolloutStorage("rl", num_envs, num_steps, obs, [act_dim], device)
  storage_zs = RolloutStorage("rl", num_envs, num_steps, obs, [act_dim], device)

  common = dict(
    num_learning_epochs=3,
    num_mini_batches=4,
    learning_rate=3e-4,
    schedule="adaptive",
    desired_kl=0.01,
    device=device,
  )
  alg_ref = PPO(actor_ref, critic_ref, storage_ref, **common)
  alg_zs = ZeroSyncPPO(actor_zs, critic_zs, storage_zs, **common)

  # Fill identical synthetic rollout transitions.
  torch.manual_seed(9876)
  with torch.inference_mode():
    for _ in range(num_steps):
      step_obs = _make_dummy_obs(num_envs, obs_dim, device)
      act = alg_ref.act(step_obs)
      alg_zs.transition.hidden_states = alg_ref.transition.hidden_states
      alg_zs.transition.actions = act.clone()
      alg_zs.transition.values = alg_ref.transition.values.clone()
      alg_zs.transition.actions_log_prob = alg_ref.transition.actions_log_prob.clone()
      alg_zs.transition.distribution_params = tuple(
        p.clone() for p in alg_ref.transition.distribution_params
      )
      alg_zs.transition.observations = step_obs.clone()
      rew = torch.randn(num_envs, device=device)
      dones = (torch.rand(num_envs, device=device) < 0.1).long()
      alg_ref.process_env_step(step_obs, rew, dones, {})
      alg_zs.process_env_step(step_obs.clone(), rew.clone(), dones.clone(), {})
    last_obs = _make_dummy_obs(num_envs, obs_dim, device)
    alg_ref.compute_returns(last_obs)
    alg_zs.compute_returns(last_obs.clone())

  return alg_ref, alg_zs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_sync_ppo_ieee_fp32_parity_and_single_sync_guard():
  prev_prec = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
  if prev_prec is not None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
  try:
    alg_ref, alg_zs = _build_ppo_pair(device="cuda", obs_norm=True)
    assert alg_zs.zero_sync_update is True

    actor_std_ptr_before = alg_zs._raw_actor.obs_normalizer._std.data_ptr()
    critic_std_ptr_before = alg_zs._raw_critic.obs_normalizer._std.data_ptr()

    # Run reference PPO update
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    out_ref = alg_ref.update()

    # Run ZeroSyncPPO update under CUDA sync debug mode="warn" and count sync warnings!
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    prev_sync_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_zs = alg_zs.update()
    finally:
      torch.cuda.set_sync_debug_mode(prev_sync_mode)

    sync_warns = [w for w in caught if "synchronize" in str(w.message).lower()]
    # Expect ONLY 1 D2H sync for the entire 3x4=12-minibatch update!
    assert len(sync_warns) <= 1, (
      f"Expected <= 1 CUDA sync inside ZeroSyncPPO.update(), got {len(sync_warns)}: "
      f"{[str(w.message) for w in sync_warns]}"
    )

    # Strict 0.0 bit-for-bit float64 adaptive LR schedule match (true IEEE div.rn.f64)!
    assert alg_zs.learning_rate == alg_ref.learning_rate

    # In-place normalizer _std data_ptr preserved for CUDA Graph replay!
    assert alg_zs._raw_actor.obs_normalizer._std.data_ptr() == actor_std_ptr_before
    assert alg_zs._raw_critic.obs_normalizer._std.data_ptr() == critic_std_ptr_before
    torch.testing.assert_close(
      alg_zs._raw_actor.obs_normalizer._std,
      alg_ref._raw_actor.obs_normalizer._std,
      rtol=1e-6,
      atol=1e-6,
    )

    # IEEE FP32 weights match within ~1-2 ULP (fused vs unfused FMA ordering)
    for (n, p_ref), (_, p_zs) in zip(
      alg_ref.actor.named_parameters(), alg_zs.actor.named_parameters(), strict=True
    ):
      torch.testing.assert_close(p_zs, p_ref, rtol=1e-5, atol=1e-6, msg=f"actor.{n}")
    for (n, p_ref), (_, p_zs) in zip(
      alg_ref.critic.named_parameters(), alg_zs.critic.named_parameters(), strict=True
    ):
      torch.testing.assert_close(p_zs, p_ref, rtol=1e-5, atol=1e-6, msg=f"critic.{n}")
    for k in ("value", "surrogate", "entropy"):
      assert abs(out_zs[k] - out_ref[k]) < 1e-5, (
        f"metric {k}: {out_zs[k]} vs {out_ref[k]}"
      )
  finally:
    if prev_prec is not None:
      torch.backends.cuda.matmul.fp32_precision = prev_prec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_sync_ppo_fail_closed_found_inf_and_checkpoint_compat():
  alg_ref, alg_zs = _build_ppo_pair(device="cuda", obs_norm=True)

  # 1. Checkpoint round-trip: ZeroSyncPPO -> vanilla PPO -> ZeroSyncPPO
  alg_zs.learning_rate = 1.2345e-4
  alg_zs._lr_fp64.fill_(alg_zs.learning_rate)
  alg_zs._lr_gpu.fill_(alg_zs.learning_rate)
  ckpt = alg_zs.save()
  for g in ckpt["optimizer_state_dict"]["param_groups"]:
    assert isinstance(g["lr"], float)
    assert g["fused"] is None
  alg_ref.load(ckpt, load_cfg=None, strict=True)
  assert abs(alg_ref.learning_rate - 1.2345e-4) < 1e-12

  _, alg_zs2 = _build_ppo_pair(device="cuda", obs_norm=True)
  alg_zs2.load(ckpt, load_cfg=None, strict=True)
  assert abs(alg_zs2.learning_rate - 1.2345e-4) < 1e-12
  assert alg_zs2.optimizer.param_groups[0]["lr"] is alg_zs2._lr_gpu
  assert abs(float(alg_zs2._lr_fp64.item()) - 1.2345e-4) < 1e-12

  # 2. Fail-closed found_inf protection: poison storage returns with NaN -> weights untouched!
  snap_actor = [p.detach().clone() for p in alg_zs2.actor.parameters()]
  snap_critic = [p.detach().clone() for p in alg_zs2.critic.parameters()]
  alg_zs2.storage.returns.fill_(float("nan"))
  with pytest.raises(FloatingPointError):
    alg_zs2.update()
  for p_old, p_new in zip(snap_actor, alg_zs2.actor.parameters(), strict=True):
    assert torch.equal(p_old, p_new)
  for p_old, p_new in zip(snap_critic, alg_zs2.critic.parameters(), strict=True):
    assert torch.equal(p_old, p_new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_inplace_empirical_normalizer_until_and_pointer():
  from rsl_rl.modules import EmpiricalNormalization

  norm = EmpiricalNormalization(12, until=16).cuda()
  std_ptr = norm._std.data_ptr()
  cache: dict = {}
  update_empirical_normalizer_inplace(norm, torch.randn(8, 12, device="cuda"), cache)
  assert int(norm.count.item()) == 8
  assert norm._std.data_ptr() == std_ptr
  update_empirical_normalizer_inplace(norm, torch.randn(8, 12, device="cuda"), cache)
  assert int(norm.count.item()) == 16
  mean_at_limit = norm._mean.clone()
  update_empirical_normalizer_inplace(
    norm, torch.full((8, 12), 100.0, device="cuda"), cache
  )
  # Beyond `until=16`, stats must freeze bit-identically!
  assert torch.equal(norm._mean, mean_at_limit)
  assert norm._std.data_ptr() == std_ptr


def test_bootstrap_cpu_env_multi_gpu_launcher_does_not_pin():
  env: dict[str, str] = {}
  res = cr.bootstrap_cpu_env_before_torch(
    is_multi_gpu_launcher=True,
    set_affinity=True,
    env=env,
  )
  assert res["is_multi_gpu_launcher"] is True
  assert res["applied_affinity"] is False
  assert env["OMP_PROC_BIND"] == "close"
  assert env["GOMP_SPINCOUNT"] == "200000"
  assert "OMP_NUM_THREADS" not in env


def test_bootstrap_cpu_env_single_gpu_sets_defaults_without_overriding_user():
  topo = cr.CpuTopology(
    allowed_cpus=tuple(range(16)),
    physical_groups=tuple(
      cr.CpuCoreGroup(
        package_id=0, core_id=c, numa_node=0, max_freq_khz=3400000, cpu_ids=(c, c + 8)
      )
      for c in range(8)
    ),
    numa_nodes=(0,),
  )
  env: dict[str, str] = {"OMP_NUM_THREADS": "3"}
  res = cr.bootstrap_cpu_env_before_torch(
    is_multi_gpu_launcher=False,
    set_affinity=False,
    topology=topo,
    env=env,
  )
  assert res["driver_primary_cpu_ids"] == [0, 1]
  assert env["OMP_NUM_THREADS"] == "3"  # user override respected
  assert env["MKL_NUM_THREADS"] == "2"
  assert env["OPENBLAS_NUM_THREADS"] == "1"
  assert env["OMP_PROC_BIND"] == "close"
  assert env["GOMP_SPINCOUNT"] == "200000"
