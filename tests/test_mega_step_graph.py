"""Tests for Phase 2 (#1A ShiftRegisterBuffer + AsyncResetPool + EnvFullStepGraph)
and Phase 3 (#1B Tier 2 RolloutStepMegaGraph) under strict IEEE FP32.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from mjlab.envs.mega_step_graph import EnvFullStepGraph
from mjlab.envs.reset_pool import AsyncResetPool, FieldSpec, choose_ring_pool_size
from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.buffers.delay_buffer import DelayBuffer
from mjlab.utils.buffers.shift_register_buffer import ShiftRegisterBuffer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shift_register_buffer_matches_circular_buffer_and_cuda_graph_zero_sync():
  n, h, d = 64, 4, 32
  device = "cuda"
  cb = CircularBuffer(max_len=h, batch_size=n, device=device)
  sr = ShiftRegisterBuffer(max_len=h, batch_size=n, device=device)

  torch.manual_seed(77)
  for step in range(10):
    frame = torch.randn(n, d, device=device)
    if step in (3, 6):
      reset_mask = torch.rand(n, device=device) < 0.25
      reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
      cb.reset(reset_ids)
      sr.reset_masked(reset_mask)
    cb.append(frame)
    sr.append(frame)
    assert torch.equal(sr.buffer, cb.buffer), f"Mismatch at step {step}"
    assert torch.equal(sr.current_length, cb.current_length)
    lags = torch.randint(0, h, (n,), device=device)
    assert torch.equal(sr[lags], cb[lags])

  # Test SDK bank 4-frame history load + preserve_tail_cols=4 inside CUDAGraph with 0 syncs!
  sdk_hist = torch.arange(16 * h * d, dtype=torch.float32, device=device).reshape(
    16, h, d
  )
  frame_in = torch.full((n, d), -5.0, dtype=torch.float32, device=device)
  reset_in = torch.zeros(n, dtype=torch.bool, device=device)
  reset_in[:10] = True
  is_sdk = torch.zeros(n, dtype=torch.bool, device=device)
  is_sdk[:5] = True
  bank_rows = torch.arange(n, dtype=torch.int32, device=device) % 16

  s = torch.cuda.Stream()
  s.wait_stream(torch.cuda.current_stream())
  g = torch.cuda.CUDAGraph()
  with torch.cuda.stream(s):
    sr.append_masked(
      frame_in,
      reset_mask=reset_in,
      is_sdk_bank=is_sdk,
      bank_row_id=bank_rows,
      sdk_hist=sdk_hist,
      preserve_tail_cols=4,
    )
    with torch.cuda.graph(g, stream=s):
      sr.append_masked(
        frame_in,
        reset_mask=reset_in,
        is_sdk_bank=is_sdk,
        bank_row_id=bank_rows,
        sdk_hist=sdk_hist,
        preserve_tail_cols=4,
      )
  torch.cuda.current_stream().wait_stream(s)

  prev_mode = torch.cuda.get_sync_debug_mode()
  torch.cuda.set_sync_debug_mode("error")
  try:
    g.replay()
  finally:
    torch.cuda.set_sync_debug_mode(prev_mode)
  torch.cuda.synchronize()

  # Lanes 0..4 loaded sdk_hist[:, :-4] and preserved frame_in[-4:] = -5.0
  for i in range(5):
    assert torch.equal(sr.buffer[i, :, :-4], sdk_hist[i, :, :-4])
    assert torch.all(sr.buffer[i, :, -4:] == -5.0)
  # Lanes 5..9 backfilled all H frames with -5.0
  for i in range(5, 10):
    assert torch.all(sr.buffer[i] == -5.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_delay_buffer_shift_register_inplace_lags():
  db = DelayBuffer(
    min_lag=0, max_lag=3, batch_size=16, device="cuda", use_shift_register=True
  )
  lag_ptr = db._current_lags.data_ptr()
  for _ in range(5):
    db.append(torch.randn(16, 8, device="cuda"))
    _ = db.compute()
  assert db._current_lags.data_ptr() == lag_ptr
  mask = torch.zeros(16, dtype=torch.bool, device="cuda")
  mask[:4] = True
  db.reset_masked(mask)
  assert torch.all(db._current_lags[:4] == 0)
  assert db._current_lags.data_ptr() == lag_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_reset_pool_cumsum_wrap_and_async_refill():
  assert choose_ring_pool_size(4096, 24, bytes_per_slot=612) == 131072
  counter = [0]

  def _producer(
    count: int, rng: np.random.Generator, state, out: dict[str, np.ndarray]
  ):
    vals = np.arange(counter[0], counter[0] + count, dtype=np.int64)
    counter[0] += count
    out["id"][:] = vals
    out["vec"][:] = vals[:, None].astype(np.float32)

  pool = AsyncResetPool(
    pool_size=256,
    field_specs={
      "id": FieldSpec((), torch.int64),
      "vec": FieldSpec((4,), torch.float32),
    },
    device="cuda",
    seed=123,
    producer_fn=_producer,
  )
  try:
    pool.prefill_all()
    assert counter[0] == 256

    # Capture consume() inside CUDAGraph and replay 6 times across wraparound boundary!
    mask_in = torch.zeros(64, dtype=torch.bool, device="cuda")
    mask_in[::2] = True  # 32 resets per step
    out_id = torch.empty(64, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
      # Warmup once then reset cursor
      _ = pool.consume(mask_in)
      pool._head.zero_()
      pool._consumed_iter.zero_()
      with torch.cuda.graph(g, stream=s):
        batch = pool.consume(mask_in)
        out_id.copy_(batch["id"])
    torch.cuda.current_stream().wait_stream(s)

    seen: list[int] = []
    for _ in range(6):  # 6 * 32 = 192 <= 256
      g.replay()
      torch.cuda.synchronize()
      seen.extend(out_id[mask_in].cpu().tolist())
    assert seen == list(range(192))
    assert int(pool.consumed_iter_tensor.item()) == 192

    # Kick async refill of 192 slots and verify wraparound refill contents
    pool.consumed_iter_tensor.zero_()
    pool.kick_async_refill(192)
    pool.wait_refill_on_stream()
    # Consume 128 more (crosses ring slot 255 -> 0..63 which were just refilled with 256..319!)
    seen2: list[int] = []
    for _ in range(4):
      g.replay()
      torch.cuda.synchronize()
      seen2.extend(out_id[mask_in].cpu().tolist())
    assert seen2 == list(range(192, 320))

    # Wraparound guard must raise if consumed > pool_size
    with pytest.raises(RuntimeError, match="wraparound"):
      pool.kick_async_refill(300)
  finally:
    pool.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_velocity_env_tier1_and_tier2_mega_graph_ieee_zero_sync(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_exact
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  env_cfg.scene.num_envs = 64
  env_cfg.sim.mujoco.ccd_iterations = 50
  configure_fp32_exact(
    env_cfg, enabled=True, mega_graph_mode="env", reset_pool_mode="async_ring"
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    ok, reason = EnvFullStepGraph.is_supported(env)
    assert ok, f"Velocity-Flat-G1 should be EnvFullStepGraph-eligible, got: {reason}"
    assert env._mega_step_graph is not None
    env.reset()

    # Warmup 2 steps + capture on step 3, then run 24-step rollout under sync_debug_mode="warn"
    act = torch.zeros(64, env.action_manager.total_action_dim, device="cuda:0")
    for _ in range(3):
      env.step(act)
    assert env._mega_step_graph.captures == 1

    env.begin_rollout_iteration(24)
    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    try:
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step_idx in range(24):
          if step_idx in (5, 17):
            env.episode_length_buf[:8] = env.max_episode_length
          obs, rew, term, trunc, extras = env.step(act)
          assert torch.isfinite(rew).all()
    finally:
      torch.cuda.set_sync_debug_mode(prev_mode)

    sync_warns = [w for w in caught if "synchronize" in str(w.message).lower()]
    assert len(sync_warns) == 0, (
      f"Expected 0 CUDA syncs inside 24-step EnvFullStepGraph rollout, got {len(sync_warns)}: "
      f"{[str(w.message) for w in sync_warns]}"
    )
    assert env._mega_step_graph.replays >= 25
    flushed = env.end_rollout_iteration()
    # Both forced-timeout steps produced a coalesced non-diluted ep_infos row!
    assert len(flushed) == 1
    assert env._mega_step_graph.last_nonzero_reset_steps >= 2
    assert all(isinstance(v, float) for row in flushed for v in row.values())

    # Verify Tier 1 EnvFullStepGraph.step() returns non-aliased observations so
    # PPO.act(obs_t) -> env.step(act) -> PPO.process_env_step never clobbers obs_t!
    o_prev, r_prev, _, _, _ = env.step(act)
    o_prev_snap = o_prev["actor"].clone()
    r_prev_snap = r_prev.clone()
    act_perturbed = torch.full_like(act, 0.25)
    o_next, _, _, _, _ = env.step(act_perturbed)
    assert torch.equal(o_prev["actor"], o_prev_snap), (
      "Tier 1 EnvFullStepGraph.step() mutated previously returned obs_t in-place!"
    )
    assert torch.equal(r_prev, r_prev_snap)
    assert not torch.equal(o_prev["actor"], o_next["actor"])

    # Now verify Tier 2 RolloutStepMegaGraph + ZeroSyncPPO end-to-end inside MjlabOnPolicyRunner!
    rl_cfg = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    rl_cfg.num_steps_per_env = 8
    rl_cfg.max_iterations = 3
    rl_cfg.save_interval = 100
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"
    agent_cfg["upload_model"] = False
    agent_cfg["zero_sync_ppo"] = True
    agent_cfg["zero_sync_rollout"] = True
    agent_cfg["mega_graph_mode"] = "rollout"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
      wrapped, agent_cfg, log_dir=str(tmp_path), device="cuda:0"
    )
    assert runner._rollout_mega_graph is None
    std_ptr_before = (
      runner.alg._raw_actor.obs_normalizer._std.data_ptr()
      if getattr(runner.alg._raw_actor, "obs_normalization", False)
      else None
    )
    runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
    assert runner._rollout_mega_graph is not None
    assert runner._rollout_mega_graph.captures == 1
    assert runner._rollout_mega_graph.replays >= 22
    if std_ptr_before is not None:
      assert runner.alg._raw_actor.obs_normalizer._std.data_ptr() == std_ptr_before
  finally:
    env.close()


def test_fp32_exact_and_fp32_fast_mode_resolution_and_ieee_enforcement():
  from types import SimpleNamespace

  from mjlab.utils.rollout_optimizations import (
    configure_fp32_exact,
    configure_fp32_fast,
    resolve_fp32_execution_mode,
  )

  # Default fp32_exact: strict IEEE FP32 + #2A/#2B + #1A Tier 1 env graph.
  r_exact = resolve_fp32_execution_mode(fp32_exact=True, fp32_fast=False)
  assert r_exact["mode"] == "fp32_exact"
  assert r_exact["allow_tf32"] is False
  assert r_exact["ieee_enabled"] is True
  assert r_exact["fp32_fast"] is False
  assert r_exact["zero_sync_ppo"] is True
  assert r_exact["zero_sync_rollout"] is True
  assert r_exact["mega_graph_mode"] == "env"
  assert r_exact["reset_pool_mode"] == "async_ring"

  # fp32_fast: still strict IEEE 32-bit FP32 (allow_tf32=False), upgrades to Tier 2 rollout graph!
  r_fast = resolve_fp32_execution_mode(fp32_exact=True, fp32_fast=True)
  assert r_fast["mode"] == "fp32_fast"
  assert r_fast["allow_tf32"] is False
  assert r_fast["ieee_enabled"] is True
  assert r_fast["fp32_fast"] is True
  assert r_fast["zero_sync_ppo"] is True
  assert r_fast["zero_sync_rollout"] is True
  assert r_fast["mega_graph_mode"] == "rollout"
  assert r_fast["reset_pool_mode"] == "async_ring"

  # strict_rng_parity overrides BOTH fp32_exact and fp32_fast while preserving IEEE FP32.
  r_parity = resolve_fp32_execution_mode(
    fp32_exact=True, fp32_fast=True, strict_rng_parity=True
  )
  assert r_parity["allow_tf32"] is False
  assert r_parity["fp32_fast"] is False
  assert r_parity["zero_sync_ppo"] is False
  assert r_parity["zero_sync_rollout"] is False
  assert r_parity["mega_graph_mode"] == "off"
  assert r_parity["reset_pool_mode"] == "off"

  # Explicit #2-only ablation (mega_graph_mode="off", reset_pool_mode="off").
  r_two = resolve_fp32_execution_mode(
    fp32_exact=True, mega_graph_mode="off", reset_pool_mode="off"
  )
  assert r_two["zero_sync_ppo"] is True
  assert r_two["mega_graph_mode"] == "off"
  assert r_two["reset_pool_mode"] == "off"

  # Off mode restores stock TF32 setting.
  r_off = resolve_fp32_execution_mode(fp32_exact=False, fp32_fast=False)
  assert r_off["mode"] == "off"
  assert r_off["allow_tf32"] is True

  cfg = SimpleNamespace(commands={"cmd": SimpleNamespace()})
  rep_fast = configure_fp32_fast(cfg)
  assert rep_fast["mode"] == "fp32_fast"
  assert rep_fast["fp32_fast"] is True
  assert cfg._mega_graph_mode == "rollout"
  assert cfg._use_fp32_fast_optimizations is True

  rep_exact = configure_fp32_exact(cfg)
  assert rep_exact["mode"] == "fp32_exact"
  assert rep_exact["fp32_fast"] is False
  assert cfg._mega_graph_mode == "env"
  assert cfg._use_fp32_fast_optimizations is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_velocity_rough_env_tier1_and_tier2_zero_sync(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_fast
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  env_cfg = load_env_cfg("Mjlab-Velocity-Rough-Unitree-G1")
  env_cfg.scene.num_envs = 32
  env_cfg.sim.mujoco.ccd_iterations = 50
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax or 0, 512)
  # Shrink terrain grid for fast unit test compilation.
  if (
    env_cfg.scene.terrain is not None
    and env_cfg.scene.terrain.terrain_generator is not None
  ):
    env_cfg.scene.terrain.terrain_generator.num_rows = 3
    env_cfg.scene.terrain.terrain_generator.num_cols = 2
  configure_fp32_fast(env_cfg, enabled=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    ok, reason = EnvFullStepGraph.is_supported(env)
    assert ok, f"Velocity-Rough-G1 should be EnvFullStepGraph-eligible, got: {reason}"
    assert env._mega_step_graph is not None
    env.reset()

    rl_cfg = load_rl_cfg("Mjlab-Velocity-Rough-Unitree-G1")
    rl_cfg.num_steps_per_env = 8
    rl_cfg.max_iterations = 2
    rl_cfg.save_interval = 100
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"
    agent_cfg["upload_model"] = False
    agent_cfg["zero_sync_ppo"] = True
    agent_cfg["zero_sync_rollout"] = True
    agent_cfg["mega_graph_mode"] = "rollout"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
      wrapped, agent_cfg, log_dir=str(tmp_path), device="cuda:0"
    )
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    assert runner._rollout_mega_graph is not None
    assert runner._rollout_mega_graph.captures == 1
    assert runner._rollout_mega_graph.replays >= 14
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_tracking_env_async_reset_pool_and_mega_graph(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_fast
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  env_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1")
  env_cfg.scene.num_envs = 32
  env_cfg.sim.mujoco.ccd_iterations = 50

  # Build a valid synthetic G1 motion .npz (T=60 frames, 30 bodies, 29 joints).
  T = 60
  n_bodies = 30
  n_joints = 29
  joint_pos = np.zeros((T, n_joints), dtype=np.float32)
  joint_vel = np.zeros((T, n_joints), dtype=np.float32)
  body_pos_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  body_pos_w[:, :, 2] = 0.78
  body_quat_w = np.zeros((T, n_bodies, 4), dtype=np.float32)
  body_quat_w[:, :, 0] = 1.0
  body_lin_vel_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((T, n_bodies, 3), dtype=np.float32)
  motion_path = tmp_path / "synthetic_g1_motion.npz"
  np.savez(
    motion_path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )
  env_cfg.commands["motion"].motion_file = str(motion_path)  # type: ignore[attr-defined]
  configure_fp32_fast(env_cfg, enabled=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    ok, reason = EnvFullStepGraph.is_supported(env)
    assert ok, f"Tracking-Flat-G1 should be EnvFullStepGraph-eligible, got: {reason}"
    assert env._mega_step_graph is not None
    motion_cmd = env.command_manager.get_term("motion")
    pool = getattr(motion_cmd, "_async_reset_pool", None)
    assert pool is not None, "AsyncResetPool should attach to MotionCommand"
    env.reset()

    rl_cfg = load_rl_cfg("Mjlab-Tracking-Flat-Unitree-G1")
    rl_cfg.num_steps_per_env = 8
    rl_cfg.max_iterations = 3
    rl_cfg.save_interval = 100
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"
    agent_cfg["upload_model"] = False
    agent_cfg["zero_sync_ppo"] = True
    agent_cfg["zero_sync_rollout"] = True
    agent_cfg["mega_graph_mode"] = "rollout"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
      wrapped, agent_cfg, log_dir=str(tmp_path), device="cuda:0"
    )
    runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
    assert runner._rollout_mega_graph is not None
    assert runner._rollout_mega_graph.captures == 1
    assert runner._rollout_mega_graph.replays >= 22
    assert pool.refill_count >= 1
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_curriculum_stage_transition_recaptures_mega_graph(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_fast
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  env_cfg.scene.num_envs = 32
  env_cfg.sim.mujoco.ccd_iterations = 50
  # Make curriculum stage trigger right at step 12 (between iter 1 and iter 2 with num_steps=8).
  curr_cfg = env_cfg.curriculum["command_vel"]
  curr_cfg.params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
    {"step": 12, "lin_vel_x": (-1.5, 1.5), "ang_vel_z": (-1.2, 1.2)},
  ]
  configure_fp32_fast(env_cfg, enabled=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    env.reset()
    rl_cfg = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    rl_cfg.num_steps_per_env = 8
    rl_cfg.save_interval = 100
    agent_cfg = asdict(rl_cfg)
    agent_cfg["logger"] = "tensorboard"
    agent_cfg["upload_model"] = False
    agent_cfg["zero_sync_ppo"] = True
    agent_cfg["zero_sync_rollout"] = True
    agent_cfg["mega_graph_mode"] = "rollout"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
      wrapped, agent_cfg, log_dir=str(tmp_path), device="cuda:0"
    )
    runner.learn(num_learning_iterations=4, init_at_random_ep_len=False)
    cmd_term = env.command_manager.get_term("twist")
    assert cmd_term.cfg.ranges.lin_vel_x == (-1.5, 1.5)
    assert runner._rollout_mega_graph is not None
    # Initial capture at stage 0 + automatic re-capture once stage 1 activates!
    assert runner._rollout_mega_graph.captures == 2
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cross_mode_checkpoint_roundtrip_and_onnx_export(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import (
    configure_fp32_exact,
    configure_fp32_fast,
  )
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)

  # 1. Train & save in fp32_fast (Tier 2 + ZeroSyncPPO).
  cfg_fast = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  cfg_fast.scene.num_envs = 32
  cfg_fast.sim.mujoco.ccd_iterations = 50
  configure_fp32_fast(cfg_fast, enabled=True)
  env1 = ManagerBasedRlEnv(cfg=cfg_fast, device="cuda:0")
  ckpt_path = tmp_path / "fast_ckpt.pt"
  try:
    env1.reset()
    rl_cfg = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    rl_cfg.num_steps_per_env = 8
    rl_cfg.save_interval = 100
    a1 = asdict(rl_cfg)
    a1["logger"] = "tensorboard"
    a1["upload_model"] = False
    a1["zero_sync_ppo"] = True
    a1["zero_sync_rollout"] = True
    a1["mega_graph_mode"] = "rollout"
    r1 = MjlabOnPolicyRunner(
      RslRlVecEnvWrapper(env1, clip_actions=rl_cfg.clip_actions),
      a1,
      log_dir=str(tmp_path / "run1"),
      device="cuda:0",
    )
    r1.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    r1.save(str(ckpt_path))
    step_saved = env1.common_step_counter
    sd_actor_saved = {
      k: v.detach().cpu().clone() for k, v in r1.alg.actor.state_dict().items()
    }
    r1.export_policy_to_onnx(str(tmp_path / "onnx_fast"), filename="policy.onnx")
    assert (tmp_path / "onnx_fast" / "policy.onnx").exists()
  finally:
    env1.close()

  # 2. Load into strict_rng_parity unfused PPO runner and verify bit-exact weights + counter.
  cfg_parity = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  cfg_parity.scene.num_envs = 32
  cfg_parity.sim.mujoco.ccd_iterations = 50
  configure_fp32_exact(cfg_parity, enabled=True, strict_rng_parity=True)
  env2 = ManagerBasedRlEnv(cfg=cfg_parity, device="cuda:0")
  try:
    env2.reset()
    a2 = asdict(rl_cfg)
    a2["logger"] = "tensorboard"
    a2["upload_model"] = False
    a2["strict_rng_parity"] = True
    a2["zero_sync_ppo"] = False
    a2["zero_sync_rollout"] = False
    a2["mega_graph_mode"] = "off"
    r2 = MjlabOnPolicyRunner(
      RslRlVecEnvWrapper(env2, clip_actions=rl_cfg.clip_actions),
      a2,
      log_dir=str(tmp_path / "run2"),
      device="cuda:0",
    )
    r2.load(str(ckpt_path))
    assert env2.common_step_counter == step_saved
    for k, v in sd_actor_saved.items():
      assert torch.equal(r2.alg.actor.state_dict()[k].detach().cpu(), v), (
        f"Mismatch in {k}"
      )
    # Confirm unfused runner can continue learning smoothly after loading ZeroSyncPPO checkpoint.
    r2.learn(num_learning_iterations=1, init_at_random_ep_len=False)
  finally:
    env2.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_edge_cases_zero_100_resets_nan_latch_2sim_and_video_guards(tmp_path):
  from dataclasses import asdict

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_exact
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
  env_cfg.scene.num_envs = 32
  env_cfg.sim.mujoco.ccd_iterations = 50
  configure_fp32_exact(env_cfg, enabled=True, mega_graph_mode="env")

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  try:
    assert env._mega_step_graph is not None
    env.reset()
    act = torch.zeros(32, env.action_manager.total_action_dim, device="cuda:0")
    # Warmup + capture.
    for _ in range(3):
      env.step(act)

    # 1. Zero-reset step: must NOT emit diluted 0.0 Episode_Reward/* rows.
    env.reset()
    env.begin_rollout_iteration(4)
    env.step(act)
    flushed_zero = env.end_rollout_iteration()
    # Any row emitted with no resets can only contain Curriculum/* keys, never Episode_Reward/*.
    for row in flushed_zero:
      assert not any(k.startswith("Episode_Reward/") for k in row), (
        f"0-reset step diluted Episode_Reward: {row}"
      )

    # 2. 100%-reset step: all 32 envs time out simultaneously -> exact count == 32.0.
    env.begin_rollout_iteration(4)
    env.episode_length_buf.fill_(env.max_episode_length - 1)
    env.step(act)
    flushed_all = env.end_rollout_iteration()
    rew_rows = [r for r in flushed_all if "Episode_Termination/time_out" in r]
    assert len(rew_rows) == 1
    assert rew_rows[0]["Episode_Termination/time_out"] == pytest.approx(32.0)

    # 3. 2SIM_TRAIN hybrid guard must refuse EnvFullStepGraph cleanly.
    env._twosim_hybrid_active = True  # type: ignore[attr-defined]
    ok_2sim, reason_2sim = EnvFullStepGraph.is_supported(env)
    assert not ok_2sim and "2SIM_TRAIN" in reason_2sim
    env._twosim_hybrid_active = False  # type: ignore[attr-defined]

    # 4. NaN guard enabled / auto_reset=False guards must refuse cleanly.
    env.sim.nan_guard.enabled = True
    ok_ng, _ = EnvFullStepGraph.is_supported(env)
    assert not ok_ng
    env.sim.nan_guard.enabled = False

    # 5. Deferred NaN latch in zero-sync rollout must fail closed with ValueError.
    rl_cfg = load_rl_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    rl_cfg.num_steps_per_env = 4
    rl_cfg.save_interval = 100
    a_cfg = asdict(rl_cfg)
    a_cfg["logger"] = "tensorboard"
    a_cfg["upload_model"] = False
    a_cfg["zero_sync_ppo"] = True
    a_cfg["zero_sync_rollout"] = True
    a_cfg["mega_graph_mode"] = "env"
    a_cfg["check_for_nan"] = True

    class _NanInjectingEnv(RslRlVecEnvWrapper):
      def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        rew = rew.clone()
        rew[0] = float("nan")
        return obs, rew, dones, extras

    nan_runner = MjlabOnPolicyRunner(
      _NanInjectingEnv(env, clip_actions=rl_cfg.clip_actions),
      a_cfg,
      log_dir=str(tmp_path / "nan_run"),
      device="cuda:0",
    )
    with pytest.raises(ValueError, match="NaN/Inf"):
      nan_runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    # 6. Custom intermediate wrapper (e.g. VideoRecorder) must prevent Tier 2 bypass!
    class _DummyVideoWrapper:
      def __init__(self, inner):
        self.env = inner
        self.unwrapped = inner

      def __getattr__(self, name):
        return getattr(self.env, name)

    a_vid = asdict(rl_cfg)
    a_vid["logger"] = "tensorboard"
    a_vid["upload_model"] = False
    a_vid["zero_sync_ppo"] = True
    a_vid["zero_sync_rollout"] = True
    a_vid["mega_graph_mode"] = "rollout"
    vid_wrapped = RslRlVecEnvWrapper(
      _DummyVideoWrapper(env),  # type: ignore[arg-type]
      clip_actions=rl_cfg.clip_actions,
    )
    vid_runner = MjlabOnPolicyRunner(
      vid_wrapped, a_vid, log_dir=str(tmp_path / "vid_run"), device="cuda:0"
    )
    vid_runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    assert vid_runner._rollout_mega_graph is None
  finally:
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_env_full_step_graph_matches_stock_physics_without_root_rewind():
  """Regression test: branchless masked writes must preserve untouched envs' exact qpos/qvel."""
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.rollout_optimizations import configure_fp32_exact
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)

  def _make_det_env(use_mega: bool):
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1")
    cfg.scene.num_envs = 16
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.seed = 123
    for k in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
      cfg.events.pop(k, None)
    for g in cfg.observations.values():
      g.enable_corruption = False
    if use_mega:
      configure_fp32_exact(cfg, enabled=True, mega_graph_mode="env")
    else:
      configure_fp32_exact(cfg, enabled=False)
    torch.manual_seed(123)
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

  e_ref = _make_det_env(False)
  e_mg = _make_det_env(True)
  try:
    e_ref.reset()
    e_mg.reset()
    for fld in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl"):
      getattr(e_mg.sim.data, fld).copy_(getattr(e_ref.sim.data, fld))
    for attr in (
      "vel_command_b",
      "time_left",
      "is_standing_env",
      "is_heading_env",
      "heading_target",
    ):
      getattr(e_mg.command_manager.get_term("twist"), attr).copy_(
        getattr(e_ref.command_manager.get_term("twist"), attr)
      )
    e_mg.sim.forward()
    e_ref.sim.forward()

    max_q = max_v = max_r = max_o = 0.0
    for step_i in range(10):
      act = torch.full(
        (16, e_ref.action_manager.total_action_dim),
        0.05 * ((step_i % 3) - 1),
        device="cuda:0",
      )
      o1, r1, t1, tr1, _ = e_ref.step(act)
      o2, r2, t2, tr2, _ = e_mg.step(act)
      assert not (t1.any() or tr1.any() or t2.any() or tr2.any())
      max_q = max(max_q, (e_ref.sim.data.qpos - e_mg.sim.data.qpos).abs().max().item())
      max_v = max(max_v, (e_ref.sim.data.qvel - e_mg.sim.data.qvel).abs().max().item())
      max_r = max(max_r, (r1 - r2).abs().max().item())
      max_o = max(max_o, (o1["actor"] - o2["actor"]).abs().max().item())

    # Before the write_root_*_masked fix, qpos drifted >0.6 and qvel >9.0 in 1 step!
    assert e_mg._mega_step_graph is not None and e_mg._mega_step_graph.captures == 1
    assert e_mg._mega_step_graph.replays == 8
    assert max_q < 1e-4, f"qpos drift too large: {max_q}"
    assert max_v < 1e-2, f"qvel drift too large: {max_v}"
    assert max_r < 2e-2, f"reward drift too large: {max_r}"
    assert max_o < 1e-2, f"obs drift too large: {max_o}"
  finally:
    e_ref.close()
    e_mg.close()
