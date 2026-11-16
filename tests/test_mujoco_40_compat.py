"""Tests for MuJoCo 4.0 / MJWarp 4.0 compatibility and CLI backend switching."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import pytest
import torch
import tyro
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

import mjlab
from mjlab._backend_bootstrap import (
  _REEXEC_GUARD_ENV,
  _sanitize_pythonpath_for_target,
  detect_current_backend,
  ensure_mujoco_backend_from_argv,
  get_physics_backend_info,
  parse_mujoco_backend_arg,
  resolve_backend_python,
)
from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
)
from mjlab.managers.event_manager import _DERIVED_FIELDS, RecomputeLevel
from mjlab.scripts.play import PlayConfig
from mjlab.scripts.train import TrainConfig
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")
IS_MUJOCO_4 = mujoco.__version__.startswith("4.")
MJLAB_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def device():
  return get_test_device()


# ---------------------------------------------------------------------------
# 1. CLI Backend Bootstrap Unit & Edge-Case Tests
# ---------------------------------------------------------------------------


def test_parse_mujoco_backend_arg_variants():
  assert parse_mujoco_backend_arg([], {}) == "auto"
  assert parse_mujoco_backend_arg(["--foo", "1"], {}) == "auto"
  assert parse_mujoco_backend_arg(["--mujoco-backend", "4.0"], {}) == "4.0"
  assert parse_mujoco_backend_arg(["--mujoco-backend=4.0-git"], {}) == "4.0-git"
  assert parse_mujoco_backend_arg(["--mj4"], {}) == "4.0-git"
  assert parse_mujoco_backend_arg(["--mujoco-backend=3.11"], {}) == "3.11"
  # Last flag wins when overridden on command line.
  assert (
    parse_mujoco_backend_arg(
      ["--mujoco-backend", "3.11", "--mj4"], {}
    )
    == "4.0-git"
  )
  assert (
    parse_mujoco_backend_arg(
      ["--mujoco-backend", "3.11", "--mujoco-backend=4.0"], {}
    )
    == "4.0"
  )
  # Arguments after "--" separator are ignored.
  assert parse_mujoco_backend_arg(["--", "--mujoco-backend", "4.0"], {}) == "auto"
  # Environment variable fallback and CLI override precedence.
  assert parse_mujoco_backend_arg([], {"MJLAB_MUJOCO_BACKEND": "4.0"}) == "4.0"
  assert (
    parse_mujoco_backend_arg(
      ["--mujoco-backend", "3.11"], {"MJLAB_MUJOCO_BACKEND": "4.0"}
    )
    == "3.11"
  )


@pytest.mark.parametrize(
  "bad_argv,bad_env",
  [
    (["--mujoco-backend"], {}),
    (["--mujoco-backend="], {}),
    (["--mujoco-backend", "5.0"], {}),
    (["--mujoco-backend=invalid"], {}),
    ([], {"MJLAB_MUJOCO_BACKEND": "bogus"}),
  ],
)
def test_parse_mujoco_backend_arg_errors(bad_argv, bad_env):
  with pytest.raises(ValueError):
    parse_mujoco_backend_arg(bad_argv, bad_env)


def test_resolve_backend_python_missing_raises(tmp_path):
  with pytest.raises(RuntimeError, match="not found"):
    resolve_backend_python("4.0", environ={}, mjlab_root=tmp_path)

  with pytest.raises(RuntimeError, match="MJLAB_MUJOCO4_PYTHON"):
    resolve_backend_python(
      "4.0",
      environ={"MJLAB_MUJOCO4_PYTHON": str(tmp_path / "missing_python")},
      mjlab_root=MJLAB_ROOT,
    )


def test_reexec_loop_and_distributed_worker_guards(monkeypatch):
  detected, _ = detect_current_backend()
  other = "3.11" if detected.startswith("4.") else "4.0"

  monkeypatch.setenv(_REEXEC_GUARD_ENV, other)
  with pytest.raises(RuntimeError, match="Re-executed into"):
    ensure_mujoco_backend_from_argv(["--mujoco-backend", other], _allow_exec=False)

  monkeypatch.delenv(_REEXEC_GUARD_ENV, raising=False)
  monkeypatch.setenv("RANK", "0")
  with pytest.raises(RuntimeError, match="Distributed worker"):
    ensure_mujoco_backend_from_argv(["--mujoco-backend", other], _allow_exec=False)


def test_sanitize_pythonpath_strips_mismatched_ros():
  py_path = Path(sys.executable)
  raw = os.pathsep.join(
    [
      "/opt/ros/humble/lib/python3.10/site-packages",
      "/workspace/custom_pkg",
    ]
  )
  cleaned = _sanitize_pythonpath_for_target(py_path, raw)
  assert "/workspace/custom_pkg" in cleaned
  if f"python{sys.version_info.major}.{sys.version_info.minor}" != "python3.10":
    assert "/opt/ros/humble/lib/python3.10/site-packages" not in cleaned


def test_cross_venv_reexec_subprocesses(tmp_path):
  venv3 = MJLAB_ROOT / ".venv" / "bin" / "python"
  venv4 = MJLAB_ROOT / ".venv-mj4" / "bin" / "python"
  if not (venv3.is_file() and venv4.is_file()):
    pytest.skip("Both .venv and .venv-mj4 are required for cross-venv execv test")

  probe = tmp_path / "probe_backend.py"
  probe.write_text(
    "import json, mjlab\n"
    "from mjlab._backend_bootstrap import get_physics_backend_info\n"
    "import mujoco\n"
    "info = get_physics_backend_info()\n"
    "info['loaded_mujoco'] = mujoco.__version__\n"
    "print(json.dumps(info))\n"
  )
  clean_env = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith("_MJLAB_BACKEND_") and k != "MJLAB_MUJOCO_BACKEND"
  }
  clean_env["PYTHONPATH"] = ""

  # 1) Default from .venv stays on 3.11 without re-exec.
  out_default = json.loads(
    subprocess.check_output([str(venv3), str(probe)], env=clean_env, text=True)
  )
  assert out_default["resolved_backend"] == "3.11"
  assert out_default["loaded_mujoco"].startswith("3.11.")
  assert out_default["reexecuted"] is False

  # 2) Passing --mujoco-backend 4.0 to .venv re-execs into 4.0.0.
  out_40 = json.loads(
    subprocess.check_output(
      [str(venv3), str(probe), "--mujoco-backend", "4.0"],
      env=clean_env,
      text=True,
    )
  )
  assert out_40["resolved_backend"] == "4.0"
  assert out_40["loaded_mujoco"].startswith("4.")
  assert out_40["reexecuted"] is True

  # 3) Passing --mujoco-backend=3.11 to .venv-mj4 switches back to 3.11.0.
  out_311 = json.loads(
    subprocess.check_output(
      [str(venv4), str(probe), "--mujoco-backend=3.11"],
      env=clean_env,
      text=True,
    )
  )
  assert out_311["resolved_backend"] == "3.11"
  assert out_311["loaded_mujoco"].startswith("3.11.")
  assert out_311["reexecuted"] is True

  # 4) Optional 4.0-git check when .venv-mj4-git is present.
  venv4_git = MJLAB_ROOT / ".venv-mj4-git" / "bin" / "python"
  if venv4_git.is_file():
    out_git = json.loads(
      subprocess.check_output(
        [str(venv3), str(probe), "--mujoco-backend", "4.0-git"],
        env=clean_env,
        text=True,
      )
    )
    assert out_git["resolved_backend"] == "4.0-git"
    assert out_git["mujoco_warp_git_editable"] is True
    assert out_git["reexecuted"] is True

  # 5) Invalid backend exits with status 2.
  bad = subprocess.run(
    [str(venv3), str(probe), "--mujoco-backend", "not-a-backend"],
    env=clean_env,
    capture_output=True,
    text=True,
  )
  assert bad.returncode == 2
  assert "Invalid MuJoCo backend" in bad.stderr


# ---------------------------------------------------------------------------
# 2. Tyro CLI Parsing Compatibility (AvoidSubcommands)
# ---------------------------------------------------------------------------


def test_tyro_cli_parses_new_sim_and_backend_flags():
  from mjlab import tasks as _mjlab_tasks  # noqa: F401

  cfg = tyro.cli(
    TrainConfig,
    args=[
      "--mujoco-backend",
      "4.0",
      "--env.sim.deterministic",
      "all",
      "--env.sim.run-qacc-smooth",
      "False",
      "--env.sim.warn-overflow",
      "False",
      "--env.sim.nccdmax",
      "32",
      "--env.sim.njmax-nnz",
      "4096",
      "--env.sim.mujoco.integrator",
      "implicit",
      "--env.sim.mujoco.noslip-iterations",
      "4",
      "--env.sim.mujoco.ccd-tolerance",
      "1e-5",
    ],
    default=TrainConfig.from_task("Mjlab-Velocity-Flat-Unitree-G1"),
    config=mjlab.TYRO_FLAGS,
  )
  assert cfg.mujoco_backend == "4.0"
  assert cfg.env.sim.deterministic == "all"
  assert cfg.env.sim.run_qacc_smooth is False
  assert cfg.env.sim.warn_overflow is False
  assert cfg.env.sim.nccdmax == 32
  assert cfg.env.sim.njmax_nnz == 4096
  assert cfg.env.sim.mujoco.integrator == "implicit"
  assert cfg.env.sim.mujoco.noslip_iterations == 4
  assert cfg.env.sim.mujoco.ccd_tolerance == pytest.approx(1e-5)

  play_cfg = tyro.cli(
    PlayConfig,
    args=["--mujoco-backend", "4.0-git", "--agent", "zero"],
    default=PlayConfig(),
    config=mjlab.TYRO_FLAGS,
  )
  assert play_cfg.mujoco_backend == "4.0-git"


# ---------------------------------------------------------------------------
# 3. SimulationCfg, MujocoCfg, and RecomputeLevel Compatibility
# ---------------------------------------------------------------------------


def test_eq_data_auto_expanded_for_set_const_when_neq_positive(device):
  eq_xml = """
  <mujoco>
    <worldbody>
      <body name="b1" pos="0 0 1">
        <freejoint/>
        <geom size="0.1"/>
      </body>
      <body name="b2" pos="0 0 0.5">
        <freejoint/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <equality>
      <connect body1="b1" body2="b2" anchor="0 0 0"/>
    </equality>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(eq_xml)
  sim = Simulation(num_envs=3, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(_DERIVED_FIELDS[RecomputeLevel.set_const_0])
  if IS_MUJOCO_4:
    assert "eq_data" in sim.expanded_fields
    assert sim.wp_model.eq_data.shape[0] == 3
  sim.recompute_constants(RecomputeLevel.set_const_0)


def test_recompute_constants_emits_no_deprecation_warnings(device):
  model = mujoco.MjModel.from_xml_string(ROBOT_XML)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(
    ("body_mass", "body_subtreemass") + _DERIVED_FIELDS[RecomputeLevel.set_const_0]
  )
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    sim.recompute_constants(RecomputeLevel.set_const_fixed)
    sim.recompute_constants(RecomputeLevel.set_const_0)
    sim.recompute_constants(RecomputeLevel.set_const)
  dep_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
  assert not dep_msgs, f"Unexpected DeprecationWarnings: {dep_msgs}"


def test_warn_overflow_mutually_exclusive_validation():
  with pytest.raises(ValueError, match="warn_overflow"):
    SimulationCfg(warn_overflow=False, warn_overflow_flags=("nefc",))


def test_version_specific_sim_options_and_errors(device):
  if IS_MUJOCO_4:
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    sim_cfg = SimulationCfg(
      deterministic="contacts_constraint",
      run_qacc_smooth=False,
      run_rne_postconstraint=False,
      warn_overflow_flags=("nefc", "ccd"),
      mujoco=MujocoCfg(integrator="implicitfast", noslip_iterations=2),
    )
    sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
    expected_det = int(
      mjwarp.DeterminismType.CONTACTS | mjwarp.DeterminismType.CONSTRAINT
    )
    assert sim.wp_model.opt.deterministic == expected_det
    assert sim.wp_model.opt.run_qacc_smooth is False
    assert sim.wp_model.opt.run_rne_postconstraint is False
    expected_ovf = int(mjwarp.OverflowType.NEFC | mjwarp.OverflowType.CCD)
    assert sim.wp_model.opt.warn_overflow == expected_ovf
    sim.step()
    assert torch.isfinite(sim.data.qpos).all()

    # Discrete integrator works standalone, and rejects noslip_iterations > 0.
    disc_model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    disc_sim = Simulation(
      num_envs=2,
      cfg=SimulationCfg(mujoco=MujocoCfg(integrator="discrete")),
      model=disc_model,
      device=device,
    )
    disc_sim.step()
    assert torch.isfinite(disc_sim.data.qpos).all()

    with pytest.raises(NotImplementedError, match="noslip"):
      Simulation(
        num_envs=1,
        cfg=SimulationCfg(
          mujoco=MujocoCfg(integrator="discrete", noslip_iterations=2)
        ),
        model=mujoco.MjModel.from_xml_string(ROBOT_XML),
        device=device,
      )

    # Speculative contacts in MuJoCo 4.0 C are unsupported in GPU mujoco_warp.
    spec_model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    with pytest.raises(NotImplementedError, match="speculative contacts"):
      Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(enableflags=("speculative",))),
        model=spec_model,
        device=device,
      )
  else:
    for bad_cfg in (
      SimulationCfg(deterministic="all"),
      SimulationCfg(run_qacc_smooth=False),
      SimulationCfg(run_rne_postconstraint=False),
      SimulationCfg(warn_overflow_flags=("nefc",)),
      SimulationCfg(mujoco=MujocoCfg(integrator="discrete")),
    ):
      m = mujoco.MjModel.from_xml_string(ROBOT_XML)
      with pytest.raises(ValueError, match="4.0"):
        Simulation(num_envs=1, cfg=bad_cfg, model=m, device=device)


# ---------------------------------------------------------------------------
# 4. Dual-Version BuiltinDcMotorActuator Tests
# ---------------------------------------------------------------------------


def test_dcmotor_dual_version_modes_and_physics(device):
  datasheet = DcMotorDatasheetParams(
    nominal_voltage=24.0, stall_torque=2.0, no_load_speed=100.0
  )
  K = 24.0 / 100.0
  R = K * 24.0 / 2.0

  # Position mode works in both 3.11 and 4.0 (voltage-space vs torque-space PID).
  pos_cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.POSITION,
    motor_params=datasheet,
    stiffness=5.0,
    damping=0.0,
    voltage_limit=1000.0,
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, pos_cfg), device)
  zero = torch.zeros(1, 2, device=device)
  pos_target = torch.tensor([[0.1, -0.05]], device=device)
  entity.write_joint_state_to_sim(position=zero, velocity=zero)
  entity.set_joint_position_target(pos_target)
  entity.set_joint_velocity_target(zero)
  entity.set_joint_effort_target(zero)
  entity.write_data_to_sim()
  sim.forward()

  v_adr = entity.indexing.joint_v_adr
  if IS_MUJOCO_4:
    assert sim.mj_model.actuator_ctrlspec[0] == int(mujoco.mjtCtrlInput.mjINPUT_POS)
    expected_tau = 5.0 * pos_target[0]  # torque-space kp in MuJoCo >= 3.14 / 4.0
  else:
    assert sim.mj_model.actuator_gainprm[0, 8] == pytest.approx(1.0)
    expected_tau = K * 5.0 * pos_target[0] / R  # voltage-space kp in MuJoCo 3.11
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected_tau, atol=1e-4)

  # Velocity mode validation and compilation in active backend.
  if IS_MUJOCO_4:
    vel_cfg = BuiltinDcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      mode=DcMotorInputMode.VELOCITY,
      motor_params=datasheet,
      stiffness=0.0,
      damping=2.0,
      voltage_limit=24.0,
    )
    _, vel_sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, vel_cfg), device)
    assert vel_sim.mj_model.actuator_ctrlspec[0] == int(mujoco.mjtCtrlInput.mjINPUT_VEL)
  else:
    vel_cfg = BuiltinDcMotorActuatorCfg(
      target_names_expr=("joint.*",),
      mode=DcMotorInputMode.VELOCITY,
      motor_params=datasheet,
      stiffness=5.0,
      damping=0.0,
      voltage_limit=24.0,
    )
    _, vel_sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, vel_cfg), device)
    assert vel_sim.mj_model.actuator_gainprm[0, 8] == pytest.approx(2.0)

  info = get_physics_backend_info()
  assert info["mujoco_version"].startswith(mujoco.__version__)
