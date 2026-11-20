"""Unit tests for the opt-in Tiangong3 analytic capsule collision SpecCfg utility."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjlab.utils.tiangong_capsule_collision import (
  TIANGONG3_CAPSULE_EXTRA_EXCLUDES,
  TIANGONG3_CAPSULE_FOOT_GEOM_NAMES,
  TIANGONG3_CAPSULE_PROFILE_VERSION,
  TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES,
  Tiangong3CapsuleCollisionCfg,
  apply_tiangong3_capsule_collisions,
)

_ALL_TIANGONG3_BODIES: tuple[str, ...] = (
  "pelvis",
  "waist_yaw_link",
  "waist_roll_link",
  "waist_pitch_link",
  "imu_waist_link",
  "camera_body_front_link",
  "head_yaw_link",
  "head_pitch_link",
  "imu_head_link",
  "camera_head_link",
  "radar_head_link",
  "hip_pitch_l_link",
  "hip_roll_l_link",
  "hip_yaw_l_link",
  "knee_pitch_l_link",
  "ankle_pitch_l_link",
  "ankle_roll_l_link",
  "hip_pitch_r_link",
  "hip_roll_r_link",
  "hip_yaw_r_link",
  "knee_pitch_r_link",
  "ankle_pitch_r_link",
  "ankle_roll_r_link",
  "shoulder_pitch_l_link",
  "shoulder_roll_l_link",
  "shoulder_yaw_l_link",
  "elbow_pitch_l_link",
  "elbow_yaw_l_link",
  "wrist_pitch_l_link",
  "wrist_roll_l_link",
  "left_tcp_link",
  "shoulder_pitch_r_link",
  "shoulder_roll_r_link",
  "shoulder_yaw_r_link",
  "elbow_pitch_r_link",
  "elbow_yaw_r_link",
  "wrist_pitch_r_link",
  "wrist_roll_r_link",
  "right_tcp_link",
)


def _make_synthetic_tiangong3_mesh_spec() -> mujoco.MjSpec:
  """Construct a lightweight synthetic Tiangong3 skeleton with mesh geoms and explicit inertias."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10.0, 10.0, 0.1],
    condim=3,
  )
  cube_verts = [
    -0.02,
    -0.02,
    -0.02,
    0.02,
    -0.02,
    -0.02,
    0.02,
    0.02,
    -0.02,
    -0.02,
    0.02,
    -0.02,
    -0.02,
    -0.02,
    0.02,
    0.02,
    -0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    -0.02,
    0.02,
    0.02,
  ]

  pelvis = spec.worldbody.add_body(
    name="pelvis",
    pos=[0.0, 0.0, 0.90],
    mass=5.2,
    inertia=[0.04, 0.04, 0.03],
    explicitinertial=1,
  )
  pelvis.add_freejoint(name="floating_base")
  spec.add_mesh(name="pelvis_mesh", uservert=cube_verts)
  pelvis.add_geom(
    name="pelvis_mesh_col",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="pelvis_mesh",
  )

  for idx, body_name in enumerate(_ALL_TIANGONG3_BODIES[1:], start=1):
    b = pelvis.add_body(
      name=body_name,
      pos=[0.0, 0.0, -0.01 * idx],
      mass=0.5 + 0.05 * idx,
      inertia=[0.002, 0.002, 0.001],
      explicitinertial=1,
    )
    mesh_name = f"{body_name}_mesh"
    spec.add_mesh(name=mesh_name, uservert=cube_verts)
    b.add_geom(
      name=f"{body_name}_col",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
    )
    # Also attach a visual-only density=0 mesh to verify full cleanup.
    b.add_geom(
      name=f"{body_name}_vis",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh_name,
      contype=0,
      conaffinity=0,
      density=0.0,
    )

  return spec


def test_capsule_collision_strips_meshes_and_preserves_inertias() -> None:
  baseline_spec = _make_synthetic_tiangong3_mesh_spec()
  baseline_model = baseline_spec.compile()
  assert baseline_model.nmesh == len(_ALL_TIANGONG3_BODIES)

  cap_spec = _make_synthetic_tiangong3_mesh_spec()
  meta = apply_tiangong3_capsule_collisions(cap_spec)
  cap_model = cap_spec.compile()

  assert meta["profile"] == "capsule"
  assert meta["version"] == TIANGONG3_CAPSULE_PROFILE_VERSION
  assert meta["mesh_collision_geoms"] == 0
  assert tuple(meta["foot_collision_geoms"]) == TIANGONG3_CAPSULE_FOOT_GEOM_NAMES  # type: ignore[arg-type]
  assert cap_model.nmesh == 0

  robot_gids = [
    gid for gid in range(cap_model.ngeom) if cap_model.geom(gid).name != "floor"
  ]
  assert len(robot_gids) == 39
  allowed_types = {
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
  }
  assert {int(cap_model.geom_type[gid]) for gid in robot_gids}.issubset(allowed_types)

  # Internal/sensor/gimbal links have 0 collision geoms attached.
  removed_body_ids = {
    cap_model.body(name).id for name in TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES
  }
  for gid in robot_gids:
    assert int(cap_model.geom_bodyid[gid]) not in removed_body_ids

  # Explicit inertial and joint properties remain bit-identical.
  for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
    np.testing.assert_array_equal(
      getattr(baseline_model, field),
      getattr(cap_model, field),
      err_msg=field,
    )

  # Floor (condim=3, priority=0) paired with any robot capsule retains 3D friction.
  floor_id = cap_model.geom("floor").id
  for gid in robot_gids:
    assert int(cap_model.geom_priority[gid]) == 0
    assert (
      max(int(cap_model.geom_condim[gid]), int(cap_model.geom_condim[floor_id])) == 3
    )

  assert cap_model.nexclude >= len(TIANGONG3_CAPSULE_EXTRA_EXCLUDES)


def test_capsule_spec_cfg_validation_and_edit() -> None:
  with pytest.raises(ValueError, match="friction"):
    Tiangong3CapsuleCollisionCfg(friction=(1.0, -0.1, 0.01)).validate()

  spec = _make_synthetic_tiangong3_mesh_spec()
  cfg = Tiangong3CapsuleCollisionCfg(friction=(0.85, 0.02, 0.005))
  cfg.edit_spec(spec)
  model = spec.compile()
  assert model.nmesh == 0
  foot_id = model.geom("foot_l_1_collision").id
  np.testing.assert_allclose(model.geom_friction[foot_id], [0.85, 0.02, 0.005])
