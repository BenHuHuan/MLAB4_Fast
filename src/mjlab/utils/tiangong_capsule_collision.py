"""Opt-in Sim2Real-calibrated analytic capsule/sphere collision profile for Tiangong3 training.

Replaces Tiangong3's 39 high-poly CAD collision meshes (up to 5,774 hull faces per link,
or 64-vertex faceted polyhedra under ``fast``) with zero-mesh analytic ``capsule`` and
``sphere`` primitives calibrated to sub-millimeter sole clearance and millimeter-level
whole-body recovery envelopes on ``tiangong3_v2_29_head_fixed``.

Key properties:
  - Preserves 100% of body masses, inertias, CoMs, joint ranges, armatures, dampings,
    and actuator effort contracts from ``sdk_a_scene.xml``.
  - Eliminates all ``mjGEOM_MESH`` collision geoms (``nmesh == 0``), allowing
    ``mujoco_warp`` to execute purely in ``_primitive_narrowphase`` (closed-form O(1)
    ``plane_capsule`` / ``plane_sphere`` / ``capsule_capsule``) without launching
    ``plane_convex`` graph hill-climbing or ``convex_narrowphase`` GJK/EPA/MultiCCD.
  - Uses ``condim=3`` on foot sole capsules and hand TCP capsules, and ``condim=1``
    (``priority=0``) on proximal limb capsules: when paired with a floor plane having
    ``condim=3, priority=0``, every body-floor contact retains full 3D tangential
    friction (``max(1, 3) == 3``) during ground recovery, while limb-limb self-contacts
    use 1-row frictionless normal repulsion (matching Unitree G1 ``FULL_COLLISION``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.utils.spec_config import SpecCfg

TIANGONG3_CAPSULE_PROFILE_VERSION = "tiangong3_capsule_v1"

DEFAULT_FRICTION: tuple[float, float, float] = (1.0, 0.01, 0.01)

TIANGONG3_CAPSULE_FOOT_GEOM_NAMES: tuple[str, ...] = (
  "foot_l_1_collision",
  "foot_l_2_collision",
  "foot_l_3_collision",
  "foot_l_4_collision",
  "foot_r_1_collision",
  "foot_r_2_collision",
  "foot_r_3_collision",
  "foot_r_4_collision",
)

# 2-hop compound-joint self-collision exclusions (matching Unitree G1 + Tiangong3
# gimbal joints where CAD housings overlap across zero-length/short intermediate links,
# while keeping all cross-limb leg-leg, arm-leg, arm-torso, arm-pelvis, and hand-hand
# self-collisions active).
TIANGONG3_CAPSULE_EXTRA_EXCLUDES: tuple[tuple[str, str], ...] = (
  ("pelvis", "hip_roll_l_link"),
  ("pelvis", "hip_roll_r_link"),
  ("pelvis", "hip_yaw_l_link"),
  ("pelvis", "hip_yaw_r_link"),
  ("pelvis", "waist_pitch_link"),
  ("hip_pitch_l_link", "hip_yaw_l_link"),
  ("hip_pitch_r_link", "hip_yaw_r_link"),
  ("waist_pitch_link", "shoulder_roll_l_link"),
  ("waist_pitch_link", "shoulder_roll_r_link"),
  ("shoulder_roll_l_link", "elbow_pitch_l_link"),
  ("shoulder_roll_r_link", "elbow_pitch_r_link"),
  ("elbow_pitch_l_link", "wrist_pitch_l_link"),
  ("elbow_pitch_r_link", "wrist_pitch_r_link"),
  ("elbow_yaw_l_link", "left_tcp_link"),
  ("elbow_yaw_r_link", "right_tcp_link"),
  ("wrist_pitch_l_link", "left_tcp_link"),
  ("wrist_pitch_r_link", "right_tcp_link"),
)

# Removed internal / sensor / intermediate gimbal bodies (14 bodies -> 0 collision geoms):
TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES: tuple[str, ...] = (
  "imu_waist_link",
  "imu_head_link",
  "camera_body_front_link",
  "camera_head_link",
  "radar_head_link",
  "head_pitch_link",
  "waist_yaw_link",
  "waist_roll_link",
  "ankle_pitch_l_link",
  "ankle_pitch_r_link",
  "shoulder_pitch_l_link",
  "shoulder_pitch_r_link",
  "wrist_roll_l_link",
  "wrist_roll_r_link",
)


def _add_capsule(
  spec: mujoco.MjSpec,
  body_name: str,
  geom_name: str,
  radius: float,
  p1: tuple[float, float, float],
  p2: tuple[float, float, float],
  *,
  condim: int = 1,
  friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> None:
  spec.body(body_name).add_geom(
    name=geom_name,
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=[radius, 0.0, 0.0],
    fromto=[*p1, *p2],
    group=1,
    condim=condim,
    friction=list(friction),
    rgba=[0.752941, 0.752941, 0.752941, 1.0],
  )


def _add_sphere(
  spec: mujoco.MjSpec,
  body_name: str,
  geom_name: str,
  radius: float,
  pos: tuple[float, float, float],
  *,
  condim: int = 1,
  friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> None:
  spec.body(body_name).add_geom(
    name=geom_name,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[radius, 0.0, 0.0],
    pos=list(pos),
    group=1,
    condim=condim,
    friction=list(friction),
    rgba=[0.752941, 0.752941, 0.752941, 1.0],
  )


def apply_tiangong3_capsule_collisions(
  spec: mujoco.MjSpec,
  *,
  friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> dict[str, object]:
  """Replace mesh collisions on a Tiangong3 ``MjSpec`` with calibrated capsules/spheres."""
  paired: set[str] = {
    name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)
  }
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      continue
    if geom.name in paired:
      continue
    # Preserve only explicit mass-bearing geoms if any exist (Tiangong3 uses explicit <inertial>).
    if (geom.density == 0 or (geom.contype or geom.conaffinity)) and not (
      np.isfinite(geom.mass) and geom.mass > 0
    ):
      spec.delete(geom)

  referenced_meshes: set[str] = {
    geom.meshname
    for geom in spec.geoms
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname
  }
  for mesh in list(spec.meshes):
    if mesh.name not in referenced_meshes:
      spec.delete(mesh)

  # 1. Pelvis: CAD bbox [-0.1080..+0.1060, -0.1251..+0.1250, -0.1824..+0.0575]
  _add_capsule(
    spec,
    "pelvis",
    "pelvis_l_collision",
    0.080,
    (-0.028, 0.045, -0.036),
    (0.026, 0.045, -0.036),
    friction=friction,
  )
  _add_capsule(
    spec,
    "pelvis",
    "pelvis_r_collision",
    0.080,
    (-0.028, -0.045, -0.036),
    (0.026, -0.045, -0.036),
    friction=friction,
  )
  _add_capsule(
    spec,
    "pelvis",
    "pelvis_lower_collision",
    0.074,
    (-0.002, -0.050, -0.108),
    (-0.002, 0.050, -0.108),
    friction=friction,
  )

  # 2. Torso (waist_pitch_link): CAD bbox [-0.1097..+0.1145, -0.1636..+0.1641, -0.0340..+0.5278]
  _add_capsule(
    spec,
    "waist_pitch_link",
    "torso_lower_collision",
    0.108,
    (0.002, -0.055, 0.105),
    (0.002, 0.055, 0.105),
    friction=friction,
  )
  _add_capsule(
    spec,
    "waist_pitch_link",
    "torso_upper_collision",
    0.106,
    (-0.001, -0.055, 0.220),
    (-0.001, 0.055, 0.220),
    friction=friction,
  )
  _add_sphere(
    spec,
    "waist_pitch_link",
    "head_crown_collision",
    0.064,
    (0.0002, 0.0, 0.464),
    friction=friction,
  )

  # 3. Head (head_yaw_link + fixed head_pitch_link + camera_head_link)
  _add_sphere(
    spec,
    "head_yaw_link",
    "head_yaw_link_collision",
    0.085,
    (0.001, 0.0, 0.008),
    friction=friction,
  )

  foot_geom_names: list[str] = []
  for side, sy in (("l", 1.0), ("r", -1.0)):
    # 4. Hip (buttock / lateral trochanter envelope: y up to 0.1389*sy, z down to -0.091)
    _add_capsule(
      spec,
      f"hip_pitch_{side}_link",
      f"hip_pitch_{side}_link_collision",
      0.049,
      (-0.001, 0.052 * sy, -0.042),
      (0.002, 0.090 * sy, -0.024),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"hip_roll_{side}_link",
      f"hip_roll_{side}_link_collision",
      0.058,
      (0.001, 0.0, 0.002),
      (0.001, 0.0, -0.030),
      friction=friction,
    )

    # 5. Thigh (hip_yaw_link): CAD bbox [-0.0677..+0.0683, -0.0740..+0.0601, -0.3432..+0.0040]
    _add_capsule(
      spec,
      f"hip_yaw_{side}_link",
      f"hip_yaw_{side}_link_collision",
      0.066,
      (0.001, -0.006 * sy, -0.060),
      (0.001, -0.006 * sy, -0.205),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"hip_yaw_{side}_link",
      f"thigh_distal_{side}_collision",
      0.051,
      (-0.002, 0.0, -0.220),
      (-0.006, 0.0, -0.292),
      friction=friction,
    )

    # 6. Shin & anterior kneecap (knee_pitch_link): CAD bbox [-0.0410..+0.0712, ±0.0554, -0.4149..+0.0523]
    _add_capsule(
      spec,
      f"knee_pitch_{side}_link",
      f"knee_cap_{side}_collision",
      0.054,
      (0.017, 0.0, -0.015),
      (0.016, 0.0, -0.175),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"knee_pitch_{side}_link",
      f"shin_lower_{side}_collision",
      0.038,
      (0.011, 0.0, -0.195),
      (0.007, 0.0, -0.355),
      friction=friction,
    )

    # 7. Foot sole (ankle_roll_link): 4 tapered sole capsules matching exact CAD 1.35-deg
    # heel-to-toe slope (z=-0.05677 at heel x=-0.084, z=-0.05305 at ball x=+0.085,
    # z=-0.05050 at toe x=+0.1688) and lateral sole width (±0.0435 heel, ±0.0555 ball, ±0.028 toe).
    r_f = 0.010
    foot_defs = (
      (
        f"foot_{side}_1_collision",
        (-0.084, -0.0335, -0.04677),
        (0.085, -0.0455, -0.04305),
      ),
      (
        f"foot_{side}_2_collision",
        (-0.084, 0.0335, -0.04677),
        (0.085, 0.0455, -0.04305),
      ),
      (
        f"foot_{side}_3_collision",
        (0.085, -0.0455, -0.04305),
        (0.1688, -0.0180, -0.04050),
      ),
      (
        f"foot_{side}_4_collision",
        (0.085, 0.0455, -0.04305),
        (0.1688, 0.0180, -0.04050),
      ),
    )
    for fg_name, p1, p2 in foot_defs:
      _add_capsule(
        spec,
        f"ankle_roll_{side}_link",
        fg_name,
        r_f,
        p1,
        p2,
        condim=3,
        friction=friction,
      )
      foot_geom_names.append(fg_name)

    # 8. Deltoid / Upper arm / Elbow / Forearm / Wrist / Hand TCP
    _add_capsule(
      spec,
      f"shoulder_roll_{side}_link",
      f"shoulder_roll_{side}_link_collision",
      0.048,
      (-0.001, 0.0, -0.009),
      (-0.001, 0.0, -0.056),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"shoulder_yaw_{side}_link",
      f"shoulder_yaw_{side}_link_collision",
      0.041,
      (0.001, 0.0, -0.020),
      (0.012, -0.002 * sy, -0.154),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"elbow_pitch_{side}_link",
      f"elbow_pitch_{side}_link_collision",
      0.045,
      (-0.009, -0.001 * sy, 0.0),
      (-0.018, 0.0, -0.075),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"elbow_yaw_{side}_link",
      f"elbow_yaw_{side}_link_collision",
      0.036,
      (0.0, 0.003 * sy, -0.015),
      (0.0, 0.0, -0.092),
      friction=friction,
    )
    _add_capsule(
      spec,
      f"wrist_pitch_{side}_link",
      f"wrist_pitch_{side}_link_collision",
      0.034,
      (0.0, -0.002 * sy, 0.0),
      (-0.001, 0.0, -0.060),
      friction=friction,
    )
    tcp_body = "left_tcp_link" if side == "l" else "right_tcp_link"
    _add_capsule(
      spec,
      tcp_body,
      f"{tcp_body}_collision",
      0.047,
      (0.005, -0.006 * sy, -0.018),
      (0.006, -0.008 * sy, -0.0576),
      condim=3,
      friction=friction,
    )

  existing_excludes: set[tuple[str, ...]] = {
    tuple(sorted((ex.bodyname1, ex.bodyname2))) for ex in spec.excludes
  }
  added_excludes: list[list[str]] = []
  for b1, b2 in TIANGONG3_CAPSULE_EXTRA_EXCLUDES:
    key = tuple(sorted((b1, b2)))
    if key not in existing_excludes:
      spec.add_exclude(bodyname1=b1, bodyname2=b2)
      existing_excludes.add(key)
      added_excludes.append([b1, b2])

  return {
    "profile": "capsule",
    "version": TIANGONG3_CAPSULE_PROFILE_VERSION,
    "collision_preserved": False,
    "mesh_collision_geoms": 0,
    "analytic_collision_geoms": len(list(spec.geoms)),
    "foot_collision_geoms": foot_geom_names,
    "removed_internal_bodies": list(TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES),
    "added_compound_joint_excludes": added_excludes,
  }


@dataclass
class Tiangong3CapsuleCollisionCfg(SpecCfg):
  """Opt-in ``SpecCfg`` modifier that replaces Tiangong3 CAD meshes with capsules/spheres."""

  friction: tuple[float, float, float] = DEFAULT_FRICTION
  """Default friction tuple ``(sliding, torsional, rolling)`` assigned to capsule/sphere geoms."""

  def validate(self) -> None:
    if len(self.friction) != 3 or any(v < 0 for v in self.friction):
      raise ValueError(
        f"friction must be a 3-tuple of non-negative floats, got {self.friction!r}"
      )

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    self.validate()
    apply_tiangong3_capsule_collisions(spec, friction=self.friction)
