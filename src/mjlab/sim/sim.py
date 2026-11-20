from __future__ import annotations

import gc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.entity.variants import VARIANT_DEPENDENT_FIELDS, build_variant_model
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import TorchArray, WarpBridge
from mjlab.utils.nan_guard import NanGuard, NanGuardCfg

_mjwarp_actuator_velocity: Any = None
_mjwarp_tendon_velocity: Any = None
try:
  from mujoco_warp._src.forward import (
    _actuator_velocity as _mw_actuator_velocity,
  )
  from mujoco_warp._src.forward import (
    _tendon_velocity as _mw_tendon_velocity,
  )

  _mjwarp_actuator_velocity = _mw_actuator_velocity
  _mjwarp_tendon_velocity = _mw_tendon_velocity
except Exception:
  pass

_mjwarp_sensor_pos: Any = None
_mjwarp_sensor_vel: Any = None
_mjwarp_subtree_vel: Any = None
try:
  from mujoco_warp._src.sensor import (
    _sensor_pos as _mw_sensor_pos,
  )
  from mujoco_warp._src.sensor import (
    _sensor_vel as _mw_sensor_vel,
  )
  from mujoco_warp._src.smooth import subtree_vel as _mw_subtree_vel

  _mjwarp_sensor_pos = _mw_sensor_pos
  _mjwarp_sensor_vel = _mw_sensor_vel
  _mjwarp_subtree_vel = _mw_subtree_vel
except Exception:
  pass

if TYPE_CHECKING:
  from mujoco import MjModel as _MjModel

  from mjlab.entity.variants import VariantMetadata
  from mjlab.sensor.sensor_context import SensorContext

# Type aliases for better IDE support while maintaining runtime compatibility
# At runtime, WarpBridge wraps the actual MJWarp objects.
if TYPE_CHECKING:
  ModelBridge = mjwarp.Model
  DataBridge = mjwarp.Data
else:
  ModelBridge = WarpBridge
  DataBridge = WarpBridge

# Minimum CUDA driver version supported for conditional CUDA graphs.
_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)


@contextmanager
def _suspend_gc():
  """Temporarily disable the garbage collector.

  Prevents GC from finalizing stale Warp Graph objects during wp.ScopedCapture, which
  would record their destructor calls into the new graph and corrupt it on replay.
  """
  enabled = gc.isenabled()
  gc.disable()
  try:
    yield
  finally:
    if enabled:
      gc.enable()


_JACOBIAN_MAP = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_CONE_MAP = {
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_INTEGRATOR_MAP: dict[str, mujoco.mjtIntegrator] = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
  "rk4": mujoco.mjtIntegrator.mjINT_RK4,
}
if hasattr(mujoco.mjtIntegrator, "mjINT_DISCRETE"):
  _INTEGRATOR_MAP["discrete"] = mujoco.mjtIntegrator.mjINT_DISCRETE
_SOLVER_MAP = {
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
}
_BROADPHASE_MAP = {
  "nxn": mjwarp.BroadphaseType.NXN,
  "sap_tile": mjwarp.BroadphaseType.SAP_TILE,
  "sap_segmented": mjwarp.BroadphaseType.SAP_SEGMENTED,
}
_BROADPHASE_FILTER_MAP = {
  "plane": mjwarp.BroadphaseFilter.PLANE,
  "sphere": mjwarp.BroadphaseFilter.SPHERE,
  "aabb": mjwarp.BroadphaseFilter.AABB,
  "obb": mjwarp.BroadphaseFilter.OBB,
}
_HAS_DETERMINISM_TYPE = hasattr(mjwarp, "DeterminismType")
_DETERMINISM_MAP: dict[str, int] = (
  {
    "none": int(mjwarp.DeterminismType.NONE),
    "contacts": int(mjwarp.DeterminismType.CONTACTS),
    "constraint": int(mjwarp.DeterminismType.CONSTRAINT),
    "atomics": int(mjwarp.DeterminismType.ATOMICS),
    "contacts_constraint": int(
      mjwarp.DeterminismType.CONTACTS | mjwarp.DeterminismType.CONSTRAINT
    ),
    "all": int(mjwarp.DeterminismType.ALL),
  }
  if _HAS_DETERMINISM_TYPE
  else {}
)
_HAS_OVERFLOW_TYPE = hasattr(mjwarp, "OverflowType")
_IS_MUJOCO_4 = str(getattr(mujoco, "__version__", "")).startswith("4.")
_OVERFLOW_FLAG_MAP: dict[str, int] = (
  {
    name.lower(): int(member)
    for name, member in mjwarp.OverflowType.__members__.items()
    if name not in ("NONE", "ALL")
  }
  if _HAS_OVERFLOW_TYPE
  else {}
)
_HAS_SET_CONST_GROUP = hasattr(mjwarp, "SetConstGroup")

DeterminismChoice = Literal[
  "none", "contacts", "constraint", "atomics", "contacts_constraint", "all"
]
OverflowFlagChoice = Literal[
  "nefc",
  "njmax_nnz",
  "broadphase",
  "narrowphase",
  "ccd",
  "hfield",
  "contact_match",
  "nvmax",
  "epa_horizon",
  "iterations",
  "ls_iterations",
  "tactile",
]

# Maps short flag names to MuJoCo enum values.
# Names match the XML <flag> attribute names (e.g. <flag contact="disable"/>).
_DISABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjDSBL_").lower(): getattr(mujoco.mjtDisableBit, name).value
  for name in dir(mujoco.mjtDisableBit)
  if name.startswith("mjDSBL_")
}
_ENABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjENBL_").lower(): getattr(mujoco.mjtEnableBit, name).value
  for name in dir(mujoco.mjtEnableBit)
  if name.startswith("mjENBL_")
}


@dataclass
class MujocoCfg:
  """Configuration for MuJoCo simulation parameters."""

  # Integrator settings.
  timestep: float = 0.002
  integrator: Literal["euler", "implicitfast", "implicit", "rk4", "discrete"] = (
    "implicitfast"
  )

  # Friction settings.
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"

  # Solver settings.
  jacobian: Literal["auto", "dense", "sparse"] = "auto"
  solver: Literal["newton", "cg", "pgs"] = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  ccd_iterations: int = 50
  ccd_tolerance: float | None = None
  noslip_iterations: int | None = None
  noslip_tolerance: float | None = None
  sdf_iterations: int | None = None
  sdf_initpoints: int | None = None

  # Other.
  gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
  # Global MuJoCo option flags. Names match the XML <flag> attributes
  # (e.g. "contact", "gravity", "sensor"). See mjtDisableBit / mjtEnableBit.
  disableflags: tuple[str, ...] = ()
  """Disable flags to set (e.g. ``("contact",)`` to disable contacts)."""
  enableflags: tuple[str, ...] = ()
  """Enable flags to set (e.g. ``("energy",)`` to enable energy computation)."""

  def apply(self, model: mujoco.MjModel) -> None:
    """Apply configuration settings to a compiled MjModel."""
    if self.integrator not in _INTEGRATOR_MAP:
      raise ValueError(
        f"Integrator {self.integrator!r} requires MuJoCo >= 4.0 "
        f"(active MuJoCo={mujoco.__version__}, supported={sorted(_INTEGRATOR_MAP)})."
      )
    model.opt.jacobian = _JACOBIAN_MAP[self.jacobian]
    model.opt.cone = _CONE_MAP[self.cone]
    model.opt.integrator = _INTEGRATOR_MAP[self.integrator]
    model.opt.solver = _SOLVER_MAP[self.solver]
    model.opt.timestep = self.timestep
    model.opt.impratio = self.impratio
    model.opt.gravity[:] = self.gravity
    model.opt.iterations = self.iterations
    model.opt.tolerance = self.tolerance
    model.opt.ls_iterations = self.ls_iterations
    model.opt.ls_tolerance = self.ls_tolerance
    model.opt.ccd_iterations = self.ccd_iterations
    if self.ccd_tolerance is not None:
      model.opt.ccd_tolerance = self.ccd_tolerance
    if self.noslip_iterations is not None:
      model.opt.noslip_iterations = self.noslip_iterations
    if self.noslip_tolerance is not None:
      model.opt.noslip_tolerance = self.noslip_tolerance
    if self.sdf_iterations is not None:
      model.opt.sdf_iterations = self.sdf_iterations
    if self.sdf_initpoints is not None:
      model.opt.sdf_initpoints = self.sdf_initpoints
    for flag in self.disableflags:
      if flag not in _DISABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown disable flag {flag!r}. Valid flags: {sorted(_DISABLE_FLAG_MAP)}"
        )
      model.opt.disableflags |= _DISABLE_FLAG_MAP[flag]
    for flag in self.enableflags:
      if flag not in _ENABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown enable flag {flag!r}. Valid flags: {sorted(_ENABLE_FLAG_MAP)}"
        )
      model.opt.enableflags |= _ENABLE_FLAG_MAP[flag]


@dataclass(kw_only=True)
class SimulationCfg:
  nconmax: int | None = None
  """Number of contacts to allocate per world.

  Contacts exist in large heterogenous arrays: one world may have more than nconmax
  contacts. If None, a heuristic value is used."""
  njmax: int | None = None
  """Number of constraints to allocate per world.

  Constraint arrays are batched by world: no world may have more than njmax
  constraints. If None, a heuristic value is used."""
  nccdmax: int | None = None
  """Max number of CCD contacts per world forwarded to ``mjwarp.put_data``."""
  njmax_nnz: int | None = None
  """Max non-zeros in sparse constraint Jacobian forwarded to ``mjwarp.put_data``."""
  naconmax: int | None = None
  """Total contact buffer allocation across all worlds forwarded to ``mjwarp.put_data``."""
  naccdmax: int | None = None
  """Total CCD contact buffer allocation across all worlds forwarded to ``mjwarp.put_data``."""
  contact_sensor_maxmatch: int = 64
  broadphase: Literal["nxn", "sap_tile", "sap_segmented"] | None = None
  """Broadphase collision algorithm. If None, use the MuJoCo Warp default."""
  broadphase_filter: tuple[Literal["plane", "sphere", "aabb", "obb"], ...] | None = None
  """Bounding-volume filters applied during broadphase collision checking.

  If None, use the MuJoCo Warp default."""
  warn_overflow: bool | None = None
  """Enable or disable device-side MJWarp overflow warnings (e.g. ``False`` silences GPU printf)."""
  warn_overflow_flags: tuple[OverflowFlagChoice, ...] | None = None
  """Granular MJWarp 4.0 ``OverflowType`` warning bitmask (requires ``mujoco-warp>=4.0``)."""
  deterministic: DeterminismChoice | None = None
  """MJWarp 4.0 deterministic execution bitmask (``none``, ``contacts``, ``constraint``, ``atomics``, ``contacts_constraint``, ``all``)."""
  run_qacc_smooth: bool | None = None
  """MJWarp 4.0 toggle: when ``False``, Newton starts from ``qacc_warmstart`` and skips ``qacc_smooth`` + ``factor_m``."""
  run_rne_postconstraint: bool | None = None
  """MJWarp 4.0 toggle for RNE post-constraint force computation."""
  multiccd: bool | None = None
  """Toggle MJWarp rich multi-point convex contacts (enabled by default in MJWarp; ``False`` sets ``mjDSBL_MULTICCD``)."""
  ls_parallel: bool | None = None
  """Deprecated and ignored. Parallel linesearch was removed in MuJoCo Warp 3.10."""
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)

  def __post_init__(self) -> None:
    if self.ls_parallel is not None:
      warnings.warn(
        "SimulationCfg.ls_parallel is deprecated and ignored; parallel "
        "linesearch was removed in MuJoCo Warp 3.10.",
        DeprecationWarning,
        stacklevel=2,
      )
    if self.warn_overflow is not None and self.warn_overflow_flags is not None:
      raise ValueError(
        "Specify at most one of SimulationCfg.warn_overflow or "
        "SimulationCfg.warn_overflow_flags."
      )

  def apply_model(self, model: _MjModel) -> None:
    """Apply ``MujocoCfg`` and ``SimulationCfg`` model-level flags prior to ``put_model``."""
    self.mujoco.apply(model)
    if self.multiccd is not None:
      multiccd_bit = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD.value)
      if self.multiccd:
        model.opt.disableflags &= ~multiccd_bit
      else:
        model.opt.disableflags |= multiccd_bit

  def apply_wp_opt(self, wp_opt: mjwarp.Option) -> None:
    """Apply MuJoCo Warp-only settings to a warp Option (post ``put_model``)."""
    wp_opt.contact_sensor_maxmatch = self.contact_sensor_maxmatch
    if self.multiccd is not None:
      multiccd_bit = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD.value)
      if self.multiccd:
        wp_opt.disableflags &= ~multiccd_bit
      else:
        wp_opt.disableflags |= multiccd_bit
    if self.broadphase is not None:
      wp_opt.broadphase = _BROADPHASE_MAP[self.broadphase]
    if self.broadphase_filter is not None:
      wp_opt.broadphase_filter = mjwarp.BroadphaseFilter(0)
      for name in self.broadphase_filter:
        wp_opt.broadphase_filter |= _BROADPHASE_FILTER_MAP[name]
    # On MuJoCo >= 4.0, default device-side GPU printf warnings to silent (False)
    # unless explicitly configured, avoiding CUDA printf FIFO spinlock contention
    # while keeping sticky d.overflow capacity bits intact.
    if self.warn_overflow is not None:
      wp_opt.warn_overflow = self.warn_overflow
    elif self.warn_overflow_flags is None and _IS_MUJOCO_4:
      wp_opt.warn_overflow = False
    if self.warn_overflow_flags is not None:
      if not _HAS_OVERFLOW_TYPE:
        raise ValueError(
          "SimulationCfg.warn_overflow_flags requires mujoco-warp >= 4.0 "
          "(launch with --mujoco-backend 4.0)."
        )
      mask = 0
      for flag_name in self.warn_overflow_flags:
        if flag_name not in _OVERFLOW_FLAG_MAP:
          raise ValueError(
            f"Unknown overflow flag {flag_name!r}. Valid: {sorted(_OVERFLOW_FLAG_MAP)}"
          )
        mask |= _OVERFLOW_FLAG_MAP[flag_name]
      wp_opt.warn_overflow = mask
    if self.deterministic is not None:
      if not _HAS_DETERMINISM_TYPE:
        raise ValueError(
          "SimulationCfg.deterministic requires mujoco-warp >= 4.0 "
          "(launch with --mujoco-backend 4.0)."
        )
      if self.deterministic not in _DETERMINISM_MAP:
        raise ValueError(
          f"Unknown deterministic mode {self.deterministic!r}. "
          f"Valid: {sorted(_DETERMINISM_MAP)}"
        )
      wp_opt.deterministic = _DETERMINISM_MAP[self.deterministic]
    if self.run_qacc_smooth is not None:
      if not hasattr(wp_opt, "run_qacc_smooth"):
        raise ValueError(
          "SimulationCfg.run_qacc_smooth requires mujoco-warp >= 4.0 "
          "(launch with --mujoco-backend 4.0)."
        )
      wp_opt.run_qacc_smooth = self.run_qacc_smooth
    elif _IS_MUJOCO_4 and hasattr(wp_opt, "run_qacc_smooth"):
      wp_opt.run_qacc_smooth = False
    if self.run_rne_postconstraint is not None:
      if not hasattr(wp_opt, "run_rne_postconstraint"):
        raise ValueError(
          "SimulationCfg.run_rne_postconstraint requires mujoco-warp >= 4.0 "
          "(launch with --mujoco-backend 4.0)."
        )
      wp_opt.run_rne_postconstraint = self.run_rne_postconstraint


class Simulation:
  """GPU-accelerated MuJoCo simulation powered by MJWarp.

  CUDA Graph Capture
  ------------------
  On CUDA devices with memory pools enabled, the simulation captures CUDA graphs
  for ``step()``, ``forward()``, and ``reset()`` operations. Graph capture records
  a sequence of GPU kernels and their memory addresses, then replays the entire
  sequence with a single kernel launch, eliminating CPU overhead from repeated
  kernel dispatches.

  **Important:** A captured graph holds pointers to the GPU arrays that existed
  at capture time. If those arrays are later replaced (e.g., via
  ``expand_model_fields()``), the graph will still read from the old arrays,
  silently ignoring any new values. The ``expand_model_fields()`` method handles
  this automatically by calling ``create_graph()`` after replacing arrays.

  If you write code that replaces model or data arrays after simulation
  initialization, you **must** call ``create_graph()`` afterward to re-capture
  the graphs with the new memory addresses.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel | None = None,
    device: str = "cuda:0",
    *,
    spec: mujoco.MjSpec | None = None,
    variant_info: list[tuple[str, VariantMetadata]] | None = None,
  ):
    self.cfg = cfg
    self.device = device
    self.wp_device = wp.get_device(self.device)
    self.num_envs = num_envs
    self._default_model_fields: dict[str, torch.Tensor] = {}
    # Fields whose DR baseline is per-world (DR's `_select_default_values`
    # uses this to know whether to index `[env, ...]` vs `[...]`).
    # Variant-dependent fields go here; `geom_dataid` does not, because
    # DR does not randomize mesh selection.
    self._per_world_default_fields: set[str] = set()
    # Fields that have per-world warp arrays (viewer sync uses this to
    # know what to copy per env). Superset of `_per_world_default_fields`
    # plus `geom_dataid` for variant scenes.
    self._expanded_fields: set[str] = set()
    # Per-entity variant assignment, keyed by entity name (no trailing
    # slash). Empty for non-variant scenes.
    self._world_to_variant: dict[str, torch.Tensor] = {}

    if spec is not None and variant_info:
      self._init_with_variants(spec, variant_info)
    elif spec is not None:
      compiled = spec.compile()
      cfg.apply_model(compiled)
      self._init_with_model(compiled)
    elif model is not None:
      cfg.apply_model(model)
      self._init_with_model(model)
    else:
      raise ValueError("Either model or spec must be provided.")

  def _init_with_model(self, model: mujoco.MjModel) -> None:
    """Standard path: build warp model from compiled MjModel."""
    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    with wp.ScopedDevice(self.wp_device):
      try:
        self._wp_model = mjwarp.put_model(self._mj_model)
      except NotImplementedError as exc:
        if "mjENBL_SPECULATIVE" in str(exc):
          raise NotImplementedError(
            "MuJoCo 4.0 speculative contacts (enableflags=('speculative',)) are "
            "currently CPU C-MuJoCo only and unsupported by GPU mujoco_warp."
          ) from exc
        raise
      self.cfg.apply_wp_opt(self._wp_model.opt)
      self._finish_init()

  def _init_with_variants(
    self,
    spec: mujoco.MjSpec,
    variant_info: list[tuple[str, VariantMetadata]],
  ) -> None:
    """Per-world mesh path: build model with per-world geom_dataid.

    ``self._mj_model`` remains a single host-side template model. The actual
    simulation model may store variant-dependent fields as per-world arrays in
    ``self._wp_model`` / ``self.model``. CPU consumers that call MuJoCo APIs on
    ``self._mj_model`` must first sync the relevant per-world fields for the env
    they are rendering or inspecting.
    """
    with wp.ScopedDevice(self.wp_device):
      try:
        result = build_variant_model(
          spec,
          self.num_envs,
          variant_info,
          configure_model=self.cfg.apply_model,
        )
      except NotImplementedError as exc:
        if "mjENBL_SPECULATIVE" in str(exc):
          raise NotImplementedError(
            "MuJoCo 4.0 speculative contacts (enableflags=('speculative',)) are "
            "currently CPU C-MuJoCo only and unsupported by GPU mujoco_warp."
          ) from exc
        raise
      self._mj_model = result.mj_model
      self._mj_data = mujoco.MjData(self._mj_model)
      mujoco.mj_forward(self._mj_model, self._mj_data)

      self._wp_model = result.wp_model
      self.cfg.apply_wp_opt(self._wp_model.opt)

      # Snapshot variant-dependent fields as per-world defaults so
      # DR scale/add operations use each variant's base values.
      for field_name in VARIANT_DEPENDENT_FIELDS:
        arr = getattr(self._wp_model, field_name)
        self._default_model_fields[field_name] = torch.as_tensor(
          arr.numpy(), device=self.device
        ).clone()
      self._per_world_default_fields.update(VARIANT_DEPENDENT_FIELDS)

      self._finish_init()

    # Register variant-dependent fields as expanded so the native
    # viewer syncs them per-world.
    self._expanded_fields.update(VARIANT_DEPENDENT_FIELDS)
    self._expanded_fields.add("geom_dataid")
    self._expanded_fields.add("geom_matid")

    # Stash variant assignments as torch tensors keyed by bare entity name
    # (build_variant_model emits "<name>/" prefixes; strip the trailing slash for
    # the public API).
    for prefix, arr in result.world_to_variant.items():
      key = prefix.rstrip("/")
      self._world_to_variant[key] = torch.as_tensor(
        arr, dtype=torch.long, device=self.device
      )

  def _finish_init(self) -> None:
    """Common initialization after warp model is created."""
    self._wp_data = mjwarp.put_data(
      self._mj_model,
      self._mj_data,
      nworld=self.num_envs,
      nconmax=self.cfg.nconmax,
      nccdmax=self.cfg.nccdmax,
      njmax=self.cfg.njmax,
      njmax_nnz=self.cfg.njmax_nnz,
      naconmax=self.cfg.naconmax,
      naccdmax=self.cfg.naccdmax,
    )

    self._reset_mask_wp = wp.zeros(self.num_envs, dtype=bool)
    self._reset_mask = TorchArray(self._reset_mask_wp)
    with wp.ScopedDevice(self.wp_device):
      self._empty_sensor_collision_wp = wp.empty((self.num_envs, 0, 8, 7), dtype=float)

    self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
    self._data_bridge = WarpBridge(self._wp_data)
    self._sensor_context: SensorContext | None = None

    self.use_cuda_graph = self._should_use_cuda_graph()
    self._inline_warp_kernels: bool = False
    self.create_graph()

    self.nan_guard = NanGuard(self.cfg.nan_guard, self.num_envs, self._mj_model)

  def create_graph(self) -> None:
    """Capture CUDA graphs for step, forward, and reset operations.

    This method must be called whenever GPU arrays in the model or data are
    replaced after initialization. The captured graphs hold pointers to the
    arrays that existed at capture time. If those arrays are replaced, the
    graphs will silently read from the old arrays, ignoring any new values.

    Called automatically by:
    - ``__init__()`` during simulation initialization
    - ``expand_model_fields()`` after replacing model arrays

    On CPU devices or when memory pools are disabled, this is a no-op.
    """
    self.step_graph = None
    self.forward_graph = None
    self.kinematics_forward_graph = None
    self.reset_graph = None
    self.sense_graph = None
    if self.use_cuda_graph:
      with _suspend_gc(), wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
          mjwarp.step(self.wp_model, self.wp_data)
        self.step_graph = capture.graph
        with wp.ScopedCapture() as capture:
          mjwarp.forward(self.wp_model, self.wp_data)
        self.forward_graph = capture.graph
        if self.can_use_kinematic_forward(allow_unread_contact_sensors=True):
          with wp.ScopedCapture() as capture:
            self._forward_kinematics_and_sensors_kernel()
          self.kinematics_forward_graph = capture.graph
        with wp.ScopedCapture() as capture:
          mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
        self.reset_graph = capture.graph
        if self._sensor_context is not None:
          with wp.ScopedCapture() as capture:
            self._sense_kernel()
          self.sense_graph = capture.graph

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    return self._wp_data

  @property
  def data(self) -> "DataBridge":
    return cast("DataBridge", self._data_bridge)

  @property
  def model(self) -> "ModelBridge":
    return cast("ModelBridge", self._model_bridge)

  @property
  def default_model_fields(self) -> dict[str, torch.Tensor]:
    """Default values for expanded model fields, used in domain randomization."""
    return self._default_model_fields

  @property
  def expanded_fields(self) -> set[str]:
    """Names of model fields that have been expanded for per-env DR."""
    return self._expanded_fields

  @property
  def per_world_default_fields(self) -> set[str]:
    """Fields with per-world defaults from mesh variant compilation."""
    return self._per_world_default_fields

  @property
  def world_to_variant(self) -> dict[str, torch.Tensor]:
    """Per-entity variant assignment for variant scenes.

    Maps entity name (without trailing slash) to a ``(num_envs,)`` tensor of
    variant indices. The index order matches the order of variants declared
    in the entity's :class:`VariantEntityCfg`. Empty for non-variant scenes.
    """
    return self._world_to_variant

  # Methods.

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model fields to support per-environment parameters."""
    if not fields:
      return

    invalid_fields = [f for f in fields if not hasattr(self._mj_model, f)]
    if invalid_fields:
      raise ValueError(f"Fields not found in model: {invalid_fields}")

    # In MJWarp >= 4.0, set_const(KIN0) recomputes per-world eq_data for body
    # connect/weld constraints; expand eq_data alongside set_const_0 fields when neq > 0.
    if (
      _HAS_SET_CONST_GROUP
      and self._mj_model.neq > 0
      and "dof_invweight0" in fields
      and "eq_data" not in fields
    ):
      fields = (*fields, "eq_data")

    expand_model_fields(self._wp_model, self.num_envs, list(fields))
    self._expanded_fields.update(fields)
    self._model_bridge.clear_cache()

    if self._sensor_context is not None:
      self._sensor_context.recreate(self._mj_model, self._expanded_fields)

    # Field expansion allocates new arrays and replaces them via setattr. The
    # CUDA graph captured the old memory addresses, so we must recreate it.
    self.create_graph()

  def get_default_field(self, field: str) -> torch.Tensor:
    """Get the default value for a model field, caching for reuse.

    Returns the original values from the C MuJoCo model (mj_model), obtained
    from the final compiled scene spec before any randomization is applied.
    Not to be confused with the GPU Warp model (wp_model) which may have
    randomized values.
    """
    if field not in self._default_model_fields:
      if not hasattr(self._mj_model, field):
        raise ValueError(f"Field '{field}' not found in model")
      model_field = getattr(self.model, field)
      default_value = getattr(self._mj_model, field)
      self._default_model_fields[field] = torch.as_tensor(
        default_value, dtype=model_field.dtype, device=self.device
      ).clone()
    return self._default_model_fields[field]

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants after domain randomization.

    Args:
      level: Which constants to recompute. ``set_const`` is the most
        expensive (covers body_mass changes), ``set_const_0`` covers
        qpos0/body_inertia/dof_armature changes, and ``set_const_fixed``
        is the cheapest (covers body_gravcomp changes).
    """
    with wp.ScopedDevice(self.wp_device):
      if _HAS_SET_CONST_GROUP:
        if level == RecomputeLevel.set_const_fixed:
          groups = mjwarp.SetConstGroup.FIXED
        elif level == RecomputeLevel.set_const_0:
          groups = mjwarp.SetConstGroup.KIN0 | mjwarp.SetConstGroup.INERT0
        elif level == RecomputeLevel.set_const:
          groups = mjwarp.SetConstGroup.ALL
        else:
          return
        mjwarp.set_const(self._wp_model, self._wp_data, groups=groups)
      else:
        fn = getattr(mjwarp, level.name)
        fn(self._wp_model, self._wp_data)

  def can_use_kinematic_forward(
    self, *, allow_unread_contact_sensors: bool = False
  ) -> bool:
    """Return True if kinematic+pos/vel sensor refresh is exact for this model."""
    m = self._wp_model
    mj = self._mj_model
    enable_flags = int(m.opt.enableflags)
    disable_flags = int(m.opt.disableflags)
    sleep_enabled = bool(
      enable_flags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    ) and not bool(disable_flags & int(mujoco.mjtDisableBit.mjDSBL_ISLAND))
    if sleep_enabled:
      return False
    if bool(enable_flags & int(mujoco.mjtEnableBit.mjENBL_ENERGY)):
      return False
    if bool(getattr(m, "sensor_e_potential", False)) or bool(
      getattr(m, "sensor_e_kinetic", False)
    ):
      return False
    if int(getattr(mj, "nhistory", 0)) != 0:
      return False
    if int(getattr(m, "nsensorcollision", 0)) != 0:
      return False
    if int(getattr(m, "nsensortaxel", 0)) != 0:
      return False
    adr_attrs: tuple[str, ...] = (
      ("sensor_limitpos_adr", "sensor_limitvel_adr")
      if allow_unread_contact_sensors
      else (
        "sensor_limitpos_adr",
        "sensor_limitvel_adr",
        "sensor_contact_pos_adr",
        "sensor_contact_vel_adr",
      )
    )
    for adr_attr in adr_attrs:
      adr = getattr(m, adr_attr, None)
      if adr is not None and int(adr.size) != 0:
        return False
    if allow_unread_contact_sensors:
      nrange = int(getattr(m, "nrangefinder", 0))
      has_pos = (
        nrange > 0 or int(getattr(getattr(m, "sensor_pos_adr", None), "size", 0)) > 0
      )
      has_vel = int(getattr(getattr(m, "sensor_vel_adr", None), "size", 0)) > 0
      c_pos = int(getattr(getattr(m, "sensor_contact_pos_adr", None), "size", 0))
      c_vel = int(getattr(getattr(m, "sensor_contact_vel_adr", None), "size", 0))
      if (
        has_pos
        and c_pos > 0
        and (not _IS_MUJOCO_4 or _mjwarp_sensor_pos is None or nrange > 0)
      ):
        return False
      if (
        has_vel
        and c_vel > 0
        and (
          not _IS_MUJOCO_4
          or _mjwarp_sensor_vel is None
          or (
            bool(getattr(m, "sensor_subtree_vel", False))
            and _mjwarp_subtree_vel is None
          )
        )
      ):
        return False
    cb = getattr(m, "callback", None)
    if cb is not None and (
      getattr(cb, "sensor", None) is not None
      or getattr(cb, "control", None) is not None
    ):
      return False
    return True

  def _forward_kinematics_and_sensors_kernel(self) -> None:
    """Refresh kinematics, COM/transmission velocities, and pos/vel sensors only."""
    m = self.wp_model
    d = self.wp_data
    nactuator = int(getattr(m, "nactuator", getattr(m, "nu", self._mj_model.nu)))
    ntendon = int(getattr(m, "ntendon", self._mj_model.ntendon))
    mjwarp.fwd_kinematics(m, d)
    mjwarp.transmission(m, d)
    if (
      not _IS_MUJOCO_4
      or (nactuator > 0 and _mjwarp_actuator_velocity is None)
      or (ntendon > 0 and _mjwarp_tendon_velocity is None)
    ):
      mjwarp.fwd_velocity(m, d)
    else:
      if nactuator > 0 and _mjwarp_actuator_velocity is not None:
        wp.launch(
          _mjwarp_actuator_velocity,
          dim=(d.nworld, nactuator),
          inputs=[
            d.qvel,
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
            d.actuator_moment,
          ],
          outputs=[d.actuator_velocity],
        )
      if ntendon > 0 and _mjwarp_tendon_velocity is not None:
        wp.launch(
          _mjwarp_tendon_velocity,
          dim=(d.nworld, ntendon),
          inputs=[
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            d.qvel,
            d.ten_J,
          ],
          outputs=[d.ten_velocity],
        )
      mjwarp.com_vel(m, d)
    nrange = int(getattr(m, "nrangefinder", 0))
    npos = int(getattr(getattr(m, "sensor_pos_adr", None), "size", 0))
    nlimitpos = int(getattr(getattr(m, "sensor_limitpos_adr", None), "size", 0))
    if (
      _IS_MUJOCO_4
      and npos > 0
      and nrange == 0
      and nlimitpos == 0
      and _mjwarp_sensor_pos is not None
    ):
      wp.launch(
        _mjwarp_sensor_pos,
        dim=(d.nworld, npos),
        inputs=[
          m.ngeom,
          m.opt.magnetic,
          m.body_geomnum,
          m.body_geomadr,
          m.body_iquat,
          m.body_mass,
          m.body_subtreemass,
          m.jnt_qposadr,
          m.geom_type,
          m.geom_bodyid,
          m.geom_quat,
          m.site_type,
          m.site_bodyid,
          m.site_size,
          m.site_quat,
          m.cam_bodyid,
          m.cam_quat,
          m.cam_fovy,
          m.cam_resolution,
          m.cam_sensorsize,
          m.cam_intrinsic,
          m.sensor_type,
          m.sensor_datatype,
          m.sensor_objtype,
          m.sensor_objid,
          m.sensor_reftype,
          m.sensor_refid,
          m.sensor_adr,
          m.sensor_cutoff,
          m.nxn_pairid,
          m.sensor_pos_adr,
          d.time,
          d.energy,
          d.qpos,
          d.xpos,
          d.xquat,
          d.xmat,
          d.xipos,
          d.ximat,
          d.geom_xpos,
          d.geom_xmat,
          d.site_xpos,
          d.site_xmat,
          d.cam_xpos,
          d.cam_xmat,
          d.subtree_com,
          d.ten_length,
          d.actuator_length,
          self._empty_sensor_collision_wp,
        ],
        outputs=[d.sensordata],
      )
    elif nrange > 0 or npos > 0 or nlimitpos > 0:
      mjwarp.sensor_pos(m, d)
    nvel = int(getattr(getattr(m, "sensor_vel_adr", None), "size", 0))
    nlimitvel = int(getattr(getattr(m, "sensor_limitvel_adr", None), "size", 0))
    if _IS_MUJOCO_4 and nvel > 0 and nlimitvel == 0 and _mjwarp_sensor_vel is not None:
      if (
        bool(getattr(m, "sensor_subtree_vel", False))
        and _mjwarp_subtree_vel is not None
      ):
        _mjwarp_subtree_vel(m, d)
      wp.launch(
        _mjwarp_sensor_vel,
        dim=(d.nworld, nvel),
        inputs=[
          m.body_rootid,
          m.jnt_dofadr,
          m.geom_bodyid,
          m.site_bodyid,
          m.cam_bodyid,
          m.sensor_type,
          m.sensor_datatype,
          m.sensor_objtype,
          m.sensor_objid,
          m.sensor_reftype,
          m.sensor_refid,
          m.sensor_adr,
          m.sensor_cutoff,
          m.sensor_vel_adr,
          d.qvel,
          d.xpos,
          d.xmat,
          d.xipos,
          d.ximat,
          d.geom_xpos,
          d.geom_xmat,
          d.site_xpos,
          d.site_xmat,
          d.cam_xpos,
          d.cam_xmat,
          d.subtree_com,
          d.ten_velocity,
          d.actuator_velocity,
          d.cvel,
          d.subtree_linvel,
          d.subtree_angmom,
        ],
        outputs=[d.sensordata],
      )
    elif nvel > 0 or nlimitvel > 0:
      mjwarp.sensor_vel(m, d)

  def _should_inline_warp_on_torch_stream(self) -> bool:
    """Return True when Warp kernels must run inline on the current PyTorch CUDA stream.

    Required when an outer CUDA Graph (e.g., ``EnvFullStepGraph`` or
    ``RolloutStepMegaGraph``) is warming up or capturing on a dedicated
    ``torch.cuda.Stream``, because nested ``wp.capture_launch`` on Warp's
    default stream cannot be recorded into an outer capturing stream.
    """
    if not self.wp_device.is_cuda:
      return False
    return bool(
      getattr(self, "_inline_warp_kernels", False)
      or torch.cuda.is_current_stream_capturing()
    )

  def _active_warp_stream(self) -> wp.Stream:
    """Return a Warp Stream matching the current PyTorch CUDA stream without creating an unregistered wrapper when already scoped."""
    torch_stream = torch.cuda.current_stream(self.device)
    cur_wp = wp.get_stream(self.wp_device)
    if cur_wp.cuda_stream == torch_stream.cuda_stream:
      return cur_wp
    return wp.stream_from_torch(torch_stream)

  def forward_kinematics_and_sensors(self) -> None:
    """Refresh kinematic quantities and pos/vel sensors without collision/solver."""
    with wp.ScopedDevice(self.wp_device):
      if self._should_inline_warp_on_torch_stream():
        with wp.ScopedStream(
          self._active_warp_stream(), sync_enter=False, sync_exit=False
        ):
          if self.can_use_kinematic_forward(allow_unread_contact_sensors=True):
            self._forward_kinematics_and_sensors_kernel()
          else:
            mjwarp.forward(self.wp_model, self.wp_data)
      elif self.use_cuda_graph and self.kinematics_forward_graph is not None:
        wp.capture_launch(self.kinematics_forward_graph)
      elif self.can_use_kinematic_forward(allow_unread_contact_sensors=True):
        self._forward_kinematics_and_sensors_kernel()
      else:
        mjwarp.forward(self.wp_model, self.wp_data)

  def forward(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self._should_inline_warp_on_torch_stream():
        with wp.ScopedStream(
          self._active_warp_stream(), sync_enter=False, sync_exit=False
        ):
          mjwarp.forward(self.wp_model, self.wp_data)
      elif self.use_cuda_graph and self.forward_graph is not None:
        wp.capture_launch(self.forward_graph)
      else:
        mjwarp.forward(self.wp_model, self.wp_data)

  def step(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      with self.nan_guard.watch(self.data):
        if self._should_inline_warp_on_torch_stream():
          with wp.ScopedStream(
            self._active_warp_stream(), sync_enter=False, sync_exit=False
          ):
            mjwarp.step(self.wp_model, self.wp_data)
        elif self.use_cuda_graph and self.step_graph is not None:
          wp.capture_launch(self.step_graph)
        else:
          mjwarp.step(self.wp_model, self.wp_data)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    with wp.ScopedDevice(self.wp_device):
      if env_ids is None:
        self._reset_mask.fill_(True)
      else:
        self._reset_mask.fill_(False)
        self._reset_mask[env_ids] = True

      if self._should_inline_warp_on_torch_stream():
        with wp.ScopedStream(
          self._active_warp_stream(), sync_enter=False, sync_exit=False
        ):
          mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
      elif self.use_cuda_graph and self.reset_graph is not None:
        wp.capture_launch(self.reset_graph)
      else:
        mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)

  def reset_mask(self, mask: torch.Tensor) -> None:
    """Branchless CUDA-Graph-safe reset using a boolean mask of shape ``(num_envs,)``."""
    with wp.ScopedDevice(self.wp_device):
      self._reset_mask.copy_(mask)
      if self._should_inline_warp_on_torch_stream():
        with wp.ScopedStream(
          self._active_warp_stream(), sync_enter=False, sync_exit=False
        ):
          mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
      elif self.use_cuda_graph and self.reset_graph is not None:
        wp.capture_launch(self.reset_graph)
      else:
        mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)

  def set_sensor_context(self, ctx: SensorContext) -> None:
    """Wire a SensorContext for camera/raycast sensing.

    Automatically re-captures CUDA graphs so the sense_graph includes
    the new sensor kernels.
    """
    self._sensor_context = ctx
    self.create_graph()

  def sense(self) -> None:
    """Execute the sense pipeline: prepare -> graph -> finalize.

    Runs BVH refit, camera rendering, and raycasting in a single
    CUDA graph launch. Should be called once per env step, right
    before observation computation.
    """
    if self._sensor_context is None:
      return

    ctx = self._sensor_context
    ctx.prepare()

    with wp.ScopedDevice(self.wp_device):
      if self._should_inline_warp_on_torch_stream():
        with wp.ScopedStream(
          self._active_warp_stream(), sync_enter=False, sync_exit=False
        ):
          self._sense_kernel()
      elif self.use_cuda_graph and self.sense_graph is not None:
        wp.capture_launch(self.sense_graph)
      else:
        self._sense_kernel()

    ctx.finalize()

  # Private methods.

  def _sense_kernel(self) -> None:
    """GPU kernel sequence for sensing (captured in sense_graph)."""
    assert self._sensor_context is not None
    ctx = self._sensor_context
    rc = ctx.render_context

    mjwarp.refit_bvh(self.wp_model, self.wp_data, rc)

    if ctx.has_cameras:
      mjwarp.render(self.wp_model, self.wp_data, rc)
      ctx.unpack_rgb()

    for sensor in ctx.raycast_sensors:
      sensor.raycast_kernel(rc=rc)

  def _should_use_cuda_graph(self) -> bool:
    """Determine if CUDA graphs can be used based on device and driver version."""
    if not self.wp_device.is_cuda:
      return False

    driver_ver = wp.get_cuda_driver_version()
    has_mempool = wp.is_mempool_enabled(self.wp_device)

    if driver_ver is None:
      print("[WARNING] CUDA Graphs disabled: driver version unavailable")
      return False

    if has_mempool and driver_ver >= _GRAPH_CAPTURE_MIN_DRIVER:
      return True

    reasons = []
    if not has_mempool:
      reasons.append("mempool disabled")
    if driver_ver < _GRAPH_CAPTURE_MIN_DRIVER:
      reasons.append(f"driver {driver_ver[0]}.{driver_ver[1]} < 12.4")
    print(f"[WARNING] CUDA Graphs disabled: {', '.join(reasons)}")
    return False
