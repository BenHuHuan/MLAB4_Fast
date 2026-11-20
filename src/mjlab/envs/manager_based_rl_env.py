import functools
import gc
import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager, ActionTermCfg
from mjlab.managers.command_manager import (
  CommandManager,
  CommandTermCfg,
  NullCommandManager,
)
from mjlab.managers.curriculum_manager import (
  CurriculumManager,
  CurriculumTermCfg,
  NullCurriculumManager,
)
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.metrics_manager import (
  MetricsManager,
  MetricsTermCfg,
  NullMetricsManager,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.managers.recorder_manager import (
  NullRecorderManager,
  RecorderManager,
  RecorderTermCfg,
)
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils import random as random_utils
from mjlab.utils.logging import print_info
from mjlab.utils.spaces import Box
from mjlab.utils.spaces import Dict as DictSpace
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
  """Configuration for a manager-based RL environment.

  This config defines all aspects of an RL environment: the physical scene,
  observations, actions, rewards, terminations, and optional features like
  commands and curriculum learning.

  The environment step size is ``sim.mujoco.timestep * decimation``. For example,
  with a 2ms physics timestep and decimation=10, the environment runs at 50Hz.
  """

  # Base environment configuration.

  decimation: int
  """Number of physics simulation steps per environment step. Higher values mean
  coarser control frequency. Environment step duration = physics_dt * decimation."""

  scene: SceneCfg
  """Scene configuration defining terrain, entities, and sensors. The scene
  specifies ``num_envs``, the number of parallel environments."""

  observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
  """Observation groups configuration. Each group (e.g., "actor", "critic") contains
  observation terms that are concatenated. Groups can have different settings for
  noise, history, and delay."""

  actions: dict[str, ActionTermCfg] = field(default_factory=dict)
  """Action terms configuration. Each term controls a specific entity/aspect
  (e.g., joint positions). Action dimensions are concatenated across terms."""

  events: dict[str, EventTermCfg] = field(
    default_factory=lambda: {
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      )
    }
  )
  """Event terms for domain randomization and state resets. Default includes
  ``reset_scene_to_default`` which resets entities to their initial state.
  Can be set to empty to disable all events including default reset."""

  seed: int | None = None
  """Random seed for reproducibility. If None, a random seed is used. The actual
  seed used is stored back into this field after initialization."""

  sim: SimulationCfg = field(default_factory=SimulationCfg)
  """Simulation configuration including physics timestep, solver iterations,
  contact parameters, and NaN guarding."""

  viewer: ViewerConfig = field(default_factory=ViewerConfig)
  """Viewer configuration for rendering (camera position, resolution, etc.)."""

  # RL-specific configuration.

  episode_length_s: float = 0.0
  """Duration of an episode (in seconds).

  Episode length in steps is computed as:
    ceil(episode_length_s / (sim.mujoco.timestep * decimation))
  """

  rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
  """Reward terms configuration."""

  terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
  """Termination terms configuration. If empty, episodes never reset. Use
  ``mdp.time_out`` with ``time_out=True`` for episode time limits."""

  commands: dict[str, CommandTermCfg] = field(default_factory=dict)
  """Command generator terms (e.g., velocity targets)."""

  curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
  """Curriculum terms for adaptive difficulty."""

  metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
  """Custom metric terms for logging per-step values as episode averages."""

  recorders: dict[str, RecorderTermCfg] = field(default_factory=dict)
  """Recorder terms for logging observations, actions, or other data during rollouts.
  If empty, a no-op manager is used with zero overhead."""

  is_finite_horizon: bool = False
  """Whether the task has a finite or infinite horizon. Defaults to False (infinite).

  - **Finite horizon (True)**: The time limit defines the task boundary. When reached,
    no future value exists beyond it, so the agent receives a terminal done signal.
  - **Infinite horizon (False)**: The time limit is an artificial cutoff. The agent
    receives a truncated done signal to bootstrap the value of continuing beyond the
    limit.
  """

  auto_reset: bool = True
  """Whether to automatically reset environments that terminate or time out.

  When True (default), ``step()`` resets done environments and returns post-reset
  observations. When False, ``step()`` returns the true terminal observation and the
  caller must explicitly call ``reset(env_ids=...)`` for done environments before the
  next ``step()``.

  Note: mjlab's bundled ``train.py`` goes through rsl_rl's ``OnPolicyRunner``, which
  does not drive manual resets. ``auto_reset=False`` is intended for users running
  their own training loop (or a wrapper that handles the reset between steps).
  """

  scale_rewards_by_dt: bool = True
  """Whether to multiply rewards by the environment step duration (dt).

  When True (default), reward values are scaled by step_dt to normalize cumulative
  episodic rewards across different simulation frequencies. Set to False for
  algorithms that expect unscaled reward signals (e.g., HER, static reward scaling).
  """


class ManagerBasedRlEnv:
  """Manager-based RL environment."""

  is_vector_env = True
  metadata = {
    "render_modes": [None, "rgb_array"],
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.config.version,
  }
  cfg: ManagerBasedRlEnvCfg

  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    # Initialize base environment state.
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}
    self._manual_reset_pending = torch.zeros(
      self.cfg.scene.num_envs, dtype=torch.bool, device=device
    )
    # Scratch buffer for per-env command dt; see step().
    self._command_dt = torch.zeros(self.cfg.scene.num_envs, device=device)

    # Initialize scene and simulation.
    self.scene = Scene(self.cfg.scene, device=device)
    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      spec=self.scene.spec,
      variant_info=self.scene.collect_variant_info(),
      device=device,
    )

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    # Wire sensor context to simulation for sense_graph.
    if self.scene.sensor_context is not None:
      self.sim.set_sensor_context(self.scene.sensor_context)

    # Print environment info.
    print_info("")
    table = PrettyTable()
    table.title = "Base Environment"
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"
    table.add_row(["Number of environments", self.num_envs])
    table.add_row(["Environment device", self.device])
    table.add_row(["Environment seed", self.cfg.seed])
    table.add_row(["Physics step-size", self.physics_dt])
    table.add_row(["Environment step-size", self.step_dt])
    print_info(table.get_string())
    print_info("")

    # Initialize RL-specific state.
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.scene.num_envs, device=device, dtype=torch.long
    )
    self.render_mode = render_mode
    self._offline_renderer: OffscreenRenderer | None = None
    if self.render_mode == "rgb_array":
      renderer = OffscreenRenderer(
        model=self.sim.mj_model,
        cfg=self.cfg.viewer,
        scene=self.scene,
        sim_model=self.sim.model,
        expanded_fields=self.sim.expanded_fields,
      )
      renderer.initialize()
      self._offline_renderer = renderer
    self.metadata["render_fps"] = 1.0 / self.step_dt

    # Load all managers.
    self.load_managers()
    self.setup_manager_visualizers()

    # Opt-in FP32 exact runtime optimization state (auto-activated when configured on cfg).
    self._zero_sync_metrics: bool = False
    self._use_kinematic_post_step_forward: bool = False
    self._decimation_substep_runner: _DecimationSubstepGraph | None = None
    self._fp32_step_opts_active: bool = False
    self._mega_step_graph: Any = None
    self._staged_episode_logs_active: bool = False
    self._fp32_step_opt_report: dict[str, Any] = {}
    if (
      type(self).step is ManagerBasedRlEnv.step
      and getattr(cfg, "_rollout_optimization_mode", "off") != "off"
    ):
      self._fp32_step_opt_report = self.enable_fp32_step_optimizations(
        zero_sync_metrics=bool(getattr(cfg, "_use_zero_sync_optimizations", True)),
        kinematic_forward=True,
        substep_graph=bool(getattr(cfg, "_use_substep_cuda_graph", True)),
        mega_graph_mode=str(getattr(cfg, "_mega_graph_mode", "off")),
        reset_pool_mode=str(getattr(cfg, "_reset_pool_mode", "off")),
      )

  # Properties.

  @property
  def num_envs(self) -> int:
    """Number of parallel environments."""
    return self.scene.num_envs

  @property
  def physics_dt(self) -> float:
    """Physics simulation step size."""
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    """Environment step size (physics_dt * decimation)."""
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self) -> str:
    """Device for computation."""
    return self.sim.device

  @property
  def max_episode_length_s(self) -> float:
    """Maximum episode length in seconds."""
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> int:
    """Maximum episode length in steps."""
    return math.ceil(self.max_episode_length_s / self.step_dt)

  @property
  def unwrapped(self) -> "ManagerBasedRlEnv":
    """Get the unwrapped environment (base case for wrapper chains)."""
    return self

  # Methods.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}
    if getattr(self.command_manager, "active_terms", None):
      self.manager_visualizers["command_manager"] = self.command_manager
    self.manager_visualizers["event_manager"] = self.event_manager
    self.manager_visualizers["reward_manager"] = self.reward_manager

  def load_managers(self) -> None:
    """Load and initialize all managers.

    Order is important! Event and command managers must be loaded first,
    then action and observation managers, then other RL managers.
    """
    # Event manager (required before everything else for domain randomization).
    self.event_manager = EventManager(self.cfg.events, self)
    print_info(f"[INFO] {self.event_manager}")

    self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

    # Command manager (must be before observation manager since observations
    # may reference commands).
    if len(self.cfg.commands) > 0:
      self.command_manager = CommandManager(self.cfg.commands, self)
    else:
      self.command_manager = NullCommandManager()
    print_info(f"[INFO] {self.command_manager}")

    # Action and observation managers.
    self.action_manager = ActionManager(self.cfg.actions, self)
    print_info(f"[INFO] {self.action_manager}")
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print_info(f"[INFO] {self.observation_manager}")

    # Other RL-specific managers.

    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print_info(f"[INFO] {self.termination_manager}")
    self.reward_manager = RewardManager(
      self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
    )
    print_info(f"[INFO] {self.reward_manager}")
    if len(self.cfg.curriculum) > 0:
      self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    else:
      self.curriculum_manager = NullCurriculumManager()
    print_info(f"[INFO] {self.curriculum_manager}")
    if len(self.cfg.metrics) > 0:
      self.metrics_manager = MetricsManager(self.cfg.metrics, self)
    else:
      self.metrics_manager = NullMetricsManager()
    print_info(f"[INFO] {self.metrics_manager}")
    if len(self.cfg.recorders) > 0:
      self.recorder_manager = RecorderManager(self.cfg.recorders, self)
    else:
      self.recorder_manager = NullRecorderManager()
    print_info(f"[INFO] {self.recorder_manager}")

    # Configure spaces for the environment.
    self._configure_gym_env_spaces()

    # Initialize startup events if defined.
    if "startup" in self.event_manager.available_modes:
      self.event_manager.apply(mode="startup")

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self.extras["log"] = dict()
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    # Scoped to env_ids so a partial reset does not advance stateful commands in the
    # other envs.
    self.command_manager.compute(dt=0.0, env_ids=env_ids)
    self.sim.sense()
    # Scoped to env_ids: only the reset envs' history/delay buffers receive
    # the post-reset frame; other envs' observation timelines are untouched.
    self.obs_buf = self.observation_manager.compute(
      update_history=True, env_ids=env_ids
    )
    self.recorder_manager.record_post_reset(env_ids)
    return self.obs_buf, self.extras

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Run one environment step: apply actions, simulate, compute RL signals.

    When ``auto_reset=True`` (default), environments that terminate or time out are
    reset in place and the returned observation is the post-reset state. When
    ``auto_reset=False``, the reset is skipped and the returned observation is the
    terminal state; the caller must call ``reset(env_ids=...)`` for done envs before
    the next ``step()``.

    **Forward-call placement.** MuJoCo's ``mj_step`` runs forward kinematics *before*
    integration, so after stepping, derived quantities (``xpos``, ``xquat``,
    ``site_xpos``, ``cvel``, ``sensordata``) lag ``qpos``/``qvel`` by one physics
    substep. Rather than calling ``sim.forward()`` twice (once after the decimation
    loop and once after the reset block), this method calls it **once**, right
    before observation computation. This single call refreshes derived quantities
    for *all* envs: non-reset envs pick up post-decimation kinematics, reset envs
    pick up post-reset kinematics.

    The tradeoff is that termination, reward, and step/interval event managers see
    derived quantities that are stale by one physics substep (the last ``mj_step``
    ran ``mj_forward`` from *pre*-integration ``qpos``). In practice, the staleness
    is negligible for reward shaping, termination checks, and event perturbations.
    Critically, the staleness is *consistent*: every env, every step, always sees
    the same lag, so the MDP is well-defined and the value function can learn the
    correct mapping.

    **Auto-reset parity.** Auto-reset reproduces the explicit ``reset()`` flow:
    events fire on the terminal state before the reset, and freshly reset envs
    advance commands with ``dt=0`` so their timers start full.

    .. note::

      Events and reset-path resamples all run *before* ``forward()``, so state
      they write (e.g. a velocity push) is refreshed automatically and visible
      in this step's observations. Command updates run *after*
      the call, so a command that writes sim state must refresh derived
      quantities itself (see ``MotionCommand._update_command``). In all cases,
      do not read derived quantities (``root_link_pose_w``, ``body_link_vel_w``,
      etc.) in the same function that writes state (``write_root_state_to_sim``,
      ``write_joint_state_to_sim``, etc.). See :ref:`faq` for details.
    """
    if self._mega_step_graph is not None:
      return self._mega_step_graph.step(action)
    if self._fp32_step_opts_active:
      return self._step_fp32_optimized(action)
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
      self.metrics_manager.compute_substep()

    # Update env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Check terminations and compute rewards.
    # NOTE: Derived quantities (xpos, xquat, ...) are stale by one physics
    # substep here. See the docstring above for why this is acceptable.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Events fire before auto-reset, on the terminal state, matching the
    # explicit flow. Staleness and forward() rules are the same as for
    # termination/reward (see docstring above).
    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    # Reset envs that terminated/timed-out and log the episode info.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_pre_reset(reset_env_ids)
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Single forward() call: recompute derived quantities from current
    # qpos/qvel for every env. For non-reset envs this resolves the
    # one-substep staleness left by mj_step; for reset envs it picks up
    # the freshly written reset state.
    self.sim.forward()

    # Pass dt=0 for freshly reset envs so their command timers start full,
    # matching reset().
    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      command_dt = self._command_dt
      command_dt.fill_(self.step_dt)
      command_dt[reset_env_ids] = 0.0
      self.command_manager.compute(dt=command_dt)
    else:
      self.command_manager.compute(dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and len(reset_env_ids) > 0:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif len(reset_env_ids) > 0:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def get_observations(self) -> dict:
    return self.observation_manager.compute()

  def render(self) -> np.ndarray | None:
    if self.render_mode == "human" or self.render_mode is None:
      return None
    elif self.render_mode == "rgb_array":
      if self._offline_renderer is None:
        raise ValueError("Offline renderer not initialized")
      debug_callback = (
        self.update_visualizers if hasattr(self, "update_visualizers") else None
      )
      self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
      return self._offline_renderer.render()
    else:
      raise NotImplementedError(
        f"Render mode {self.render_mode} is not supported. "
        f"Please use: {self.metadata['render_modes']}."
      )

  def close(self) -> None:
    if getattr(self, "_mega_step_graph", None) is not None:
      self._mega_step_graph.close()
      self._mega_step_graph = None
    if self._offline_renderer is not None:
      self._offline_renderer.close()
    self.recorder_manager.close()

  def begin_rollout_iteration(self, num_steps: int) -> None:
    """Order GPU stream after async reset-pool DMA and enable zero-dilution metric staging."""
    mg = getattr(self, "_mega_step_graph", None)
    if mg is not None:
      mg.begin_iteration(num_steps)
      self._staged_episode_logs_active = True

  def end_rollout_iteration(self) -> list[dict[str, float]]:
    """Flush staged per-step metrics (skipping 0-reset steps) and kick async reset-pool refill."""
    mg = getattr(self, "_mega_step_graph", None)
    self._staged_episode_logs_active = False
    if mg is not None:
      return mg.end_iteration()
    return []

  @staticmethod
  def seed(seed: int = -1) -> int:
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print_info(f"Setting seed: {seed}")
    random_utils.seed_rng(seed)
    return seed

  def update_visualizers(self, visualizer: DebugVisualizer) -> None:
    for mod in self.manager_visualizers.values():
      mod.debug_vis(visualizer)
    for sensor in self.scene.sensors.values():
      sensor.debug_vis(visualizer)

  # Private methods.

  def _configure_gym_env_spaces(self) -> None:
    from mjlab.utils.spaces import batch_space

    self.single_observation_space = DictSpace()
    for group_name, group_term_names in self.observation_manager.active_terms.items():
      has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
      group_dim = self.observation_manager.group_obs_dim[group_name]
      if has_concatenated_obs:
        assert isinstance(group_dim, tuple)
        self.single_observation_space.spaces[group_name] = Box(
          shape=group_dim, low=-math.inf, high=math.inf
        )
      else:
        assert not isinstance(group_dim, tuple)
        group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        # Create a nested dict for this group.
        group_space = DictSpace()
        for term_name, term_dim, _term_cfg in zip(
          group_term_names, group_dim, group_term_cfgs, strict=False
        ):
          group_space.spaces[term_name] = Box(
            shape=term_dim, low=-math.inf, high=math.inf
          )
        self.single_observation_space.spaces[group_name] = group_space

    action_dim = sum(self.action_manager.action_term_dim)
    self.single_action_space = Box(shape=(action_dim,), low=-math.inf, high=math.inf)

    self.observation_space = batch_space(self.single_observation_space, self.num_envs)
    self.action_space = batch_space(self.single_action_space, self.num_envs)

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.curriculum_manager.compute(env_ids=env_ids)
    self.sim.reset(env_ids)
    self.scene.reset(env_ids)

    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )

    # NOTE: This is order sensitive.
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # metrics manager.
    info = self.metrics_manager.reset(env_ids)
    self.extras["log"].update(info)
    # curriculum manager.
    info = self.curriculum_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager.
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
    self._manual_reset_pending[env_ids] = False

  def _can_use_kinematic_post_step_forward(self) -> bool:
    """Conservatively check if post-step forward only needs kinematics + pos/vel sensors."""
    from mjlab.sensor.builtin_sensor import BuiltinSensor
    from mjlab.sensor.contact_sensor import ContactSensor

    if not self.sim.can_use_kinematic_forward(allow_unread_contact_sensors=True):
      return False
    if len(getattr(self.recorder_manager, "active_terms", ())) > 0:
      return False

    forbidden_sensors: set[str] = set()
    for name, sensor in self.scene.sensors.items():
      if isinstance(sensor, ContactSensor):
        forbidden_sensors.add(name)
      elif isinstance(sensor, BuiltinSensor):
        try:
          mj_sensor = self.sim.mj_model.sensor(sensor._prefixed_name)  # type: ignore[attr-defined]
        except Exception:
          mj_sensor = None
        if mj_sensor is None:
          try:
            mj_sensor = self.sim.mj_model.sensor(name)
          except Exception:
            mj_sensor = None
        if mj_sensor is not None and int(mj_sensor.needstage[0]) == int(
          mujoco.mjtStage.mjSTAGE_ACC
        ):
          forbidden_sensors.add(name)
        elif getattr(sensor, "cfg", None) is not None and getattr(
          sensor.cfg, "sensor_type", None
        ) in (
          "accelerometer",
          "force",
          "torque",
          "touch",
          "framelinacc",
          "frameangacc",
        ):
          forbidden_sensors.add(name)

    forbidden_attrs = {
      "qacc",
      "joint_acc",
      "qfrc_actuator",
      "qfrc_external",
      "qfrc_smooth",
      "qfrc_constraint",
      "qfrc_bias",
      "qfrc_passive",
      "actuator_force",
      "nacon",
      "nefc",
    }

    def _value_mentions_forbidden_sensor(val: Any) -> bool:
      if isinstance(val, str):
        return val in forbidden_sensors
      if isinstance(val, dict):
        return any(_value_mentions_forbidden_sensor(v) for v in val.values())
      if isinstance(val, (list, tuple, set)):
        return any(_value_mentions_forbidden_sensor(v) for v in val)
      name_attr = getattr(val, "name", None)
      if isinstance(name_attr, str) and name_attr in forbidden_sensors:
        return True
      return False

    def _callable_mentions_forbidden(fn: Any) -> bool:
      while isinstance(fn, functools.partial):
        if _value_mentions_forbidden_sensor(
          fn.args
        ) or _value_mentions_forbidden_sensor(fn.keywords):
          return True
        fn = fn.func
      code = getattr(fn, "__code__", None)
      if code is None and callable(fn):
        code = getattr(fn.__call__, "__code__", None)
      if code is None:
        return True
      if forbidden_attrs.intersection(code.co_names):
        return True
      for const in code.co_consts:
        if isinstance(const, str) and const in forbidden_sensors:
          return True
      return False

    for group_cfgs in self.observation_manager._group_obs_term_cfgs.values():
      for term_cfg in group_cfgs:
        if _value_mentions_forbidden_sensor(term_cfg.params):
          return False
        if _callable_mentions_forbidden(term_cfg.func):
          return False

    if isinstance(self.command_manager, CommandManager):
      for cmd_term in self.command_manager._terms.values():
        if _callable_mentions_forbidden(
          cmd_term._update_command
        ) or _callable_mentions_forbidden(cmd_term._update_metrics):
          return False

    return True

  def _step_fp32_optimized(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    """Exact FP32 step path with optional substep CUDA graph & kinematic forward."""
    if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
      pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
      raise RuntimeError(
        f"Environments {pending_ids.cpu().tolist()} must be reset via "
        "reset(env_ids=...) before calling step() again when auto_reset=False."
      )

    self.extras["log"] = dict()
    self.action_manager.process_action(action.to(self.device))

    if self._decimation_substep_runner is not None:
      self._decimation_substep_runner.run()
    else:
      for _ in range(self.cfg.decimation):
        self._sim_step_counter += 1
        self.action_manager.apply_action()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=self.physics_dt)
        self.metrics_manager.compute_substep()

    self.episode_length_buf += 1
    self.common_step_counter += 1

    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs

    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    if self.cfg.auto_reset:
      reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
      has_resets = reset_env_ids.numel() > 0 if not self._zero_sync_metrics else None
      if self._zero_sync_metrics:
        # Still check reset count only when auto_reset runs; 1D nonzero shape check
        # is already materialized on CPU by nonzero(), so .numel() is free here!
        has_resets = reset_env_ids.numel() > 0
      if has_resets:
        self.recorder_manager.record_pre_reset(reset_env_ids)
        self._reset_idx(reset_env_ids)
        self.scene.write_data_to_sim()
    else:
      reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
      has_resets = reset_env_ids.numel() > 0

    forward_fn = self.sim.forward
    unwrapped_ok = getattr(forward_fn, "__func__", None) is type(
      self.sim
    ).forward or bool(getattr(forward_fn, "_is_capacity_check_wrapper", False))
    if self._use_kinematic_post_step_forward and unwrapped_ok:
      self.sim.forward_kinematics_and_sensors()
    else:
      forward_fn()

    if self.cfg.auto_reset and has_resets:
      command_dt = self._command_dt
      command_dt.fill_(self.step_dt)
      command_dt[reset_env_ids] = 0.0
      self.command_manager.compute(dt=command_dt)
    else:
      self.command_manager.compute(dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    if self.cfg.auto_reset and has_resets:
      self.recorder_manager.record_post_reset(reset_env_ids)
    elif has_resets:
      self._manual_reset_pending[reset_env_ids] = True

    self.recorder_manager.record_post_step()

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def enable_fp32_step_optimizations(
    self,
    *,
    zero_sync_metrics: bool = True,
    kinematic_forward: bool = True,
    substep_graph: bool = True,
    mega_graph_mode: str | None = None,
    reset_pool_mode: str | None = None,
  ) -> dict[str, Any]:
    """Enable exact FP32 step optimizations (including Tier 1 EnvFullStepGraph) when safe."""
    self._zero_sync_metrics = bool(zero_sync_metrics)
    if kinematic_forward and self._can_use_kinematic_post_step_forward():
      self._use_kinematic_post_step_forward = True
    else:
      self._use_kinematic_post_step_forward = False

    if substep_graph and _DecimationSubstepGraph.is_supported(self):
      if self._decimation_substep_runner is None:
        self._decimation_substep_runner = _DecimationSubstepGraph(self)
    else:
      self._decimation_substep_runner = None

    eff_mega = (
      str(getattr(self.cfg, "_mega_graph_mode", "off"))
      if mega_graph_mode is None
      else str(mega_graph_mode)
    )
    eff_pool = (
      str(getattr(self.cfg, "_reset_pool_mode", "off"))
      if reset_pool_mode is None
      else str(reset_pool_mode)
    )
    if getattr(self.cfg, "_strict_rng_parity", False):
      eff_mega = "off"
      eff_pool = "off"

    mega_reason = "disabled"
    if eff_mega in ("env", "rollout"):
      from mjlab.envs.mega_step_graph import EnvFullStepGraph
      from mjlab.envs.reset_pool import build_mjlab_motion_reset_pool

      ok, mega_reason = EnvFullStepGraph.is_supported(self)
      if ok:
        if self._mega_step_graph is None:
          pools = []
          if eff_pool == "async_ring":
            for cmd_term in self.command_manager._terms.values():
              if (
                type(cmd_term).__name__ == "MotionCommand"
                and getattr(cmd_term, "_async_reset_pool", None) is None
              ):
                pool = build_mjlab_motion_reset_pool(
                  cmd_term,
                  seed=int(getattr(self.cfg, "seed", 0) or 0),
                )
                cmd_term._async_reset_pool = pool  # type: ignore[attr-defined]
                pools.append(
                  (pool, lambda _c=cmd_term: _c.bin_failed_count.detach().cpu().numpy())
                )
          self._mega_step_graph = EnvFullStepGraph(self, reset_pools=pools)
      else:
        if self._mega_step_graph is not None:
          self._mega_step_graph.close()
          self._mega_step_graph = None
    elif self._mega_step_graph is not None:
      self._mega_step_graph.close()
      self._mega_step_graph = None

    self._fp32_step_opts_active = type(self).step is ManagerBasedRlEnv.step and (
      self._zero_sync_metrics
      or self._use_kinematic_post_step_forward
      or self._decimation_substep_runner is not None
    )

    return {
      "zero_sync_metrics": self._zero_sync_metrics,
      "kinematic_post_step_forward": self._use_kinematic_post_step_forward,
      "decimation_substep_graph": self._decimation_substep_runner is not None,
      "mega_step_graph": self._mega_step_graph is not None,
      "mega_step_graph_reason": mega_reason,
    }


class _DecimationSubstepGraph:
  """CUDA-graph capture of standard ManagerBasedRlEnv decimation substeps."""

  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.graph: torch.cuda.CUDAGraph | None = None
    self.signature: tuple[Any, ...] | None = None
    self.warm_steps: int = 0
    self.captures: int = 0
    self.replays: int = 0
    self.stream = torch.cuda.Stream(device=env.device)
    self.warp_stream = wp.Stream(
      device=env.sim.wp_device, cuda_stream=self.stream.cuda_stream
    )

  @staticmethod
  def is_supported(env: ManagerBasedRlEnv) -> bool:
    from mjlab.actuator.builtin_actuator import (
      BuiltinMotorActuator,
      BuiltinMuscleActuator,
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
    )
    from mjlab.actuator.pd_actuator import IdealPdActuator
    from mjlab.envs.mdp.actions.actions import (
      JointEffortAction,
      JointPositionAction,
      JointVelocityAction,
      RelativeJointPositionAction,
    )
    from mjlab.sensor.contact_sensor import ContactSensor
    from mjlab.sensor.sensor import Sensor

    if (
      not str(env.device).startswith("cuda")
      or not env.sim.use_cuda_graph
      or getattr(env.sim.cfg, "deterministic", None) in ("atomics", "all")
    ):
      return False
    if (
      bool(getattr(env.metrics_manager, "_substep_term_indices", ()))
      or env.sim.nan_guard.enabled
    ):
      return False
    allowed_actions = (
      JointPositionAction,
      RelativeJointPositionAction,
      JointVelocityAction,
      JointEffortAction,
    )
    for term in env.action_manager._terms.values():
      if type(term) not in allowed_actions:
        return False
    allowed_actuators = (
      BuiltinPositionActuator,
      BuiltinVelocityActuator,
      BuiltinMotorActuator,
      BuiltinMuscleActuator,
      IdealPdActuator,
    )
    for entity in env.scene.entities.values():
      for act in entity.actuators:
        if type(act) not in allowed_actuators or act.has_delay:
          return False
    for sensor in env.scene.sensors.values():
      if isinstance(sensor, ContactSensor):
        if type(sensor).update is not ContactSensor.update:
          return False
      elif type(sensor).update is not Sensor.update:
        return False
    return True

  def _body(self) -> None:
    env = self.env
    for _ in range(env.cfg.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      env.metrics_manager.compute_substep()

  def run(self) -> None:
    from mjlab.sensor.contact_sensor import ContactSensor

    env = self.env
    histories = tuple(
      buf.data_ptr()
      for sensor in env.scene.sensors.values()
      if isinstance(sensor, ContactSensor) and sensor._history_state is not None
      for buf in sensor._history_state.values()
    )
    signature = (
      id(env.sim.wp_model),
      id(env.sim.wp_data),
      id(env.sim.step_graph),
      getattr(env.sim.step, "__func__", env.sim.step),
      env.cfg.decimation,
      histories,
    )
    if self.signature != signature:
      self.graph = None
      self.warm_steps = 0
      self.signature = signature
    if self.warm_steps < 2:
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      try:
        env.sim._inline_warp_kernels = True
        with torch.inference_mode(False), torch.no_grad():
          self._body()
      finally:
        env.sim._inline_warp_kernels = prev_inline
      self.warm_steps += 1
    else:
      current = torch.cuda.current_stream(env.device)
      self.stream.wait_stream(current)
      ctrl = env.sim.data.ctrl
      ctrl_stream = ctrl._torch_stream
      use_cuda_graph = env.sim.use_cuda_graph
      prev_inline = getattr(env.sim, "_inline_warp_kernels", False)
      was_enabled = gc.isenabled()
      try:
        with (
          torch.inference_mode(False),
          torch.no_grad(),
          torch.cuda.stream(self.stream),
          wp.ScopedStream(self.warp_stream, sync_enter=False, sync_exit=False),
        ):
          if self.graph is None:
            ctrl._torch_stream = self.stream
            env.sim.use_cuda_graph = False
            env.sim._inline_warp_kernels = True
            gc.disable()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
              self.graph,
              stream=self.stream,
              capture_error_mode="thread_local",
            ):
              with wp.ScopedCapture(
                stream=self.warp_stream,
                external=True,
                capture_mode=wp.CaptureMode.THREAD_LOCAL,
              ):
                self._body()
            self.captures += 1
          self.graph.replay()
          self.replays += 1
      finally:
        ctrl._torch_stream = ctrl_stream
        env.sim.use_cuda_graph = use_cuda_graph
        env.sim._inline_warp_kernels = prev_inline
        if was_enabled:
          gc.enable()
        current.wait_stream(self.stream)
      for sensor in env.scene.sensors.values():
        sensor._invalidate_cache()
    env._sim_step_counter += env.cfg.decimation
