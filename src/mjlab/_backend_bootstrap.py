"""Zero-dependency bootstrap for CLI runtime MuJoCo / MuJoCo-Warp backend selection.

Because ``mujoco`` 3.x and ``mujoco`` 4.x ship mutually exclusive native C
extensions (``libmujoco.so``) and paired ``mujoco-warp`` kernels, switching
physics engines inside an already-imported Python process is unsound. Instead,
when ``--mujoco-backend`` (or ``MJLAB_MUJOCO_BACKEND``) requests a backend other
than ``"auto"`` that does not match the active interpreter, this module
re-executes the process into the corresponding virtual environment *before*
heavy imports (``mujoco``, ``warp``, ``torch``, ``tyro``) occur.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

MujocoBackendChoice = Literal["auto", "3.11", "4.0", "4.0-git"]
VALID_MUJOCO_BACKENDS: tuple[str, ...] = ("auto", "3.11", "4.0", "4.0-git")

_REEXEC_GUARD_ENV = "_MJLAB_BACKEND_REEXEC"
_REEXEC_FROM_ENV = "_MJLAB_BACKEND_REEXEC_FROM"


def _mjlab_repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def normalize_mj4_shorthand_argv(argv: list[str] | None = None) -> list[str]:
  """Expand ``--mj4`` shorthand into ``--mujoco-backend 4.0-git`` before ``--``."""
  args = sys.argv[1:] if argv is None else list(argv)
  out: list[str] = []
  for i, tok in enumerate(args):
    if tok == "--":
      out.extend(args[i:])
      break
    if tok == "--mj4":
      out.extend(["--mujoco-backend", "4.0-git"])
    else:
      out.append(tok)
  return out


def parse_mujoco_backend_arg(
  argv: list[str] | None = None,
  environ: dict[str, str] | None = None,
) -> MujocoBackendChoice:
  """Extract ``--mujoco-backend`` (or ``--mj4`` shorthand) from CLI args or ``MJLAB_MUJOCO_BACKEND``."""
  args = sys.argv[1:] if argv is None else list(argv)
  env = os.environ if environ is None else environ

  cli_value: str | None = None
  i = 0
  while i < len(args):
    tok = args[i]
    if tok == "--":
      break
    if tok == "--mj4":
      cli_value = "4.0-git"
      i += 1
      continue
    if tok == "--mujoco-backend":
      if i + 1 >= len(args):
        raise ValueError(
          f"--mujoco-backend requires a value; valid choices: {VALID_MUJOCO_BACKENDS}"
        )
      cli_value = args[i + 1]
      i += 2
      continue
    if tok.startswith("--mujoco-backend="):
      cli_value = tok.split("=", 1)[1]
      i += 1
      continue
    i += 1

  raw = cli_value if cli_value is not None else env.get("MJLAB_MUJOCO_BACKEND", "auto")
  normalized = raw.strip()
  if normalized not in VALID_MUJOCO_BACKENDS:
    raise ValueError(
      f"Invalid MuJoCo backend {raw!r}. Valid choices: {VALID_MUJOCO_BACKENDS}"
    )
  return normalized  # type: ignore[return-value]


def _dist_version_or_none(dist_name: str) -> str | None:
  try:
    return _metadata.version(dist_name)
  except _metadata.PackageNotFoundError:
    return None


def _mujoco_warp_direct_url_info() -> tuple[bool, str | None]:
  """Return ``(is_editable_or_vcs, url)`` for installed ``mujoco-warp``."""
  try:
    dist = _metadata.distribution("mujoco-warp")
  except _metadata.PackageNotFoundError:
    return False, None

  raw = dist.read_text("direct_url.json")
  if not raw:
    return False, None
  try:
    data = json.loads(raw)
  except json.JSONDecodeError:
    return False, None

  url = data.get("url")
  is_editable = bool(data.get("dir_info", {}).get("editable", False))
  is_vcs = "vcs_info" in data
  return (is_editable or is_vcs), url


def detect_current_backend() -> tuple[str, dict[str, Any]]:
  """Detect active MuJoCo backend using package metadata without loading C libs."""
  mj_ver = _dist_version_or_none("mujoco") or _dist_version_or_none("mujoco-core") or ""
  mj_core_ver = _dist_version_or_none("mujoco-core")
  mjw_ver = _dist_version_or_none("mujoco-warp") or ""
  mjw_git, mjw_url = _mujoco_warp_direct_url_info()

  if mj_ver.startswith("4."):
    resolved = "4.0-git" if mjw_git else "4.0"
  elif mj_ver.startswith("3.11."):
    resolved = "3.11"
  elif mj_ver:
    resolved = mj_ver
  else:
    resolved = "unknown"

  details: dict[str, Any] = {
    "resolved_backend": resolved,
    "mujoco_version": mj_ver or None,
    "mujoco_core_version": mj_core_ver,
    "mujoco_warp_version": mjw_ver or None,
    "mujoco_warp_git_editable": mjw_git,
    "mujoco_warp_source_url": mjw_url,
  }
  return resolved, details


def resolve_backend_python(
  backend: Literal["3.11", "4.0", "4.0-git"],
  environ: dict[str, str] | None = None,
  mjlab_root: Path | None = None,
) -> Path:
  """Locate the Python interpreter for the requested MuJoCo backend."""
  env = os.environ if environ is None else environ
  root = mjlab_root if mjlab_root is not None else _mjlab_repo_root()

  env_var_map = {
    "3.11": "MJLAB_MUJOCO3_PYTHON",
    "4.0": "MJLAB_MUJOCO4_PYTHON",
    "4.0-git": "MJLAB_MUJOCO4_GIT_PYTHON",
  }
  default_venv_map = {
    "3.11": root / ".venv" / "bin" / "python",
    "4.0": root / ".venv-mj4" / "bin" / "python",
    "4.0-git": root / ".venv-mj4-git" / "bin" / "python",
  }

  env_key = env_var_map[backend]
  override = env.get(env_key, "").strip()
  if override:
    candidate = Path(override).expanduser()
    if not (candidate.is_file() and os.access(candidate, os.X_OK)):
      raise RuntimeError(
        f"{env_key}={override!r} for --mujoco-backend={backend} does not point to "
        f"an executable Python binary."
      )
    return candidate

  candidate = default_venv_map[backend]
  if not (candidate.is_file() and os.access(candidate, os.X_OK)):
    raise RuntimeError(
      f"Requested --mujoco-backend={backend}, but interpreter was not found at "
      f"{candidate} (and {env_key} is unset)."
    )
  return candidate


def _matches_requested_backend(
  requested: MujocoBackendChoice,
  detected: str,
  environ: dict[str, str] | None = None,
  mjlab_root: Path | None = None,
) -> bool:
  if requested == "auto":
    return True
  if detected == requested:
    return True

  # If the user explicitly pointed MJLAB_MUJOCO*_PYTHON at the current interpreter,
  # trust that interpreter once verified to have the matching major version.
  env = os.environ if environ is None else environ
  try:
    target = resolve_backend_python(requested, environ=env, mjlab_root=mjlab_root)
  except RuntimeError:
    return False

  cur_bin = os.path.realpath(os.path.dirname(os.path.abspath(sys.executable)))
  tgt_bin = os.path.realpath(os.path.dirname(os.path.abspath(target)))
  if cur_bin == tgt_bin:
    try:
      same_exe = os.path.samefile(sys.executable, target)
    except OSError:
      same_exe = os.path.abspath(sys.executable) == os.path.abspath(target)
  else:
    same_exe = False

  if not same_exe:
    return False
  if requested in ("4.0", "4.0-git"):
    return detected.startswith("4.")
  if requested == "3.11":
    return detected == "3.11"
  return False


def _sanitize_pythonpath_for_target(target_python: Path, pythonpath: str) -> str:
  """Drop ROS site-packages entries compiled for a mismatched Python minor version."""
  if not pythonpath:
    return pythonpath
  lib_dir = target_python.parent.parent / "lib"
  py_dirs = sorted(p.name for p in lib_dir.glob("python3.*") if p.is_dir())
  if not py_dirs:
    return pythonpath
  target_py_tag = py_dirs[0]  # e.g. "python3.12"
  kept: list[str] = []
  for entry in pythonpath.split(os.pathsep):
    if "/opt/ros/" in entry and "python3." in entry and target_py_tag not in entry:
      continue
    kept.append(entry)
  return os.pathsep.join(kept)


def ensure_mujoco_backend_from_argv(
  argv: list[str] | None = None,
  *,
  _allow_exec: bool = True,
) -> MujocoBackendChoice:
  """Ensure the current process runs under the requested ``--mujoco-backend``.

  Fast-path ``"auto"`` performs no filesystem or metadata I/O.
  When another backend is requested and the active interpreter does not match,
  re-executes the current command via :func:`os.execv` into the target venv.
  """
  try:
    requested = parse_mujoco_backend_arg(argv=argv)
  except ValueError as exc:
    if argv is None and _allow_exec:
      print(f"[ERROR] {exc}", file=sys.stderr)
      raise SystemExit(2) from exc
    raise

  if requested == "auto":
    return "auto"

  detected, details = detect_current_backend()
  if _matches_requested_backend(requested, detected):
    return requested

  if os.environ.get(_REEXEC_GUARD_ENV) == requested:
    raise RuntimeError(
      f"Re-executed into {sys.executable} for --mujoco-backend={requested}, "
      f"but detected backend is {detected!r} "
      f"(mujoco={details['mujoco_version']}, "
      f"mujoco-warp={details['mujoco_warp_version']})."
    )

  if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
    raise RuntimeError(
      f"Distributed worker (RANK={os.environ.get('RANK')}) started with "
      f"interpreter {sys.executable} ({detected}) which does not match "
      f"--mujoco-backend={requested}."
    )

  target_python = resolve_backend_python(requested)
  if not _allow_exec:
    return requested

  orig_argv = getattr(sys, "orig_argv", None)
  if orig_argv and len(orig_argv) >= 2 and orig_argv[1] != "-":
    tail_args = list(orig_argv[1:])
  elif sys.argv and sys.argv[0] and sys.argv[0] not in ("-", "-c"):
    tail_args = list(sys.argv)
  else:
    raise RuntimeError(
      f"Cannot re-exec interactive/stdin session into {target_python} for "
      f"--mujoco-backend={requested}. Invoke {target_python} directly or run a script."
    )

  os.environ[_REEXEC_FROM_ENV] = sys.executable
  os.environ[_REEXEC_GUARD_ENV] = requested
  if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = _sanitize_pythonpath_for_target(
      target_python, os.environ["PYTHONPATH"]
    )

  sys.stdout.flush()
  sys.stderr.flush()
  os.execv(str(target_python), [str(target_python), *tail_args])
  return requested  # pragma: no cover


def get_physics_backend_info(
  requested: str | None = None,
) -> dict[str, Any]:
  """Collect version and interpreter provenance for logging and experiment artifacts."""
  if requested is None:
    try:
      requested = parse_mujoco_backend_arg()
    except ValueError:
      requested = "invalid"

  resolved, details = detect_current_backend()
  reexec_target = os.environ.get(_REEXEC_GUARD_ENV)
  return {
    "requested_backend": requested,
    "resolved_backend": resolved,
    "reexecuted": bool(reexec_target),
    "reexecuted_from": os.environ.get(_REEXEC_FROM_ENV),
    "python_executable": sys.executable,
    "mujoco_version": details["mujoco_version"],
    "mujoco_core_version": details["mujoco_core_version"],
    "mujoco_warp_version": details["mujoco_warp_version"],
    "mujoco_warp_git_editable": details["mujoco_warp_git_editable"],
    "mujoco_warp_source_url": details["mujoco_warp_source_url"],
    "warp_version": _dist_version_or_none("warp-lang"),
    "torch_version": _dist_version_or_none("torch"),
  }


def format_physics_backend_banner(requested: str | None = None) -> str:
  """Return a concise one-line human-readable summary of the active physics backend."""
  info = get_physics_backend_info(requested=requested)
  src = " (git editable)" if info["mujoco_warp_git_editable"] else ""
  reexec = " [re-exec]" if info["reexecuted"] else ""
  return (
    f"[INFO] Physics backend: {info['resolved_backend']} "
    f"(requested={info['requested_backend']}, "
    f"mujoco={info['mujoco_version']}, "
    f"mujoco_warp={info['mujoco_warp_version']}{src}, "
    f"warp={info['warp_version']}, "
    f"python={info['python_executable']}){reexec}"
  )
