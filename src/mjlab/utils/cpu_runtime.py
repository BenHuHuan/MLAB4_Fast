"""NUMA-, PCIe-, L3/CCD-, and P/E-core-aware server CPU topology & thread confinement.

Compared with UniLab's ``cpu_runtime.py`` + ``distributed/affinity.py``, this module:
1. Groups SMT sibling logical CPUs by physical core ``(package_id, core_id)``.
2. Detects NUMA node locality, L3/LLC cache domains (e.g. AMD Zen 5 multi-CCD), and
   heterogeneous clock tiers (P-core vs E-core frequency classification).
3. Resolves the active CUDA device's PCIe NUMA node (``/sys/bus/pci/devices/<bdf>/numa_node``)
   and performs stratified P/E + L3/CCD-aligned rank slicing so multi-GPU ranks neither cross
   socket interconnects nor thrash shared L3 CCDs.
4. Splits each rank's physical cores into **disjoint** ``driver_cpu_ids`` (pinned to the main
   CUDA/Warp submission thread + bounded OpenMP pool) and ``worker_cpu_ids``, further separating:
   - ``worker_compute_cpu_ids``: 1 primary non-SMT logical CPU per homogeneous P-core (preferring
     non-driver L3 CCDs) for barrier-synchronized C-MuJoCo ``Batch`` Sim2Sim pools;
   - ``worker_io_cpu_ids``: E-cores or non-driver-CCD worker cores for GIL-free checkpointing,
     ONNX export, and coalesced TensorBoard/W&B/JSON status writers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Iterable, Sequence


_SYS_CPU_ROOT = Path("/sys/devices/system/cpu")
_SYS_NODE_ROOT = Path("/sys/devices/system/node")
_SYS_PCI_ROOT = Path("/sys/bus/pci/devices")
_PROC_TASK_ROOT = Path("/proc/self/task")

# Cores below 85% of peak boost frequency are classified as E-cores on hybrid chips.
_HYBRID_FREQ_RATIO = 0.85

_LOCK = threading.Lock()
_ACTIVE_PLAN: CpuExecutionPlan | None = None
_REGISTERED_WORKER_TIDS: set[int] = set()


@dataclass(frozen=True)
class CpuCoreGroup:
    """One physical CPU core and all of its allowed logical SMT siblings."""

    package_id: int
    core_id: int
    numa_node: int
    max_freq_khz: int
    cpu_ids: tuple[int, ...]
    llc_id: int = 0
    is_performance_core: bool = True


@dataclass(frozen=True)
class CpuTopology:
    """Discovered Linux CPU topology restricted to the current process affinity mask."""

    allowed_cpus: tuple[int, ...]
    physical_groups: tuple[CpuCoreGroup, ...]
    numa_nodes: tuple[int, ...]
    llc_ids: tuple[int, ...] = ()
    is_hybrid_frequency: bool = False

    @property
    def num_logical_cpus(self) -> int:
        return len(self.allowed_cpus)

    @property
    def num_physical_cpus(self) -> int:
        return len(self.physical_groups)

    @property
    def num_physical_cores(self) -> int:
        return len(self.physical_groups)

    @property
    def num_performance_cores(self) -> int:
        count = sum(1 for g in self.physical_groups if g.is_performance_core)
        return count if count > 0 else len(self.physical_groups)

    @property
    def num_efficiency_cores(self) -> int:
        return sum(1 for g in self.physical_groups if not g.is_performance_core)


@dataclass(frozen=True)
class CpuExecutionPlan:
    """Disjoint CPU allocation for one training rank."""

    rank: int
    local_rank: int
    world_size: int
    preferred_numa_node: int | None
    rank_cpu_ids: tuple[int, ...]
    driver_cpu_ids: tuple[int, ...]
    worker_cpu_ids: tuple[int, ...]
    driver_physical_cores: int
    worker_physical_cores: int
    torch_num_threads: int
    worker_compute_cpu_ids: tuple[int, ...] = ()
    worker_io_cpu_ids: tuple[int, ...] = ()
    worker_performance_cores: int = 0
    llc_domains: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "preferred_numa_node": self.preferred_numa_node,
            "llc_domains": list(self.llc_domains),
            "rank_logical_cpus": len(self.rank_cpu_ids),
            "driver_physical_cores": self.driver_physical_cores,
            "worker_physical_cores": self.worker_physical_cores,
            "worker_performance_cores": self.worker_performance_cores,
            "driver_cpu_ids": list(self.driver_cpu_ids),
            "worker_cpu_ids": list(self.worker_cpu_ids),
            "worker_compute_cpu_ids": list(self.worker_compute_cpu_ids),
            "worker_io_cpu_ids": list(self.worker_io_cpu_ids),
            "disjoint_driver_and_worker": bool(
                set(self.driver_cpu_ids).isdisjoint(self.worker_cpu_ids)
            ),
            "torch_num_threads": self.torch_num_threads,
        }


def parse_cpulist(spec: str) -> tuple[int, ...]:
    """Parse Linux cpulist syntax such as ``'0-3,8,12-15'`` into sorted unique ints."""
    spec = spec.strip()
    if not spec:
        return ()
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid CPU range {token!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    return tuple(sorted(out))


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except OSError:
        return default
    except ValueError:
        return default


def _allowed_logical_cpus() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                return tuple(cpus)
        except OSError:
            pass
    count = os.cpu_count() or 1
    return tuple(range(count))


def _discover_cpu_to_numa(
    allowed: Iterable[int],
    node_root: Path = _SYS_NODE_ROOT,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if node_root.is_dir():
        for entry in sorted(node_root.iterdir()):
            name = entry.name
            if not (name.startswith("node") and name[4:].isdigit()):
                continue
            nid = int(name[4:])
            cpulist_file = entry / "cpulist"
            if not cpulist_file.is_file():
                continue
            try:
                for cid in parse_cpulist(cpulist_file.read_text(encoding="utf-8")):
                    mapping[cid] = nid
            except (OSError, ValueError):
                continue
    for cid in allowed:
        mapping.setdefault(cid, 0)
    return mapping


def _discover_cpu_llc_id(cpu_dir: Path, fallback_pkg_id: int) -> int:
    """Resolve Last-Level Cache (L3 / CCD) domain ID for ``cpu_dir``."""
    cache_root = cpu_dir / "cache"
    if not cache_root.is_dir():
        return fallback_pkg_id

    chosen_dir: Path | None = None
    idx3 = cache_root / "index3"
    if idx3.is_dir():
        chosen_dir = idx3
    else:
        best_level = -1
        for entry in sorted(cache_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("index"):
                continue
            lvl = _read_int(entry / "level", -1)
            if lvl >= 2 and lvl >= best_level:
                best_level = lvl
                chosen_dir = entry

    if chosen_dir is None:
        return fallback_pkg_id

    explicit_id = _read_int(chosen_dir / "id", -1)
    if explicit_id >= 0:
        return (fallback_pkg_id << 16) | explicit_id if fallback_pkg_id > 0 else explicit_id

    shared_list = chosen_dir / "shared_cpu_list"
    if shared_list.is_file():
        try:
            shared = parse_cpulist(shared_list.read_text(encoding="utf-8"))
            if shared:
                return (fallback_pkg_id << 16) | shared[0]
        except (OSError, ValueError):
            pass
    return fallback_pkg_id


def discover_cpu_topology(
    allowed_cpus: Sequence[int] | None = None,
    *,
    sys_cpu_root: Path = _SYS_CPU_ROOT,
    sys_node_root: Path = _SYS_NODE_ROOT,
) -> CpuTopology:
    """Discover physical cores, SMT siblings, NUMA nodes, L3 CCDs, and P/E frequency tiers."""
    allowed = (
        tuple(sorted(set(int(x) for x in allowed_cpus)))
        if allowed_cpus is not None
        else _allowed_logical_cpus()
    )
    if not allowed:
        allowed = (0,)

    cpu_to_numa = _discover_cpu_to_numa(allowed, node_root=sys_node_root)
    grouped: dict[tuple[int, int], list[int]] = {}
    freqs: dict[tuple[int, int], int] = {}
    numas: dict[tuple[int, int], int] = {}
    llcs: dict[tuple[int, int], int] = {}

    for cid in allowed:
        cpu_dir = sys_cpu_root / f"cpu{cid}"
        topo_dir = cpu_dir / "topology"
        pkg_id = _read_int(topo_dir / "physical_package_id", 0)
        core_id = _read_int(topo_dir / "core_id", cid)
        key = (pkg_id, core_id)
        grouped.setdefault(key, []).append(cid)
        freq = _read_int(
            cpu_dir / "cpufreq" / "cpuinfo_max_freq",
            _read_int(cpu_dir / "cpufreq" / "scaling_max_freq", 0),
        )
        if freq > freqs.get(key, 0):
            freqs[key] = freq
        numas.setdefault(key, cpu_to_numa.get(cid, 0))
        llcs.setdefault(key, _discover_cpu_llc_id(cpu_dir, pkg_id))

    pos_freqs = [f for f in freqs.values() if f > 0]
    is_hybrid = False
    perf_threshold = 0
    if pos_freqs:
        peak_freq = max(pos_freqs)
        min_freq = min(pos_freqs)
        if min_freq < int(peak_freq * _HYBRID_FREQ_RATIO):
            is_hybrid = True
            perf_threshold = int(peak_freq * _HYBRID_FREQ_RATIO)

    groups = tuple(
        CpuCoreGroup(
            package_id=pkg_id,
            core_id=core_id,
            numa_node=numas[(pkg_id, core_id)],
            max_freq_khz=freqs.get((pkg_id, core_id), 0),
            cpu_ids=tuple(sorted(cids)),
            llc_id=llcs[(pkg_id, core_id)],
            is_performance_core=(not is_hybrid) or (freqs.get((pkg_id, core_id), 0) >= perf_threshold),
        )
        for (pkg_id, core_id), cids in sorted(
            grouped.items(),
            key=lambda kv: (numas[kv[0]], llcs[kv[0]], kv[0][0], kv[0][1], kv[1][0]),
        )
    )
    numa_nodes = tuple(sorted({g.numa_node for g in groups}))
    llc_ids = tuple(sorted({g.llc_id for g in groups}))
    return CpuTopology(
        allowed_cpus=allowed,
        physical_groups=groups,
        numa_nodes=numa_nodes,
        llc_ids=llc_ids,
        is_hybrid_frequency=is_hybrid,
    )


def detect_cuda_numa_node(
    device: str | int | object | None = None,
    *,
    sys_pci_root: Path = _SYS_PCI_ROOT,
) -> int | None:
    """Return the Linux NUMA node attached to ``device``'s PCIe root complex, or None."""
    if device is None:
        return None
    dev_str = str(device)
    if dev_str == "cpu" or dev_str.startswith("cpu:"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if isinstance(device, int):
            idx = device
        elif dev_str.startswith("cuda:"):
            idx = int(dev_str.split(":", 1)[1])
        elif dev_str == "cuda":
            idx = torch.cuda.current_device()
        else:
            idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        domain = int(getattr(props, "pci_domain_id", 0))
        bus = int(getattr(props, "pci_bus_id", -1))
        dev_id = int(getattr(props, "pci_device_id", 0))
        if bus < 0:
            return None
        bdf = f"{domain:04x}:{bus:02x}:{dev_id:02x}.0"
        numa_path = sys_pci_root / bdf / "numa_node"
        if not numa_path.is_file():
            # Scan function suffixes if function != 0
            candidates = sorted(sys_pci_root.glob(f"{domain:04x}:{bus:02x}:{dev_id:02x}.*"))
            if not candidates:
                return None
            numa_path = candidates[0] / "numa_node"
        raw = _read_int(numa_path, -1)
        return raw if raw >= 0 else None
    except Exception:
        return None


def _slice_groups_stratified(
    groups: Sequence[CpuCoreGroup],
    num_slices: int,
    slice_idx: int,
) -> tuple[CpuCoreGroup, ...]:
    """Partition ``groups`` into ``num_slices`` disjoint CCD-aligned, P/E-balanced slices."""
    ordered = sorted(
        groups,
        key=lambda g: (g.numa_node, g.llc_id, g.package_id, g.core_id, g.cpu_ids[0]),
    )
    total = len(ordered)
    if num_slices <= 1 or total <= 1:
        return tuple(ordered)
    if total < num_slices:
        return (ordered[slice_idx % total],)

    p_cores = [g for g in ordered if g.is_performance_core]
    e_cores = [g for g in ordered if not g.is_performance_core]

    # When both P-cores and E-cores exist, give every rank an equal share of P-cores first
    # and fill each rank's exact target quota from E-cores so all ranks get fast driver cores.
    if p_cores and e_cores and len(p_cores) >= num_slices:
        buckets: list[list[CpuCoreGroup]] = [[] for _ in range(num_slices)]
        targets = [
            ((r + 1) * total) // num_slices - (r * total) // num_slices
            for r in range(num_slices)
        ]
        n_p = len(p_cores)
        for r in range(num_slices):
            p_lo = (r * n_p) // num_slices
            p_hi = ((r + 1) * n_p) // num_slices
            buckets[r].extend(p_cores[p_lo:p_hi])
        e_cursor = 0
        for r in range(num_slices):
            need = max(0, targets[r] - len(buckets[r]))
            buckets[r].extend(e_cores[e_cursor : e_cursor + need])
            e_cursor += need
        return tuple(buckets[slice_idx])

    lo = (slice_idx * total) // num_slices
    hi = ((slice_idx + 1) * total) // num_slices
    return tuple(ordered[lo:hi])


def _select_rank_groups(
    topology: CpuTopology,
    local_world_size: int,
    local_rank: int,
    preferred_numa_node: int | None,
) -> tuple[CpuCoreGroup, ...]:
    groups = list(topology.physical_groups)
    total = len(groups)
    if local_world_size <= 1 or total <= 1:
        return tuple(groups)
    if local_rank < 0 or local_rank >= local_world_size:
        raise ValueError(f"local_rank={local_rank} outside [0, {local_world_size})")

    # When NUMA nodes cleanly partition across ranks and preferred_numa_node is valid:
    if preferred_numa_node is not None and len(topology.numa_nodes) > 1:
        local_node_groups = [g for g in groups if g.numa_node == preferred_numa_node]
        ranks_per_node = max(1, local_world_size // len(topology.numa_nodes))
        if len(local_node_groups) >= ranks_per_node:
            sub_rank = local_rank % ranks_per_node
            chosen = _slice_groups_stratified(local_node_groups, ranks_per_node, sub_rank)
            if chosen:
                return chosen

    return _slice_groups_stratified(groups, local_world_size, local_rank)


def partition_server_cpus(
    *,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    local_world_size: int | None = None,
    device: str | int | object | None = None,
    driver_physical_cores: int | None = None,
    topology: CpuTopology | None = None,
    preferred_numa_node: int | None = None,
) -> CpuExecutionPlan:
    """Partition server CPUs into disjoint per-rank driver, compute-worker, and I/O-worker sets.

    - ``driver_cpu_ids`` gets the fastest socket-local P-cores (2 by default on >=6-core ranks,
      1 on 2..5-core ranks) on a single L3 CCD so CUDA/Warp kernel submission never contends with
      background workers.
    - ``worker_compute_cpu_ids`` selects 1 primary logical thread per physical P-core (preferring
      non-driver L3 CCDs) for homogeneous C-MuJoCo ``Batch`` pools without SMT/E-core stragglers.
    - ``worker_io_cpu_ids`` prefers E-cores or non-driver L3 CCD worker cores for async checkpoint
      compression, ONNX export, and coalesced TensorBoard/W&B/status logging.
    """
    topo = topology or discover_cpu_topology()
    eff_local_world = (
        int(local_world_size)
        if local_world_size is not None
        else int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    )
    eff_local_world = max(1, eff_local_world)
    numa = (
        preferred_numa_node
        if preferred_numa_node is not None
        else detect_cuda_numa_node(device)
    )
    if numa is None and len(topo.numa_nodes) == 1:
        numa = topo.numa_nodes[0]

    rank_groups = list(
        _select_rank_groups(topo, eff_local_world, local_rank, numa)
    )
    n_phys = len(rank_groups)

    def _driver_priority(g: CpuCoreGroup) -> tuple[int, int, int, int, int, int]:
        numa_penalty = 0 if (numa is None or g.numa_node == numa) else 1
        ecore_penalty = 0 if g.is_performance_core else 1
        return (numa_penalty, ecore_penalty, -g.max_freq_khz, g.llc_id, g.package_id, g.core_id)

    ordered = sorted(rank_groups, key=_driver_priority)

    if n_phys <= 1:
        drv_groups = ordered
        wrk_groups = ordered
    else:
        if driver_physical_cores is None:
            default_drv = 2 if n_phys >= 6 else 1
        else:
            default_drv = max(1, int(driver_physical_cores))
        n_drv = min(default_drv, n_phys - 1)
        drv_groups = ordered[:n_drv]
        wrk_groups = ordered[n_drv:]

    drv_llcs = {g.llc_id for g in drv_groups}
    p_wrk = [g for g in wrk_groups if g.is_performance_core]
    e_wrk = [g for g in wrk_groups if not g.is_performance_core]

    # Homogeneous compute workers: P-cores only (fallback to wrk_groups if only E-cores remain),
    # 1 primary thread per physical core, non-driver L3 CCDs placed first.
    comp_candidates = p_wrk if p_wrk else list(wrk_groups)
    comp_ordered = sorted(
        comp_candidates,
        key=lambda g: (
            0 if g.llc_id not in drv_llcs else 1,
            -g.max_freq_khz,
            g.llc_id,
            g.package_id,
            g.core_id,
        ),
    )
    worker_compute_cpus = tuple(g.cpu_ids[0] for g in comp_ordered)

    # Background I/O workers: prefer E-cores on hybrid chips, else non-driver L3 CCD worker cores.
    non_drv_wrk = [g for g in wrk_groups if g.llc_id not in drv_llcs]
    if e_wrk:
        io_groups = e_wrk
    elif non_drv_wrk:
        io_groups = non_drv_wrk
    else:
        io_groups = wrk_groups
    worker_io_cpus = tuple(sorted(cid for g in io_groups for cid in g.cpu_ids))

    rank_cpus = tuple(sorted(cid for g in rank_groups for cid in g.cpu_ids))
    driver_cpus = tuple(sorted(cid for g in drv_groups for cid in g.cpu_ids))
    worker_cpus = tuple(sorted(cid for g in wrk_groups for cid in g.cpu_ids))
    llc_domains = tuple(sorted({g.llc_id for g in rank_groups}))
    torch_threads = max(1, min(4, len(drv_groups)))

    return CpuExecutionPlan(
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size),
        preferred_numa_node=numa,
        rank_cpu_ids=rank_cpus,
        driver_cpu_ids=driver_cpus,
        worker_cpu_ids=worker_cpus,
        driver_physical_cores=len(drv_groups),
        worker_physical_cores=len(wrk_groups),
        torch_num_threads=torch_threads,
        worker_compute_cpu_ids=worker_compute_cpus,
        worker_io_cpu_ids=worker_io_cpus,
        worker_performance_cores=len(p_wrk) if p_wrk else len(wrk_groups),
        llc_domains=llc_domains,
    )


def register_worker_thread(tid: int | None = None) -> int:
    """Mark an OS thread TID as a background worker so driver confinement skips it."""
    native_id = threading.get_native_id() if tid is None else int(tid)
    with _LOCK:
        _REGISTERED_WORKER_TIDS.add(native_id)
    return native_id


def pin_current_thread(cpu_ids: Sequence[int], *, is_worker: bool = False) -> None:
    """Pin the calling OS thread (not the whole process) to ``cpu_ids``."""
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return
    tid = threading.get_native_id()
    if is_worker:
        register_worker_thread(tid)
    try:
        os.sched_setaffinity(tid, set(int(c) for c in cpu_ids))
    except OSError:
        pass


def confine_process_threads(
    cpu_ids: Sequence[int],
    *,
    skip_tids: Iterable[int] = (),
) -> int:
    """Pin current process and all non-worker threads in ``/proc/self/task`` to ``cpu_ids``."""
    if not cpu_ids or not hasattr(os, "sched_setaffinity"):
        return 0
    mask = set(int(c) for c in cpu_ids)
    skipped = set(int(t) for t in skip_tids)
    pinned = 0
    if _PROC_TASK_ROOT.is_dir():
        for entry in _PROC_TASK_ROOT.iterdir():
            if not entry.name.isdigit():
                continue
            tid = int(entry.name)
            if tid in skipped:
                continue
            try:
                os.sched_setaffinity(tid, mask)
                pinned += 1
            except OSError:
                continue
    else:
        try:
            os.sched_setaffinity(0, mask)
            pinned = 1
        except OSError:
            pass
    return pinned


def apply_cpu_execution_plan(
    plan: CpuExecutionPlan,
    *,
    configure_torch_threads: bool = True,
) -> dict[str, object]:
    """Apply ``plan`` to the current training rank without touching RNG or GPU state."""
    global _ACTIVE_PLAN
    with _LOCK:
        _ACTIVE_PLAN = plan
        worker_tids = set(_REGISTERED_WORKER_TIDS)

    pinned_threads = confine_process_threads(plan.driver_cpu_ids, skip_tids=worker_tids)
    if configure_torch_threads:
        import torch

        if torch.get_num_threads() != plan.torch_num_threads:
            torch.set_num_threads(plan.torch_num_threads)

    info = plan.to_dict()
    info["pinned_existing_threads"] = pinned_threads
    return info


def get_active_cpu_plan() -> CpuExecutionPlan | None:
    """Return the active ``CpuExecutionPlan`` if configured in this process."""
    with _LOCK:
        return _ACTIVE_PLAN


def get_worker_cpu_ids() -> tuple[int, ...]:
    """Return reserved background worker logical CPUs (or all allowed CPUs if unpartitioned)."""
    plan = get_active_cpu_plan()
    if plan is not None and plan.worker_cpu_ids:
        return plan.worker_cpu_ids
    return _allowed_logical_cpus()


def get_performance_cpu_ids(
    cpu_ids: Sequence[int] | None = None,
    *,
    one_per_core: bool = True,
    topology: CpuTopology | None = None,
) -> tuple[int, ...]:
    """Filter ``cpu_ids`` down to high-frequency P-cores (optionally 1 logical thread per core)."""
    topo = topology or discover_cpu_topology()
    allowed_set = set(int(c) for c in cpu_ids) if cpu_ids is not None else set(topo.allowed_cpus)
    if not allowed_set:
        return ()
    pos_freqs = [g.max_freq_khz for g in topo.physical_groups if g.max_freq_khz > 0]
    threshold = 0
    if pos_freqs:
        peak = max(pos_freqs)
        if min(pos_freqs) < int(peak * _HYBRID_FREQ_RATIO):
            threshold = int(peak * _HYBRID_FREQ_RATIO)

    out: list[int] = []
    for g in topo.physical_groups:
        if threshold > 0 and g.max_freq_khz > 0 and g.max_freq_khz < threshold:
            continue
        if not g.is_performance_core and threshold == 0:
            continue
        present = [cid for cid in g.cpu_ids if cid in allowed_set]
        if not present:
            continue
        if one_per_core:
            out.append(present[0])
        else:
            out.extend(present)
    if not out:
        # Fallback if subset contains only E-cores: still deduplicate SMT siblings if one_per_core
        for g in topo.physical_groups:
            present = [cid for cid in g.cpu_ids if cid in allowed_set]
            if present:
                if one_per_core:
                    out.append(present[0])
                else:
                    out.extend(present)
    return tuple(out)


def get_worker_compute_cpu_ids() -> tuple[int, ...]:
    """Return 1 primary logical CPU per physical P-core for homogeneous compute worker pools."""
    plan = get_active_cpu_plan()
    if plan is not None and plan.worker_compute_cpu_ids:
        return plan.worker_compute_cpu_ids
    return get_performance_cpu_ids(get_worker_cpu_ids(), one_per_core=True)


def get_batch_eval_cpu_ids() -> tuple[int, ...]:
    """Return homogeneous primary P-core logical CPUs for C-MuJoCo ``Batch`` Sim2Sim workers."""
    return get_worker_compute_cpu_ids()


def get_worker_io_cpu_ids() -> tuple[int, ...]:
    """Return background I/O logical CPUs (preferring E-cores or non-driver L3 CCDs)."""
    plan = get_active_cpu_plan()
    if plan is not None and plan.worker_io_cpu_ids:
        return plan.worker_io_cpu_ids
    topo = discover_cpu_topology()
    e_groups = [g for g in topo.physical_groups if not g.is_performance_core]
    if e_groups:
        return tuple(sorted(cid for g in e_groups for cid in g.cpu_ids))
    return get_worker_cpu_ids()


def get_optimal_batch_sim_threads(
    nsim: int | None = None,
    *,
    max_cap: int = 32,
) -> int:
    """Size C-MuJoCo ``Batch`` thread pools to homogeneous physical P-cores (zero SMT/E-core skew)."""
    compute_cpus = get_worker_compute_cpu_ids()
    count = max(1, min(int(max_cap), len(compute_cpus)))
    if nsim is not None:
        count = max(1, min(count, int(nsim)))
    return count


def get_cold_path_worker_count(max_cap: int = 16) -> int:
    """Determine optimal thread count for GIL-releasing cold-path I/O, SHA-256, and MuJoCo forward."""
    env_override = os.environ.get("MJLAB_CPU_IO_WORKERS")
    if env_override is not None:
        try:
            return max(1, int(env_override))
        except ValueError:
            pass
    plan = get_active_cpu_plan()
    if plan is not None:
        perf_cores = plan.worker_performance_cores or plan.worker_physical_cores
        return max(1, min(max_cap, perf_cores, len(plan.worker_cpu_ids)))
    topo = discover_cpu_topology()
    return max(1, min(max_cap, topo.num_performance_cores))


def make_worker_thread_pool(
    max_workers: int | None = None,
    *,
    cpu_ids: Sequence[int] | None = None,
) -> ThreadPoolExecutor:
    """Create a ``ThreadPoolExecutor`` whose OS threads pin themselves to ``worker_cpu_ids``."""
    workers = max_workers if max_workers is not None else get_cold_path_worker_count()
    workers = max(1, int(workers))
    if cpu_ids is not None:
        worker_cpus = tuple(int(c) for c in cpu_ids)
    else:
        plan = get_active_cpu_plan()
        worker_cpus = plan.worker_cpu_ids if plan is not None else ()

    def _init_worker() -> None:
        if worker_cpus:
            pin_current_thread(worker_cpus, is_worker=True)

    return ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="sm_cpu_worker",
        initializer=_init_worker if worker_cpus else None,
    )
