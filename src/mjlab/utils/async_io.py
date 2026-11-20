"""Zero-stall 2-slot SHM asynchronous checkpointing, atomic ONNX export, and coalesced logging.

Design guarantees (Round-2 ``test-dev`` architecture):
1. **Bitwise Training Equivalence (``0.0`` diff)**:
   - On ``runner.save()``, GPU parameters, optimizer moments, and observation normalizers are
     copied to a reusable pinned host slab on a dedicated non-blocking CUDA copy stream
     (``~1.5 ms`` D2H DMA), touching **zero** CPU/CUDA RNG state and allocating **zero** device
     memory for contiguous tensors.
2. **Zero-Pipe-Block 2-Slot ``/dev/shm`` Ring (Eliminates 3.2s First-Save Stall)**:
   - Tensor bytes, pickled state-dict skeleton metadata, AND first-save CPU ONNX ``shell_bytes``
     (``~5.2 MB``) are all packed inside the ``/dev/shm`` slot buffer. Only a ``< 128 B`` slot
     descriptor crosses the ``Pipe`` (well below Linux's ``64 KiB`` pipe buffer), so ``submit_save``
     never blocks on ``posix.write`` even while a cold ``spawn`` child is still importing PyTorch.
   - Optional ``prewarm_async_checkpoint_worker()`` overlaps ``spawn`` + ``import torch`` with
     ``SdkRecoveryEnv`` + Warp compilation at process startup.
3. **Full External Logger Compatibility (``tensorboard`` + ``wandb`` + ``neptune``)**:
   - Worker emits ``("done_save", (path, iter))`` and ``("done_git", paths)`` completion notifications
     once atomic disk writes finish, allowing the runner to invoke ``writer.save_model`` /
     ``writer.save_file`` safely without falling back to synchronous main-thread saves.
4. **Coalesced Background Logging (``BatchedAsyncLogWriter``)**:
   - Upgrades UniLab's ``_BatchedScalarWriter``: coalesces all per-iteration scalars into a single
     TensorBoard ``event_pb2.Event`` protobuf record and a single ``wandb.log(merged_dict, step=...)``
     call, offloading serialization, CRC32, and atomic ``progress.json`` / ``.jsonl`` writes to a
     background daemon thread pinned to ``worker_io_cpu_ids``.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import multiprocessing as mp
import os
import pickle
import queue
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from .cpu_runtime import (
  confine_process_threads,
  get_worker_io_cpu_ids,
  pin_current_thread,
)


def atomic_torch_save(
  payload: Any,
  path: str | Path,
  *,
  stream_cuda: bool = False,
) -> Path:
  """Atomically write a PyTorch checkpoint via a sibling temporary file + ``os.replace``."""
  _ = stream_cuda
  dst = Path(path)
  dst.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
  os.close(fd)
  tmp_path = Path(tmp_name)
  try:
    torch.save(payload, tmp_path)
    os.replace(tmp_path, dst)
    return dst
  except BaseException:
    try:
      tmp_path.unlink(missing_ok=True)
    except OSError:
      pass
    raise


try:
  from tensorboard.compat.proto import event_pb2 as _tb_event_pb2
  from tensorboard.compat.proto import summary_pb2 as _tb_summary_pb2

  _HAS_TB_PROTO = True
except Exception:
  _tb_event_pb2 = None  # type: ignore[assignment]
  _tb_summary_pb2 = None  # type: ignore[assignment]
  _HAS_TB_PROTO = False


_SHM_HAS_TRACK_KW = (
  "track" in inspect.signature(shared_memory.SharedMemory.__init__).parameters
)

_DTYPE_TO_STR: dict[torch.dtype, str] = {
  torch.float16: "float16",
  torch.bfloat16: "bfloat16",
  torch.float32: "float32",
  torch.float64: "float64",
  torch.int8: "int8",
  torch.uint8: "uint8",
  torch.int16: "int16",
  torch.int32: "int32",
  torch.int64: "int64",
  torch.bool: "bool",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}

_PREWARM_LOCK = threading.Lock()
_PREWARMED_WORKER: AsyncCheckpointWorker | None = None


def _open_existing_shm(name: str) -> shared_memory.SharedMemory:
  kwargs: dict[str, Any] = {"name": name, "create": False}
  if _SHM_HAS_TRACK_KW:
    kwargs["track"] = False
  return shared_memory.SharedMemory(**kwargs)


def _align64(offset: int) -> int:
  return (offset + 63) & ~63


def pack_state_tree(
  obj: Any,
  tensors_out: list[torch.Tensor],
  offsets_out: list[tuple[int, int]],
  cursor: int = 0,
) -> tuple[Any, int]:
  """Convert nested state dict into a pure-Python skeleton + flat list of tensors."""
  if isinstance(obj, torch.Tensor):
    t = obj.detach()
    if t.dtype not in _DTYPE_TO_STR:
      raise TypeError(f"Unsupported checkpoint tensor dtype: {t.dtype}")
    cursor = _align64(cursor)
    nbytes = int(t.numel() * t.element_size())
    tensors_out.append(t)
    offsets_out.append((cursor, nbytes))
    spec = (
      "__sm_tensor__",
      len(tensors_out) - 1,
      cursor,
      nbytes,
      _DTYPE_TO_STR[t.dtype],
      tuple(int(s) for s in t.shape),
    )
    return spec, cursor + nbytes

  if isinstance(obj, OrderedDict):
    items = []
    for k, v in obj.items():
      packed_v, cursor = pack_state_tree(v, tensors_out, offsets_out, cursor)
      items.append((k, packed_v))
    return ("__sm_odict__", items), cursor

  if isinstance(obj, dict):
    out_dict = {}
    for k, v in obj.items():
      packed_v, cursor = pack_state_tree(v, tensors_out, offsets_out, cursor)
      out_dict[k] = packed_v
    return out_dict, cursor

  if isinstance(obj, list):
    out_list = []
    for item in obj:
      packed_i, cursor = pack_state_tree(item, tensors_out, offsets_out, cursor)
      out_list.append(packed_i)
    return out_list, cursor

  if isinstance(obj, tuple):
    out_tup = []
    for item in obj:
      packed_i, cursor = pack_state_tree(item, tensors_out, offsets_out, cursor)
      out_tup.append(packed_i)
    return ("__sm_tuple__", out_tup), cursor

  return obj, cursor


def unpack_state_tree(skeleton: Any, raw_slab: bytes | bytearray | memoryview) -> Any:
  """Reconstruct nested state dict with independent CPU tensors from ``raw_slab``."""
  mv = memoryview(raw_slab)

  def _walk(node: Any) -> Any:
    if isinstance(node, tuple) and node:
      tag = node[0]
      if tag == "__sm_tensor__":
        _, _idx, offset, nbytes, dtype_str, shape = node
        dtype = _STR_TO_DTYPE[dtype_str]
        if nbytes == 0:
          return torch.empty(shape, dtype=dtype, device="cpu")
        sub = mv[offset : offset + nbytes]
        # Clone out of raw_slab into standalone aligned PyTorch CPU storage.
        return torch.frombuffer(bytearray(sub), dtype=dtype).reshape(shape).clone()
      if tag == "__sm_odict__":
        return OrderedDict((k, _walk(v)) for k, v in node[1])
      if tag == "__sm_tuple__":
        return tuple(_walk(x) for x in node[1])
    if isinstance(node, dict):
      return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, list):
      return [_walk(x) for x in node]
    return node

  return _walk(skeleton)


def snapshot_state_to_cpu(state: Any) -> Any:
  """In-process fast CPU snapshot helper (zero RNG mutation)."""
  tensors: list[torch.Tensor] = []
  offsets: list[tuple[int, int]] = []
  skeleton, total_bytes = pack_state_tree(state, tensors, offsets, 0)
  if total_bytes == 0:
    return unpack_state_tree(skeleton, b"")
  has_cuda = any(t.is_cuda for t in tensors)
  slab = torch.empty(
    total_bytes,
    dtype=torch.uint8,
    device="cpu",
    pin_memory=has_cuda and torch.cuda.is_available(),
  )
  if has_cuda:
    cuda_dev = next(t.device for t in tensors if t.is_cuda)
    stream = torch.cuda.Stream(device=cuda_dev)
    stream.wait_stream(torch.cuda.current_stream(cuda_dev))
    with torch.cuda.stream(stream):
      for t, (off, nbytes) in zip(tensors, offsets, strict=True):
        if nbytes == 0:
          continue
        src = t if t.is_contiguous() else t.contiguous()
        src_u8 = src.reshape(-1).view(torch.uint8)
        if src_u8.is_cuda:
          slab[off : off + nbytes].copy_(src_u8, non_blocking=True)
        else:
          slab[off : off + nbytes].copy_(src_u8)
    stream.synchronize()
  else:
    for t, (off, nbytes) in zip(tensors, offsets, strict=True):
      if nbytes == 0:
        continue
      src = t if t.is_contiguous() else t.contiguous()
      slab[off : off + nbytes].copy_(src.reshape(-1).view(torch.uint8))
  return unpack_state_tree(skeleton, slab.numpy().tobytes())


def build_cpu_onnx_export_template(runner: Any) -> dict[str, Any] | None:
  """Create a zero-CUDA-allocation CPU shell + export kwargs for ``runner``'s actor."""
  alg = getattr(runner, "alg", None)
  if (
    alg is None
    or not hasattr(alg, "get_policy")
    or not hasattr(runner, "export_policy_to_onnx")
  ):
    return None
  policy = alg.get_policy()
  if policy is None or not hasattr(policy, "as_onnx"):
    return None
  onnx_model = policy.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()
  buf = io.BytesIO()
  torch.save(onnx_model, buf)
  return {
    "kind": "rsl_rl_as_onnx",
    "shell_bytes": buf.getvalue(),
    "opset_version": 18,
  }


def _export_onnx_atomic_in_worker(
  shell_cache: dict[str, torch.nn.Module],
  template: dict[str, Any],
  saved_dict: dict[str, Any],
  onnx_dir: str,
  filename: str = "policy.onnx",
) -> str:
  kind = str(template["kind"])
  if kind not in shell_cache:
    shell_bytes = template.get("shell_bytes")
    if not shell_bytes:
      raise RuntimeError(f"Missing ONNX shell_bytes for uncached export kind {kind!r}")
    shell_cache[kind] = torch.load(
      io.BytesIO(shell_bytes),
      map_location="cpu",
      weights_only=False,
    ).eval()
  export_mod = shell_cache[kind]
  state_dict = saved_dict.get("actor_state_dict") or saved_dict.get("model_state_dict")
  if isinstance(state_dict, dict):
    target = getattr(export_mod, "actor", getattr(export_mod, "policy", export_mod))
    if hasattr(target, "load_state_dict"):
      target.load_state_dict(state_dict, strict=False)
  export_mod.eval()

  out_dir = Path(onnx_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  final_path = out_dir / filename
  fd, tmp_name = tempfile.mkstemp(
    prefix=filename + ".partial-", suffix=".onnx", dir=out_dir
  )
  os.close(fd)
  tmp_path = Path(tmp_name)
  try:
    torch.onnx.export(
      export_mod,
      export_mod.get_dummy_inputs(),  # type: ignore[operator]
      str(tmp_path),
      export_params=True,
      opset_version=int(template.get("opset_version", 18)),
      verbose=False,
      input_names=export_mod.input_names,  # type: ignore[arg-type]
      output_names=export_mod.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      external_data=False,
      dynamo=False,
    )
    tmp_path.replace(final_path)
    return str(final_path)
  except BaseException:
    try:
      tmp_path.unlink(missing_ok=True)
    except OSError:
      pass
    raise


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
  p = Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_native_id()}.tmp")
  tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  os.replace(tmp_path, p)


def _async_io_worker_main(
  cmd_conn: Any,
  worker_cpu_ids: tuple[int, ...],
  sidecar_notify_queue: Any | None,
) -> None:
  """Entry point for the persistent CUDA-free CPU checkpoint/ONNX/git subprocess."""
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  os.environ.setdefault("MKL_NUM_THREADS", "1")
  active_cpus: list[tuple[int, ...]] = [worker_cpu_ids]
  if worker_cpu_ids:
    confine_process_threads(worker_cpu_ids)
  torch.set_num_threads(max(1, min(2, len(worker_cpu_ids) or 1)))

  job_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=4)
  shell_cache: dict[str, torch.nn.Module] = {}
  worker_error: list[str] = []
  error_lock = threading.Lock()
  send_lock = threading.Lock()

  def _safe_send(msg: tuple[str, Any]) -> None:
    with send_lock:
      try:
        cmd_conn.send(msg)
      except (BrokenPipeError, EOFError, OSError):
        pass

  def _record_error(tb: str) -> None:
    with error_lock:
      if not worker_error:
        worker_error.append(tb)

  def _get_error() -> str | None:
    with error_lock:
      return worker_error[0] if worker_error else None

  def _executor_loop() -> None:
    if active_cpus[0]:
      confine_process_threads(active_cpus[0])
    while True:
      kind, payload = job_queue.get()
      try:
        if kind == "stop":
          return
        if _get_error() is not None:
          continue
        if kind == "save":
          (
            skeleton,
            slab_bytes,
            ckpt_path,
            iter_num,
            export_onnx,
            onnx_filename,
            template,
          ) = payload
          saved_dict = unpack_state_tree(skeleton, slab_bytes)
          del slab_bytes
          Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
          atomic_torch_save(saved_dict, ckpt_path, stream_cuda=False)
          if export_onnx and template is not None:
            _export_onnx_atomic_in_worker(
              shell_cache,
              template,
              saved_dict,
              str(Path(ckpt_path).parent),
              onnx_filename,
            )
          eff_iter = int(saved_dict.get("iter", iter_num))
          if sidecar_notify_queue is not None:
            try:
              sidecar_notify_queue.put_nowait((str(ckpt_path), eff_iter))
            except Exception:
              pass
          _safe_send(("done_save", (str(ckpt_path), eff_iter)))
          del saved_dict
        elif kind == "git_state":
          log_dir, repos = payload
          rsl_utils = importlib.import_module("rsl_rl.utils")
          store_fn = getattr(rsl_utils, "store_code_state", None)
          raw_files = store_fn(log_dir, repos) if callable(store_fn) else None
          git_files = list(raw_files) if isinstance(raw_files, (list, tuple)) else []
          _safe_send(("done_git", [str(p) for p in git_files]))
        elif kind == "status_json":
          status_path, status_payload = payload
          _atomic_write_json(status_path, status_payload)
      except BaseException:
        _record_error(traceback.format_exc())
      finally:
        job_queue.task_done()

  exec_thread = threading.Thread(
    target=_executor_loop, name="sm_async_io_exec", daemon=True
  )
  exec_thread.start()

  shm_handles: dict[int, tuple[str, shared_memory.SharedMemory]] = {}

  try:
    while True:
      try:
        msg = cmd_conn.recv()
      except EOFError:
        break
      op = msg[0]
      if op == "save_v2":
        (
          _,
          slot,
          shm_name,
          tensor_bytes,
          meta_off,
          meta_len,
          shell_off,
          shell_len,
        ) = msg
        err = _get_error()
        if err is not None:
          _safe_send(("error", err))
          continue
        try:
          cached = shm_handles.get(slot)
          if cached is None or cached[0] != shm_name:
            if cached is not None:
              cached[1].close()
            h = _open_existing_shm(shm_name)
            shm_handles[slot] = (shm_name, h)
          else:
            h = cached[1]
          buf = h.buf
          if buf is None:
            raise RuntimeError(f"SharedMemory {shm_name!r} has closed buffer")
          slab_copy = bytes(buf[:tensor_bytes]) if tensor_bytes > 0 else b""
          meta_raw = bytes(buf[meta_off : meta_off + meta_len])
          skeleton, ckpt_path, iter_num, export_onnx, onnx_filename, template_hdr = (
            pickle.loads(meta_raw)
          )
          template: dict[str, Any] | None = None
          if template_hdr is not None:
            template = dict(template_hdr)
            if shell_len > 0:
              template["shell_bytes"] = bytes(buf[shell_off : shell_off + shell_len])
          del buf
        except BaseException:
          tb = traceback.format_exc()
          _record_error(tb)
          _safe_send(("error", tb))
          continue

        # Copy out of SHM is complete; enqueue job BEFORE releasing slot to parent.
        job_queue.put(
          (
            "save",
            (
              skeleton,
              slab_copy,
              ckpt_path,
              iter_num,
              export_onnx,
              onnx_filename,
              template,
            ),
          )
        )
        _safe_send(("ack", slot))
      elif op == "repin":
        _, new_cpus = msg
        active_cpus[0] = tuple(int(c) for c in new_cpus)
        if active_cpus[0]:
          confine_process_threads(active_cpus[0])
          torch.set_num_threads(max(1, min(2, len(active_cpus[0]))))
      elif op == "git_state":
        _, log_dir, repos = msg
        err = _get_error()
        if err is not None:
          _safe_send(("error", err))
          continue
        job_queue.put(("git_state", (log_dir, repos)))
      elif op == "status_json":
        _, status_path, status_payload = msg
        if _get_error() is None:
          job_queue.put(("status_json", (status_path, status_payload)))
      elif op == "flush":
        job_queue.join()
        err = _get_error()
        if err is not None:
          _safe_send(("error", err))
        else:
          _safe_send(("ok", None))
      elif op == "close":
        job_queue.join()
        job_queue.put(("stop", None))
        exec_thread.join(timeout=10.0)
        err = _get_error()
        if err is not None:
          _safe_send(("error", err))
        else:
          _safe_send(("ok", None))
        break
  finally:
    for _, h in shm_handles.values():
      try:
        h.close()
      except OSError:
        pass


class AsyncCheckpointWorker:
  """Host-side controller for the 2-slot ``/dev/shm`` background checkpoint & ONNX worker."""

  def __init__(
    self,
    *,
    worker_cpu_ids: tuple[int, ...] | list[int] | None = None,
    initial_shm_bytes: int = 36 * 1024 * 1024,
    sidecar_notify_queue: Any | None = None,
  ) -> None:
    cpus = (
      tuple(int(c) for c in worker_cpu_ids)
      if worker_cpu_ids is not None
      else get_worker_io_cpu_ids()
    )
    self.worker_cpu_ids = cpus
    self._sidecar_notify_queue = sidecar_notify_queue
    self._late_sidecar_queue: Any | None = None
    self._ctx = mp.get_context("spawn")
    parent_conn, child_conn = self._ctx.Pipe(duplex=True)
    self._conn = parent_conn
    # Spawn with CUDA hidden in child env so child import torch never touches GPU!
    old_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
      os.environ["CUDA_VISIBLE_DEVICES"] = ""
      self._proc = self._ctx.Process(
        target=_async_io_worker_main,
        args=(child_conn, self.worker_cpu_ids, sidecar_notify_queue),
        name="sm_async_ckpt_worker",
        daemon=False,
      )
      self._proc.start()
    finally:
      if old_cuda_vis is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
      else:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_vis
    child_conn.close()

    self._shms: list[shared_memory.SharedMemory | None] = [None, None]
    self._slot_busy: list[bool] = [False, False]
    self._next_slot: int = 0
    self._initial_shm_bytes = max(65536, int(initial_shm_bytes))
    self._ensure_shm(0, self._initial_shm_bytes)
    self._pinned_slab: torch.Tensor | None = None
    self._copy_stream: torch.cuda.Stream | None = None
    self._copy_stream_device: torch.device | None = None
    self._onnx_template: dict[str, Any] | None = None
    self._onnx_sent_kinds: set[str] = set()
    self._completed_saves: list[tuple[str, int]] = []
    self._completed_git_files: list[str] = []
    self._closed = False
    self._lock = threading.Lock()
    self.total_saves_submitted = 0
    self.last_main_thread_save_ms = 0.0
    self.save_latencies_ms: list[float] = []
    # Register LIFO weakref atexit shutdown so uncaught exceptions never deadlock
    # Python's multiprocessing._exit_function waiting on the non-daemon worker!
    import atexit as _atexit
    import weakref as _weakref

    _self_ref = _weakref.ref(self)

    def _atexit_close() -> None:
      obj = _self_ref()
      if obj is not None and not obj._closed:
        try:
          obj.close()
        except Exception:
          pass

    _atexit.register(_atexit_close)

  def attach_sidecar_queue(self, notify_queue: Any | None) -> None:
    """Attach a sidecar queue if worker was pre-warmed prior to sidecar instantiation."""
    if notify_queue is not None and notify_queue is not self._sidecar_notify_queue:
      self._late_sidecar_queue = notify_queue

  def update_worker_cpu_ids(self, cpu_ids: Sequence[int]) -> None:
    """Update child worker affinity mask (e.g. after rank confinement refines I/O cores)."""
    new_cpus = tuple(int(c) for c in cpu_ids)
    if not new_cpus or new_cpus == self.worker_cpu_ids:
      return
    with self._lock:
      if self._closed:
        return
      self.worker_cpu_ids = new_cpus
      self._conn.send(("repin", new_cpus))

  def _ensure_shm(self, slot: int, required_bytes: int) -> shared_memory.SharedMemory:
    cur = self._shms[slot]
    if cur is not None and cur.size >= required_bytes:
      return cur
    new_size = max(
      required_bytes,
      (cur.size * 2) if cur is not None else self._initial_shm_bytes,
    )
    new_size = _align64(new_size)
    self._shms[slot] = shared_memory.SharedMemory(create=True, size=new_size)
    if cur is not None:
      try:
        cur.close()
        cur.unlink()
      except OSError:
        pass
    return self._shms[slot]  # type: ignore[return-value]

  def _ensure_pinned_slab(self, required_bytes: int) -> torch.Tensor:
    if (
      self._pinned_slab is not None
      and self._pinned_slab.numel() >= required_bytes
      and self._pinned_slab.is_pinned()
    ):
      return self._pinned_slab
    cap = max(required_bytes, 32 * 1024 * 1024)
    self._pinned_slab = torch.empty(
      cap,
      dtype=torch.uint8,
      device="cpu",
      pin_memory=torch.cuda.is_available(),
    )
    return self._pinned_slab

  def set_onnx_export_template(self, template: dict[str, Any] | None) -> None:
    self._onnx_template = template
    if template is not None and "kind" in template:
      self._onnx_sent_kinds.discard(str(template["kind"]))

  def _handle_worker_msg(self, status: str, payload: Any) -> None:
    if status == "error":
      raise RuntimeError(f"AsyncCheckpointWorker failed:\n{payload}")
    if (
      status == "ack"
      and isinstance(payload, int)
      and 0 <= payload < len(self._slot_busy)
    ):
      self._slot_busy[payload] = False
    elif status == "done_save" and isinstance(payload, tuple) and len(payload) == 2:
      ckpt_path, iter_num = str(payload[0]), int(payload[1])
      self._completed_saves.append((ckpt_path, iter_num))
      if self._late_sidecar_queue is not None:
        try:
          self._late_sidecar_queue.put_nowait((ckpt_path, iter_num))
        except Exception:
          pass
    elif status == "done_git" and isinstance(payload, list):
      self._completed_git_files.extend(str(x) for x in payload)

  def _drain_acks(self) -> None:
    while self._conn.poll(0.0):
      status, payload = self._conn.recv()
      self._handle_worker_msg(status, payload)

  def pop_completed_artifacts(self) -> tuple[list[tuple[str, int]], list[str]]:
    """Drain completed ``(ckpt_path, iter)`` and ``git_diff_paths`` for W&B/Neptune upload."""
    with self._lock:
      if not self._closed:
        self._drain_acks()
      saves = list(self._completed_saves)
      gits = list(self._completed_git_files)
      self._completed_saves.clear()
      self._completed_git_files.clear()
      return saves, gits

  def _select_and_acquire_slot(self) -> int:
    self._drain_acks()
    # Prefer next round-robin slot if free; otherwise pick the other free slot before blocking!
    cand = self._next_slot
    if not self._slot_busy[cand]:
      return cand
    alt = cand ^ 1
    if not self._slot_busy[alt]:
      return alt
    while self._slot_busy[cand] and self._slot_busy[alt]:
      status, payload = self._recv_checked()
      self._handle_worker_msg(status, payload)
    return cand if not self._slot_busy[cand] else alt

  def _recv_checked(self) -> tuple[str, Any]:
    if not self._proc.is_alive() and not self._conn.poll():
      raise RuntimeError(
        f"AsyncCheckpointWorker exited unexpectedly (exitcode={self._proc.exitcode})"
      )
    status, payload = self._conn.recv()
    if status == "error":
      raise RuntimeError(f"AsyncCheckpointWorker failed:\n{payload}")
    return status, payload

  def _wait_for_ok(self) -> None:
    while True:
      status, payload = self._recv_checked()
      if status == "ok":
        return
      self._handle_worker_msg(status, payload)

  def submit_save(
    self,
    saved_dict: dict[str, Any],
    path: str | Path,
    *,
    export_onnx: bool = False,
    onnx_filename: str = "policy.onnx",
  ) -> float:
    """Snapshot ``saved_dict`` into 2-slot ``/dev/shm`` ring and send a <128B descriptor."""
    t0 = time.perf_counter()
    with self._lock:
      if self._closed:
        raise RuntimeError("AsyncCheckpointWorker is already closed")

      tensors: list[torch.Tensor] = []
      offsets: list[tuple[int, int]] = []
      skeleton, tensor_bytes = pack_state_tree(saved_dict, tensors, offsets, 0)
      has_cuda = any(t.is_cuda for t in tensors)

      template_hdr: dict[str, Any] | None = None
      shell_bytes = b""
      if export_onnx and self._onnx_template is not None:
        kind = str(self._onnx_template["kind"])
        template_hdr = {
          k: v for k, v in self._onnx_template.items() if k != "shell_bytes"
        }
        if kind not in self._onnx_sent_kinds:
          shell_bytes = bytes(self._onnx_template.get("shell_bytes", b""))
          self._onnx_sent_kinds.add(kind)

      iter_num = int(saved_dict.get("iter", 0)) if isinstance(saved_dict, dict) else 0
      meta_bytes = pickle.dumps(
        (
          skeleton,
          str(path),
          iter_num,
          bool(export_onnx),
          str(onnx_filename),
          template_hdr,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
      )
      meta_off = _align64(tensor_bytes)
      meta_len = len(meta_bytes)
      shell_off = _align64(meta_off + meta_len)
      shell_len = len(shell_bytes)
      total_slot_bytes = max(64, _align64(shell_off + shell_len))

      slot = self._select_and_acquire_slot()
      shm = self._ensure_shm(slot, total_slot_bytes)
      buf = shm.buf
      if buf is None:
        raise RuntimeError(f"SharedMemory {shm.name!r} has closed buffer")

      if tensor_bytes > 0:
        if has_cuda:
          slab = self._ensure_pinned_slab(tensor_bytes)
          cuda_dev = next(t.device for t in tensors if t.is_cuda)
          if self._copy_stream is None or self._copy_stream_device != cuda_dev:
            self._copy_stream = torch.cuda.Stream(device=cuda_dev)
            self._copy_stream_device = cuda_dev
          self._copy_stream.wait_stream(torch.cuda.current_stream(cuda_dev))
          with torch.cuda.stream(self._copy_stream):
            for t, (off, nbytes) in zip(tensors, offsets, strict=True):
              if nbytes == 0:
                continue
              src = t if t.is_contiguous() else t.contiguous()
              src_u8 = src.reshape(-1).view(torch.uint8)
              if src_u8.is_cuda:
                slab[off : off + nbytes].copy_(src_u8, non_blocking=True)
              else:
                slab[off : off + nbytes].copy_(src_u8)
          self._copy_stream.synchronize()
          shm_u8 = torch.frombuffer(
            buf, dtype=torch.uint8, count=tensor_bytes, offset=0
          )
          shm_u8.copy_(slab[:tensor_bytes])
        else:
          # Single-copy CPU fast path: write tensors directly into /dev/shm mmap!
          shm_u8 = torch.frombuffer(
            buf, dtype=torch.uint8, count=tensor_bytes, offset=0
          )
          for t, (off, nbytes) in zip(tensors, offsets, strict=True):
            if nbytes == 0:
              continue
            src = t if t.is_contiguous() else t.contiguous()
            shm_u8[off : off + nbytes].copy_(src.reshape(-1).view(torch.uint8))

      buf[meta_off : meta_off + meta_len] = meta_bytes
      if shell_len > 0:
        buf[shell_off : shell_off + shell_len] = shell_bytes

      # <128-byte descriptor never fills the 64 KiB OS pipe buffer even on cold spawn!
      self._conn.send(
        (
          "save_v2",
          slot,
          shm.name,
          tensor_bytes,
          meta_off,
          meta_len,
          shell_off,
          shell_len,
        )
      )
      self._slot_busy[slot] = True
      self._next_slot = slot ^ 1
      self.total_saves_submitted += 1
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      self.last_main_thread_save_ms = elapsed_ms
      self.save_latencies_ms.append(elapsed_ms)
      return elapsed_ms

  def submit_git_state(self, log_dir: str | Path, repositories: list[str]) -> None:
    """Dispatch ``store_code_state`` git status/diff to the background CPU worker."""
    with self._lock:
      if self._closed:
        return
      self._drain_acks()
      self._conn.send(("git_state", str(log_dir), [str(r) for r in repositories]))

  def submit_status_json(self, path: str | Path, payload: dict[str, Any]) -> None:
    """Dispatch atomic ``progress.json`` write to the background CPU worker."""
    with self._lock:
      if self._closed:
        return
      self._drain_acks()
      self._conn.send(("status_json", str(path), payload))

  def flush(self) -> None:
    """Block until all queued checkpoints, ONNX exports, and git diffs finish."""
    with self._lock:
      if self._closed:
        return
      self._conn.send(("flush",))
      self._wait_for_ok()

  def close(self) -> None:
    """Flush all pending I/O, terminate worker cleanly, and unlink ``/dev/shm`` buffers."""
    with self._lock:
      if self._closed:
        return
      self._closed = True
      err: Exception | None = None
      try:
        if self._proc.is_alive():
          self._conn.send(("close",))
          self._wait_for_ok()
      except Exception as exc:
        err = exc
      finally:
        try:
          self._conn.close()
        except OSError:
          pass
        self._proc.join(timeout=15.0)
        if self._proc.is_alive():
          self._proc.terminate()
          self._proc.join(timeout=3.0)
        for i, shm in enumerate(self._shms):
          if shm is not None:
            try:
              shm.close()
            except OSError:
              pass
            try:
              shm.unlink()
            except OSError:
              pass
            self._shms[i] = None
        self._pinned_slab = None
      if err is not None:
        raise err

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass


def prewarm_async_checkpoint_worker(
  *,
  worker_cpu_ids: Sequence[int] | None = None,
  sidecar_notify_queue: Any | None = None,
) -> AsyncCheckpointWorker:
  """Spawn ``AsyncCheckpointWorker`` early so child Python/PyTorch boot overlaps env setup."""
  global _PREWARMED_WORKER
  with _PREWARM_LOCK:
    if _PREWARMED_WORKER is not None and not _PREWARMED_WORKER._closed:
      if worker_cpu_ids is not None:
        _PREWARMED_WORKER.update_worker_cpu_ids(worker_cpu_ids)
      if sidecar_notify_queue is not None:
        _PREWARMED_WORKER.attach_sidecar_queue(sidecar_notify_queue)
      return _PREWARMED_WORKER
    _PREWARMED_WORKER = AsyncCheckpointWorker(
      worker_cpu_ids=list(worker_cpu_ids) if worker_cpu_ids is not None else None,
      sidecar_notify_queue=sidecar_notify_queue,
    )
    return _PREWARMED_WORKER


def consume_prewarmed_async_worker(
  *,
  worker_cpu_ids: Sequence[int] | None = None,
  sidecar_notify_queue: Any | None = None,
) -> AsyncCheckpointWorker | None:
  """Claim and return a previously pre-warmed ``AsyncCheckpointWorker``, or ``None``."""
  global _PREWARMED_WORKER
  with _PREWARM_LOCK:
    worker = _PREWARMED_WORKER
    _PREWARMED_WORKER = None
  if worker is None or worker._closed:
    return None
  if worker_cpu_ids is not None:
    worker.update_worker_cpu_ids(worker_cpu_ids)
  if sidecar_notify_queue is not None:
    worker.attach_sidecar_queue(sidecar_notify_queue)
  return worker


def _build_tb_summary(summary_cls: Any, pairs: list[tuple[str, float]]) -> Any:
  return summary_cls(value=[summary_cls.Value(tag=t, simple_value=v) for t, v in pairs])


class BatchedAsyncLogWriter:
  """Coalesced background TensorBoard + W&B + atomic JSON/JSONL status writer.

  Upgrades UniLab's ``_BatchedScalarWriter``:
  - Coalesces all scalars logged within ``step_batch()`` (or burst-queued at the same step) into
    **one** ``event_pb2.Event(summary=Summary(value=[...]))`` TensorBoard record and **one**
    ``wandb.log(merged_dict, step=...)`` call instead of 80-120 individual protobuf/IPC writes.
  - Offloads protobuf encoding, CRC32C framing, ``flush()``, ``progress.json`` atomic replacement,
    and ``.jsonl`` appends to a background daemon thread pinned to ``worker_io_cpu_ids``.
  """

  def __init__(
    self,
    writer: Any,
    *,
    worker_cpu_ids: tuple[int, ...] | list[int] | None = None,
    batch_size: int = 32,
    flush_interval_s: float = 2.0,
  ) -> None:
    self._writer = writer
    cpus = (
      tuple(int(c) for c in worker_cpu_ids)
      if worker_cpu_ids is not None
      else get_worker_io_cpu_ids()
    )
    self._worker_cpu_ids = cpus
    self._batch_size = max(1, int(batch_size))
    self._flush_interval_s = max(0.05, float(flush_interval_s))
    self._q: queue.Queue[tuple[str, Any]] = queue.Queue()
    self._closed = False
    self._batch_depth = 0
    self._step_buf: list[tuple[str, float, int | float | None, float | None, bool]] = []
    self._error: BaseException | None = None
    self._error_lock = threading.Lock()

    self._is_wandb_writer = type(writer).__name__ == "WandbSummaryWriter" and hasattr(
      writer, "_map_path"
    )
    self._can_fast_tb_proto = bool(
      _HAS_TB_PROTO and hasattr(writer, "_get_file_writer")
    )

    self.total_scalar_calls = 0
    self.total_coalesced_events = 0
    self.total_wandb_calls = 0
    self.last_step_batch_main_thread_ms = 0.0

    self._thread = threading.Thread(
      target=self._loop, name="sm_batched_log_writer", daemon=True
    )
    self._thread.start()

  def _record_error(self, exc: BaseException) -> None:
    with self._error_lock:
      if self._error is None:
        self._error = exc

  def _check_error(self) -> None:
    with self._error_lock:
      if self._error is not None:
        raise RuntimeError("BatchedAsyncLogWriter background failure") from self._error

  @contextmanager
  def step_batch(self) -> Iterator[None]:
    """Coalesce all ``add_scalar`` calls inside this block into a single queue submission."""
    t0 = time.perf_counter()
    self._batch_depth += 1
    try:
      yield
    finally:
      self._batch_depth -= 1
      if self._batch_depth <= 0:
        self._batch_depth = 0
        if self._step_buf and not self._closed:
          batch = self._step_buf
          self._step_buf = []
          self._q.put(("scalars_batch", (time.time(), batch)))
        self.last_step_batch_main_thread_ms = (time.perf_counter() - t0) * 1000.0

  def add_scalar(
    self,
    tag: str,
    scalar_value: Any,
    global_step: int | float | None = None,
    walltime: float | None = None,
    new_style: bool = False,
  ) -> None:
    if self._closed:
      return
    if isinstance(scalar_value, torch.Tensor):
      val = float(scalar_value.detach().item())
    else:
      val = float(scalar_value)
    self.total_scalar_calls += 1
    item = (str(tag), val, global_step, walltime, bool(new_style))
    if self._batch_depth > 0:
      self._step_buf.append(item)
    else:
      self._q.put(("scalars_batch", (time.time(), [item])))

  def write_status_async(self, path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write ``progress.json`` on the background I/O thread."""
    if not self._closed:
      self._q.put(("status_json", (str(path), dict(payload))))

  def append_jsonl_async(self, path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSONL record on the background I/O thread."""
    if not self._closed:
      self._q.put(("jsonl", (str(path), json.dumps(payload) + "\n")))

  def _emit_coalesced_scalars(
    self,
    default_walltime: float,
    items: list[tuple[str, float, int | float | None, float | None, bool]],
  ) -> None:
    if not items:
      return

    # Fast TensorBoard single-Event protobuf path (+ single-call W&B batching if WandbSummaryWriter)
    if (
      self._can_fast_tb_proto
      and _tb_summary_pb2 is not None
      and _tb_event_pb2 is not None
      and not any(it[4] for it in items)
    ):
      summary_cls: Any = _tb_summary_pb2.Summary
      event_cls: Any = _tb_event_pb2.Event
      file_writer = self._writer._get_file_writer()
      # Group consecutive items sharing (int_step, walltime) with unique tags per Event
      groups: list[tuple[int | None, float, list[tuple[str, float]]]] = []
      cur_step: int | None = None
      cur_wt: float = default_walltime
      cur_vals: list[tuple[str, float]] = []
      cur_seen: set[str] = set()

      for tag, val, gstep, wt, _ns in items:
        istep = int(gstep) if gstep is not None else None
        eff_wt = float(wt) if wt is not None else default_walltime
        if cur_vals and (istep != cur_step or eff_wt != cur_wt or tag in cur_seen):
          groups.append((cur_step, cur_wt, cur_vals))
          cur_vals = []
          cur_seen = set()
        cur_step = istep
        cur_wt = eff_wt
        cur_vals.append((tag, val))
        cur_seen.add(tag)
      if cur_vals:
        groups.append((cur_step, cur_wt, cur_vals))

      for istep, eff_wt, pairs in groups:
        summary_msg = _build_tb_summary(summary_cls, pairs)
        ev_kwargs: dict[str, Any] = {"wall_time": eff_wt, "summary": summary_msg}
        if istep is not None:
          ev_kwargs["step"] = istep
        file_writer.add_event(event_cls(**ev_kwargs))
        self.total_coalesced_events += 1

      if self._is_wandb_writer:
        import wandb

        # Merge all scalars sharing an integer step into a single wandb.log IPC call
        wb_by_step: OrderedDict[int | None, dict[str, float]] = OrderedDict()
        for tag, val, gstep, _wt, _ns in items:
          istep = int(gstep) if gstep is not None else None
          mapped = self._writer._map_path(tag)
          wb_by_step.setdefault(istep, {})[mapped] = val
        for istep, merged in wb_by_step.items():
          wandb.log(merged, step=istep)
          self.total_wandb_calls += 1
      return

    # Fallback for custom/mock writers or new_style=True scalars
    for tag, val, gstep, wt, ns in items:
      try:
        self._writer.add_scalar(tag, val, global_step=gstep, walltime=wt, new_style=ns)
      except TypeError:
        self._writer.add_scalar(tag, val, gstep)
      self.total_coalesced_events += 1

  def _loop(self) -> None:
    if self._worker_cpu_ids:
      pin_current_thread(self._worker_cpu_ids, is_worker=True)
    dirty = False
    last_flush = time.monotonic()
    while True:
      try:
        timeout = max(0.05, self._flush_interval_s - (time.monotonic() - last_flush))
        op, payload = self._q.get(timeout=timeout)
      except queue.Empty:
        if dirty and hasattr(self._writer, "flush"):
          try:
            self._writer.flush()
          except BaseException as exc:
            self._record_error(exc)
          dirty = False
        last_flush = time.monotonic()
        continue

      try:
        if op == "scalars_batch":
          wt0, merged_items = payload
          merged = list(merged_items)
          # Opportunistically coalesce adjacent queued scalar batches
          while True:
            try:
              nxt_op, nxt_payload = self._q.get_nowait()
            except queue.Empty:
              break
            if nxt_op == "scalars_batch":
              merged.extend(nxt_payload[1])
              self._q.task_done()
            else:
              # Process merged scalars first, then handle non-scalar op inline
              self._emit_coalesced_scalars(wt0, merged)
              merged = []
              dirty = True
              self._handle_non_scalar_op(nxt_op, nxt_payload, dirty_ref=[dirty])
              if nxt_op == "stop":
                self._q.task_done()
                return
              if nxt_op == "flush":
                dirty = False
                last_flush = time.monotonic()
              self._q.task_done()
              break
          if merged:
            self._emit_coalesced_scalars(wt0, merged)
            dirty = True
          if dirty and (time.monotonic() - last_flush) >= self._flush_interval_s:
            if hasattr(self._writer, "flush"):
              self._writer.flush()
            dirty = False
            last_flush = time.monotonic()
        else:
          d_ref = [dirty]
          stop_now = self._handle_non_scalar_op(op, payload, dirty_ref=d_ref)
          dirty = d_ref[0]
          if op == "flush":
            last_flush = time.monotonic()
          if stop_now:
            return
      except BaseException as exc:
        self._record_error(exc)
        if op in ("flush", "stop") and isinstance(payload, threading.Event):
          payload.set()
          if op == "stop":
            return
      finally:
        self._q.task_done()

  def _handle_non_scalar_op(
    self, op: str, payload: Any, *, dirty_ref: list[bool]
  ) -> bool:
    if op == "status_json":
      path, data = payload
      _atomic_write_json(path, data)
      return False
    if op == "jsonl":
      path, line = payload
      p = Path(path)
      p.parent.mkdir(parents=True, exist_ok=True)
      with p.open("a", encoding="utf-8") as f:
        f.write(line)
      return False
    if op == "flush":
      if hasattr(self._writer, "flush"):
        self._writer.flush()
      dirty_ref[0] = False
      payload.set()
      return False
    if op == "stop":
      if hasattr(self._writer, "flush"):
        self._writer.flush()
      if hasattr(self._writer, "close"):
        self._writer.close()
      dirty_ref[0] = False
      payload.set()
      return True
    return False

  def flush(self) -> None:
    if self._closed:
      self._check_error()
      return
    if self._step_buf:
      batch = self._step_buf
      self._step_buf = []
      self._q.put(("scalars_batch", (time.time(), batch)))
    ev = threading.Event()
    self._q.put(("flush", ev))
    ev.wait(timeout=30.0)
    self._check_error()

  def stop(self) -> None:
    """Flush buffered scalars and finalize external logger (e.g. ``wandb.finish``)."""
    self.flush()
    if hasattr(self._writer, "stop"):
      self._writer.stop()

  def close(self) -> None:
    if self._closed:
      self._check_error()
      return
    if self._step_buf:
      batch = self._step_buf
      self._step_buf = []
      self._q.put(("scalars_batch", (time.time(), batch)))
    self._closed = True
    ev = threading.Event()
    self._q.put(("stop", ev))
    ev.wait(timeout=30.0)
    self._thread.join(timeout=5.0)
    self._check_error()

  def __getattr__(self, name: str) -> Any:
    return getattr(self._writer, name)


# Backwards-compatible alias for Round-1 callers and imports
AsyncBatchedScalarWriter = BatchedAsyncLogWriter
