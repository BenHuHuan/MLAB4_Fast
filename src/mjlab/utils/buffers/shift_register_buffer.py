"""CUDA-Graph-safe chronological shift-register history buffer (zero Python pointer state).

Unlike ``CircularBuffer`` (which advances a Python integer ``_pointer`` on every
``append()`` and therefore freezes inside single-step CUDA Graph replay),
``ShiftRegisterBuffer`` keeps its internal tensor strictly in chronological order
``(batch_size, max_len, *feat_shape)`` (index 0 = oldest, index -1 = newest) and
updates all environments in ONE branchless Warp kernel launch with zero scratch
buffers, zero host synchronization, and zero-copy ``.buffer`` access.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp


@wp.kernel
def _shift_reg_append_masked_f32(
  buf: wp.array3d(dtype=float),
  frame: wp.array2d(dtype=float),
  reset_mask: wp.array(dtype=wp.bool),
  max_len: int,
):
  n, d = wp.tid()
  val = frame[n, d]
  if reset_mask[n]:
    for h in range(max_len):
      buf[n, h, d] = val
  else:
    for h in range(max_len - 1):
      buf[n, h, d] = buf[n, h + 1, d]
    buf[n, max_len - 1, d] = val


@wp.kernel
def _shift_reg_append_with_sdk_bank_f32(
  buf: wp.array3d(dtype=float),
  frame: wp.array2d(dtype=float),
  reset_mask: wp.array(dtype=wp.bool),
  is_sdk_bank: wp.array(dtype=wp.bool),
  bank_row_id: wp.array(dtype=wp.int32),
  sdk_hist: wp.array3d(dtype=float),
  max_len: int,
  feat_dim: int,
  preserve_tail_cols: int,
):
  n, d = wp.tid()
  val = frame[n, d]
  if is_sdk_bank[n]:
    if preserve_tail_cols > 0 and d >= (feat_dim - preserve_tail_cols):
      for h in range(max_len):
        buf[n, h, d] = val
    else:
      row = bank_row_id[n]
      for h in range(max_len):
        buf[n, h, d] = sdk_hist[row, h, d]
  elif reset_mask[n]:
    for h in range(max_len):
      buf[n, h, d] = val
  else:
    for h in range(max_len - 1):
      buf[n, h, d] = buf[n, h + 1, d]
    buf[n, max_len - 1, d] = val


class ShiftRegisterBuffer:
  """Fixed-length chronological history buffer compatible with single-step CUDA Graphs.

  Storage layout is always ``(batch_size, max_len, *feat_shape)`` (index 0 oldest,
  index -1 newest), matching the external view of ``CircularBuffer.buffer`` with zero
  copy or permutation.
  """

  def __init__(self, max_len: int, batch_size: int, device: str) -> None:
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}")

    self._max_len = int(max_len)
    self._batch_size = int(batch_size)
    self._device = str(device)
    self._buffer: torch.Tensor | None = None
    self._feat_shape: tuple[int, ...] = ()
    self._all_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._max_len_tensor = torch.full(
      (batch_size,), self._max_len, dtype=torch.long, device=device
    )
    self._pending_reset_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    self._is_cuda = torch.device(device).type == "cuda"
    if self._is_cuda:
      wp.init()

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def device(self) -> str:
    return self._device

  @property
  def max_length(self) -> int:
    return self._max_len

  @property
  def current_length(self) -> torch.Tensor:
    """Per-batch count of valid frames. Shape: ``(batch_size,)``."""
    return torch.minimum(self._num_pushes, self._max_len_tensor)

  @property
  def is_initialized(self) -> bool:
    return self._buffer is not None

  @property
  def buffer(self) -> torch.Tensor:
    """Chronological history (oldest -> newest), zero-copy view ``(batch_size, max_len, ...)``."""
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")
    return self._buffer

  def _ensure_allocated(self, data: torch.Tensor) -> None:
    if self._buffer is None:
      self._feat_shape = tuple(data.shape[1:])
      self._buffer = torch.zeros(
        (self._batch_size, self._max_len, *self._feat_shape),
        dtype=data.dtype,
        device=self._device,
      )

  def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
    """Zero out values/counters and mark specified rows for backfill on next append."""
    if batch_ids is None:
      self._num_pushes.zero_()
      self._pending_reset_mask.fill_(True)
      if self._buffer is not None:
        self._buffer.zero_()
      return

    if isinstance(batch_ids, torch.Tensor):
      if batch_ids.dtype == torch.bool:
        self.reset_masked(batch_ids)
        return
      if batch_ids.numel() == 0:
        return
    elif len(batch_ids) == 0:
      return

    self._num_pushes[batch_ids] = 0
    self._pending_reset_mask[batch_ids] = True
    if self._buffer is not None:
      self._buffer[batch_ids] = 0.0

  def reset_masked(self, reset_mask: torch.Tensor) -> None:
    """Branchless CUDA-Graph-safe reset using a boolean mask of shape ``(batch_size,)``."""
    mask = reset_mask.to(device=self._device, dtype=torch.bool)
    self._num_pushes.masked_fill_(mask, 0)
    self._pending_reset_mask.logical_or_(mask)
    if self._buffer is not None:
      view_shape = (self._batch_size,) + (1,) * (self._buffer.ndim - 1)
      self._buffer.masked_fill_(mask.view(view_shape), 0.0)

  def backfill(self, data: torch.Tensor, batch_ids: torch.Tensor) -> None:
    """Fill given rows' entire history with one frame, leaving other rows untouched."""
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")
    data = data.to(self._device)
    self._buffer[batch_ids] = data[batch_ids].unsqueeze(1)
    self._num_pushes[batch_ids] = 1
    self._pending_reset_mask[batch_ids] = False

  def append_masked(
    self,
    data: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
    *,
    is_sdk_bank: torch.Tensor | None = None,
    bank_row_id: torch.Tensor | None = None,
    sdk_hist: torch.Tensor | None = None,
    preserve_tail_cols: int = 0,
  ) -> None:
    """Branchless 1-kernel shift-append + reset backfill (CUDA-Graph capturable).

    When ``reset_mask`` is None, consumes internal ``_pending_reset_mask``.
    When ``is_sdk_bank``, ``bank_row_id``, and ``sdk_hist`` are provided, SDK-bank
    reset rows load their full ``max_len`` historical frames directly from GPU VRAM
    in the same single Warp kernel launch.
    """
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")
    data = data.to(self._device)
    self._ensure_allocated(data)
    assert self._buffer is not None

    if reset_mask is None:
      eff_mask = self._pending_reset_mask
    else:
      eff_mask = self._pending_reset_mask | reset_mask.to(
        device=self._device, dtype=torch.bool
      )

    use_sdk = (
      is_sdk_bank is not None and bank_row_id is not None and sdk_hist is not None
    )

    if self._is_cuda and self._buffer.dtype == torch.float32:
      flat_d = int(data[0].numel())
      buf_3d = self._buffer.view(self._batch_size, self._max_len, flat_d)
      frame_2d = data.contiguous().view(self._batch_size, flat_d)
      mask_bool = eff_mask.contiguous()
      torch_stream = torch.cuda.current_stream(self._buffer.device)
      cur_wp = wp.get_stream(str(self._buffer.device))
      wp_stream = (
        cur_wp
        if cur_wp.cuda_stream == torch_stream.cuda_stream
        else wp.stream_from_torch(torch_stream)
      )
      if use_sdk:
        assert (
          is_sdk_bank is not None and bank_row_id is not None and sdk_hist is not None
        )
        sdk_mask = (
          eff_mask & is_sdk_bank.to(device=self._device, dtype=torch.bool)
        ).contiguous()
        row_i32 = bank_row_id.to(device=self._device, dtype=torch.int32).contiguous()
        hist_3d = sdk_hist.contiguous().view(-1, self._max_len, flat_d)
        wp.launch(
          _shift_reg_append_with_sdk_bank_f32,
          dim=(self._batch_size, flat_d),
          inputs=[
            wp.from_torch(buf_3d, dtype=wp.float32),
            wp.from_torch(frame_2d, dtype=wp.float32),
            wp.from_torch(mask_bool, dtype=wp.bool),
            wp.from_torch(sdk_mask, dtype=wp.bool),
            wp.from_torch(row_i32, dtype=wp.int32),
            wp.from_torch(hist_3d, dtype=wp.float32),
            self._max_len,
            flat_d,
            int(preserve_tail_cols),
          ],
          stream=wp_stream,
        )
      else:
        wp.launch(
          _shift_reg_append_masked_f32,
          dim=(self._batch_size, flat_d),
          inputs=[
            wp.from_torch(buf_3d, dtype=wp.float32),
            wp.from_torch(frame_2d, dtype=wp.float32),
            wp.from_torch(mask_bool, dtype=wp.bool),
            self._max_len,
          ],
          stream=wp_stream,
        )
    else:
      # Portable PyTorch fallback (CPU or non-float32 dtype)
      shifted = self._buffer.clone()
      if self._max_len > 1:
        shifted[:, :-1] = self._buffer[:, 1:]
      shifted[:, -1] = data
      view_shape = (self._batch_size,) + (1,) * (self._buffer.ndim - 1)
      backfilled = data.unsqueeze(1).expand_as(self._buffer)
      torch.where(eff_mask.view(view_shape), backfilled, shifted, out=self._buffer)
      if use_sdk:
        assert (
          is_sdk_bank is not None and bank_row_id is not None and sdk_hist is not None
        )
        sdk_mask = eff_mask & is_sdk_bank.to(device=self._device, dtype=torch.bool)
        gathered = sdk_hist.index_select(0, bank_row_id.to(dtype=torch.long)).view_as(
          self._buffer
        )
        if preserve_tail_cols > 0:
          gathered = gathered.clone()
          gathered[..., -preserve_tail_cols:] = data[
            ..., -preserve_tail_cols:
          ].unsqueeze(1)
        torch.where(sdk_mask.view(view_shape), gathered, self._buffer, out=self._buffer)

    # Update push counters branchlessly: reset rows become 1, others increment by 1.
    next_pushes = torch.where(
      eff_mask,
      torch.ones_like(self._num_pushes),
      self._num_pushes + 1,
    )
    if use_sdk:
      assert is_sdk_bank is not None
      sdk_mask = eff_mask & is_sdk_bank.to(device=self._device, dtype=torch.bool)
      next_pushes = torch.where(sdk_mask, self._max_len_tensor, next_pushes)
    self._num_pushes.copy_(next_pushes)
    self._pending_reset_mask.zero_()

  def append(self, data: torch.Tensor) -> None:
    """Append a new frame for all batch elements (backfilling any reset rows)."""
    self.append_masked(data, reset_mask=None)

  def __getitem__(self, key: torch.Tensor | int) -> torch.Tensor:
    """Retrieve lagged frames per batch (LIFO: lag=0 is newest, lag=max_len-1 is oldest)."""
    if self._buffer is None:
      raise RuntimeError("Buffer not initialized. Call append() first.")
    if isinstance(key, int):
      idx = max(0, min(self._max_len - 1, self._max_len - 1 - key))
      return self._buffer[:, idx]
    if key.shape != (self._batch_size,):
      raise ValueError(
        f"Expected key shape ({self._batch_size},), got {tuple(key.shape)}"
      )
    lags = key.to(device=self._device, dtype=torch.long)
    # Match CircularBuffer LIFO clamping against current_length - 1
    valid_lags = torch.minimum(lags, (self.current_length - 1).clamp_min(0)).clamp_min(
      0
    )
    slot_idx = (self._max_len - 1 - valid_lags).clamp(0, self._max_len - 1)
    return self._buffer[self._all_indices, slot_idx]
