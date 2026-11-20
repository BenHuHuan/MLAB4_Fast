# MLAB4_Fast

**MLAB4_Fast** is a specialized, high-throughput fork of [`mjlab`](https://github.com/mujocolab/mjlab) pre-configured for **MuJoCo 4.0 + MuJoCo-Warp 4.0** with **lossless IEEE `fp32_exact` acceleration**, **NUMA/P-core CPU isolation**, and **zero-stall Ping-Pong async I/O** permanently enabled out-of-the-box.

---

## Key Features

1. **MuJoCo 4.0 & MuJoCo-Warp 4.0 Native Stack (`mujoco>=4.0.0`, `mujoco-warp>=4.0.0`, `warp-lang>=1.19.0`)**
   - Full MuJoCo 4.0 compatibility patches across `sim`, `contact_sensor`, `entity/data`, `actuator`, and `scripts/train.py` (`mujoco_backend="4.0"` by default).
2. **Permanently-Enabled `fp32_exact` Mode (`fp32_exact=True` by default)**
   - IEEE FP32 precision (`allow_tf32=False`) + substep CUDA Graphs + zero-sync rollout optimizations with **bitwise `0.0` numerical drift**.
3. **Lossless Server CPU & Async I/O Optimizations (`cpu_optimizations=True` by default)**
   - **NUMA/PCIe & P-Core Partitioning** (`mjlab.utils.cpu_runtime`): Separates `driver_cpu_ids` (CUDA/Warp submission) from `worker_cpu_ids` (background I/O & eval), and filters out E-cores on hybrid CPUs.
   - **Ping-Pong SharedMemory Async Checkpointer** (`AsyncCheckpointWorker`): Eliminates the ~3s first-save cold-start stall and cuts steady-state checkpoint main-thread blocking by **7.4×** (`101.47 ms → 13.79 ms` on L4).
   - **Background Batched Scalar Writer** (`AsyncBatchedScalarWriter`): Coalesces TensorBoard scalar writes on a pinned background thread for a **44.7×** main-thread logging speedup (`2.685 ms → 0.060 ms`).

---

## Installation (via `uv`)

Requires [`uv`](https://docs.astral.sh/uv/) and Python `>=3.11,<3.14`.

### GPU (CUDA 12.8 / 13.x)

```bash
git clone https://github.com/BenHuHuan/MLAB4_Fast.git
cd MLAB4_Fast
uv sync --extra cu128
```

### CPU-only

```bash
git clone https://github.com/BenHuHuan/MLAB4_Fast.git
cd MLAB4_Fast
uv sync --extra cpu
```

> **Note:** On systems with a pre-installed CUDA PyTorch (such as Google Colab), you can also install into an existing environment with `uv pip install -e .`.

---

## Quick Start

### Train with MuJoCo 4.0 + `fp32_exact` + CPU/AsyncIO (all enabled by default)

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1
```

### Override defaults (optional)

```bash
# Disable fp32_exact or CPU isolation if debugging
uv run train Mjlab-Velocity-Flat-Unitree-G1 --fp32-exact False --cpu-optimizations False

# Select a different backend
uv run train Mjlab-Velocity-Flat-Unitree-G1 --mujoco-backend 4.0-git
```

### Run Tests

```bash
uv run pytest tests/test_mlab4_fast.py tests/test_mujoco_40_compat.py -q
```

---

## Verified Benchmarks (NVIDIA L4 + Local 24-Core Workstation)

| Metric | Sync / Baseline | MLAB4_Fast (`fp32_exact` + CPU/AsyncIO) | Speedup |
| :--- | :---: | :---: | :---: |
| Checkpoint + ONNX Save (L4 steady-state) | `101.47 ms` | **`13.79 ms`** | **`7.4×`** |
| Checkpoint First Save (L4, prewarmed) | `101.47 ms` | **`29.64 ms`** | **`3.4×`** |
| TensorBoard Scalar Logging (50 iters × 80 tags) | `2.685 ms` (`p99 4.233 ms`) | **`0.060 ms`** (`p99 0.269 ms`) | **`44.7×`** |
| Bitwise Accuracy vs Reference (`cuda:0` & CPU) | — | **`0.0` exact match** | Lossless |
| `SdkRecoveryEnv` 5-iter Concurrent Stress (L4) | — | `9.01–23.71 ms` save + `744.2 FPS` CPU Batch Sidecar | Verified |

### Tiangong3 Training Collision Optimization (Default Training Analytic Capsules vs. CAD Meshes)

Full root-cause diagnosis, Sim2Real envelope calibration, and multi-regime benchmarks: [**Tiangong3 Capsule Collision Report (`docs/TIANGONG3_CAPSULE_COLLISION_REPORT.md`)**](docs/TIANGONG3_CAPSULE_COLLISION_REPORT.md) (`mjlab.utils.tiangong_capsule_collision.Tiangong3CapsuleCollisionCfg` — enabled by default alongside `fp32_exact` for fresh Tiangong3 training while keeping full CAD meshes for `--resume`, `play`, and Sim2Sim evaluation).

| Benchmark (`tiangong3_v2_29_head_fixed`, MuJoCo 4.0) | Original CAD `lean` | `fast` (`maxhullvert=64`) | Training Default `capsule` (`nmesh=0`) | Gain vs. `lean` |
| :--- | :---: | :---: | :---: | :---: |
| `MjSpec.compile()` Hull Build (`nworld=2048`) | `1,018.99 ms` | `1,008.76 ms` | **`0.65 ms`** | **`1,568×` faster** |
| Pure Physics Step (`2048` Mixed) / Peak VRAM | `45,377 steps/s` / `5,750 MiB` | `75,649 steps/s` / `2,989 MiB` | **`92,626 steps/s` / `1,268 MiB`** | **`2.04×` / `-78.0%` VRAM** |
| Pure Physics Step (`4096` Ground Recovery) | CUDA OOM | `86,004 steps/s` *(tip spike)* | **`103,619 steps/s` / `2,536 MiB`** | **`+20.5%` vs. `fast`** |
| End-to-End `SDK-A` Training (`2048 --fp32-exact`) | `22,668 FPS` (`18.10 h` / 30k) | `31,345 FPS` (`13.08 h` / 30k) | **`35,657 FPS` (`11.49 h` / 30k)** | **`+57.3%` (`-6.6 h`)** |
| 30k `TO19` Policy Zero-Shot Stand / Recovery Parity | `96.88%` / `98.44%` | `96.88%` / `98.44%` | **`96.88%` / `98.44%`** | **`0.00%` drift (`0.273 mm` MAE)** |

---

## Acknowledgements

Built on top of [`mujocolab/mjlab`](https://github.com/mujocolab/mjlab), [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco), and [`google-deepmind/mujoco_warp`](https://github.com/google-deepmind/mujoco_warp).
