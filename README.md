# MLAB4_Fast

**MLAB4_Fast** is a specialized, high-throughput fork of [`mjlab`](https://github.com/mujocolab/mjlab) pre-configured for **MuJoCo 4.0 + MuJoCo-Warp 4.0** with **lossless IEEE `fp32_exact` acceleration**, **NUMA/P-core CPU isolation**, and **zero-stall Ping-Pong async I/O** permanently enabled out-of-the-box.

---

## Key Features

1. **MuJoCo 4.0 & MuJoCo-Warp 4.0 Native Stack (`mujoco>=4.0.0`, `mujoco-warp>=4.0.0`, `warp-lang>=1.19.0`)**
   - Full MuJoCo 4.0 compatibility patches across `sim`, `contact_sensor`, `entity/data`, `actuator`, and `scripts/train.py` (`mujoco_backend="4.0"` by default).
2. **Strict Lossless IEEE FP32 Two-Tier Mega-Graph (`--fp32-exact` default & `--fp32-fast`)**
   - Strictly enforces 23-bit mantissa IEEE 754 FP32 (`allow_tf32=False`, `fp32_precision="ieee"`, zero BF16/TF32 precision loss).
   - **`fp32_exact` (default, `#2A + #2B + #1A`):** Pre-import OpenMP/NUMA confinement + `ZeroSyncPPO` + Tier 1 full-step `EnvFullStepGraph` (branchless masked resets, `ShiftRegisterBuffer`, `AsyncResetPool` for motion tracking), cutting rollout+update CUDA syncs by **98.9%** (`464 → 5`) and boosting 4,096-env Unitree G1 throughput by **`+28.3%`** (`329.2k FPS` ablation / `336.1k FPS` sustained on RTX PRO 6000).
   - **`fp32_fast` (`--fp32-fast True`, adds `#1B`):** Tier 2 `RolloutStepMegaGraph` fusing actor/critic inference, `EnvFullStepGraph`, reward bootstrapping, and deferred NaN latch into one CUDA graph per rollout step (**`+29.8%`** ablation `333.0k FPS`, sustained **`343.7k FPS` / peak `358.7k FPS`** on RTX PRO 6000; auto-guards `VideoRecorder` and non-Gaussian policies).
   - Pass `--strict-rng-parity True` whenever bit-exact legacy RNG draw ordering vs. stock `mjlab` is required.
3. **Lossless Server CPU, Zero-Sync PPO & Async I/O (`cpu_optimizations=True` by default)**
   - **Pre-Import CPU/OpenMP/NUMA Partitioning** (`mjlab.utils.cpu_runtime`): Initializes `OMP_NUM_THREADS=2`, `OMP_PROC_BIND=close`, `GOMP_SPINCOUNT=200000`, and `interop=1` before `import torch`; separates `driver_primary_cpu_ids` (CUDA/Warp submission) from worker compute, async I/O, and `AsyncResetPool` cores.
   - **Zero-Sync IEEE FP32 PPO** (`mjlab.rl.zero_sync_ppo.ZeroSyncPPO`): GPU `float64` exact KL-LR schedule (`0.0` ULP drift), native fused Adam `found_inf` fail-closed NaN guard, in-place `EmpiricalNormalization`, and 1 coalesced D2H sync per `update()` (`-18.1%` learn time).
   - **Ping-Pong SharedMemory Async Checkpointer** (`AsyncCheckpointWorker`) & **Background Batched Scalar Writer** (`AsyncBatchedScalarWriter`) with LIFO `atexit` shutdown safety.

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

### Train with MuJoCo 4.0 + `fp32_exact` (default) or `fp32_fast`

```bash
# Default: strict IEEE fp32_exact (#2A + #2B + #1A Tier 1 EnvFullStepGraph)
uv run train Mjlab-Velocity-Flat-Unitree-G1

# Max strict IEEE FP32 throughput: fp32_fast (+ #1B Tier 2 RolloutStepMegaGraph)
uv run train Mjlab-Velocity-Flat-Unitree-G1 --fp32-fast True
```

### Override defaults (optional)

```bash
# Bit-exact legacy RNG draw ordering vs stock mjlab
uv run train Mjlab-Velocity-Flat-Unitree-G1 --strict-rng-parity True

# Disable fp32_exact or CPU isolation if debugging
uv run train Mjlab-Velocity-Flat-Unitree-G1 --fp32-exact False --cpu-optimizations False

# Select a different backend
uv run train Mjlab-Velocity-Flat-Unitree-G1 --mujoco-backend 4.0-git
```

### Run Tests

```bash
uv run pytest tests/test_zero_sync_ppo.py tests/test_mega_step_graph.py tests/test_mlab4_fast.py tests/test_mujoco_40_compat.py -q
```

---

## Verified Benchmarks

### NVIDIA RTX PRO 6000 Blackwell (Workstation + Server, Strict IEEE FP32)

Full reports & interactive telemetry: [**Architecture Report (`docs/CPU_MEGA_GRAPH_REPORT.md`)**](docs/CPU_MEGA_GRAPH_REPORT.md) · [**420M-Step Large-Scale Validation (`docs/LARGE_SCALE_VALIDATION_REPORT.md`)**](docs/LARGE_SCALE_VALIDATION_REPORT.md) · [**Interactive HTML Dashboard (`docs/cpu_mega_graph_report.html`)**](docs/cpu_mega_graph_report.html)

| Benchmark (`allow_tf32=False`, Strict IEEE FP32) | Baseline FPS | `fp32_exact` (`#2+#1A`) | `fp32_fast` (`#2+#1A+#1B`) | Highlights |
| :--- | :---: | :---: | :---: | :--- |
| `Velocity-Flat-G1` (4,096 envs, **3 seeds × 300 iters = 265M steps**, Workstation PRO 6000) | `221,609` | **`356,505`** (`+60.9%`) | **`368,373`** (`+66.2%`, peak **`394.9k`**) | Welch `p = 0.882` (rew), `0.969` (`err_xy`) |
| `Velocity-Flat-G1` (**16,384 envs**, Cloud PRO 6000 95 GB) | `492,645` | **`560,528`** (`+13.8%`) | **`565,079`** (`+14.7%`, peak **`568.9k`**) | Only `2.31 GB` VRAM; syncs `464 → 5` (`-98.9%`) |
| `Velocity-Rough-G1` (4,096 envs, raycast + terrain, 2 seeds × 80 iters) | `148,755` | **`175,074`** (`+17.7%`) | **`176,875`** (`+18.9%`, peak `187.4k`) | Syncs `1,015 → 28` (`-97.2%`) |
| `Tracking-Flat-G1` (4,096 envs, **real 131.5s LAFAN1 dance**, `reset_pool_mode="auto"`) | `276,288` | **`378,142`** (`+36.9%`) | **`387,666`** (`+40.3%`, peak `394.8k`) | `auto` live GPU adaptive: `body_pos` `0.1137 vs 0.1138` |

### Async I/O & CPU Isolation (NVIDIA L4 + Local 24-Core Workstation)

| Metric | Sync / Baseline | MLAB4_Fast (`fp32_exact` + CPU/AsyncIO) | Speedup |
| :--- | :---: | :---: | :---: |
| Checkpoint + ONNX Save (L4 steady-state) | `101.47 ms` | **`13.79 ms`** | **`7.4×`** |
| Checkpoint First Save (L4, prewarmed) | `101.47 ms` | **`29.64 ms`** | **`3.4×`** |
| TensorBoard Scalar Logging (50 iters × 80 tags) | `2.685 ms` (`p99 4.233 ms`) | **`0.060 ms`** (`p99 0.269 ms`) | **`44.7×`** |
| Bitwise Accuracy vs Reference (`cuda:0` & CPU) | — | **`0.0` exact match** | Lossless |
| `SdkRecoveryEnv` 5-iter Concurrent Stress (L4) | — | `9.01–23.71 ms` save + `744.2 FPS` CPU Batch Sidecar | Verified |

---

## Acknowledgements

Built on top of [`mujocolab/mjlab`](https://github.com/mujocolab/mjlab), [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco), and [`google-deepmind/mujoco_warp`](https://github.com/google-deepmind/mujoco_warp).
