# vllm-cuda

<a href="https://github.com/lemonade-sdk/vllm-cuda/releases/latest" title="Download the latest release">
  <img src="https://img.shields.io/github/v/release/lemonade-sdk/vllm-cuda?logo=github&logoColor=white" alt="GitHub release (latest by date)" />
</a>
<a href="https://github.com/lemonade-sdk/vllm-cuda/releases/latest" title="View latest release date">
  <img src="https://img.shields.io/github/release-date/lemonade-sdk/vllm-cuda?logo=github&logoColor=white" alt="Latest release date" />
</a>
<a href="LICENSE" title="View license">
  <img src="https://img.shields.io/github/license/lemonade-sdk/vllm-cuda?logo=opensourceinitiative&logoColor=white" alt="License" />
</a>
<a href="https://developer.nvidia.com/cuda-toolkit" title="Powered by CUDA">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white" alt="NVIDIA CUDA" />
</a>
<a href="https://github.com/vllm-project/vllm" title="Powered by vLLM">
  <img src="https://img.shields.io/badge/Powered%20by-vLLM-blue" alt="Powered by vLLM" />
</a>

Distribution repository for self-contained, per-architecture **vLLM + CUDA**
bundles consumed by [Lemonade](https://github.com/lemonade-sdk/lemonade)'s
`vllm` backend (`cuda` device).

**No vLLM code is forked or vendored here.** Unlike AMD ROCm — which has no
official PyPI wheels and needs a from-scratch build against AMD's private
wheel indices (see the sibling
[`vllm-rocm`](https://github.com/lemonade-sdk/vllm-rocm) repo) — NVIDIA CUDA
is upstream vLLM's primary supported platform: `vllm`'s official PyPI wheel
pulls in PyTorch's official CUDA wheel as a normal dependency, and PyTorch's
CUDA wheels bundle the CUDA runtime themselves (via `nvidia-*-cu12`
packages). So this repo's job is narrower than `vllm-rocm`'s: **shrink**
upstream's already-working fat binaries into per-architecture downloads,
rather than **producing** a working ROCm build from scratch.

> **Status: prototype.** This repo currently holds a build/release pipeline
> only. The `nvprune` pruning step below has not yet been validated against a
> real `vllm` release output or on real NVIDIA hardware, and Lemonade does not
> yet consume its releases (that wiring is tracked separately in
> `lemonade-sdk/lemonade`).

## Why per-architecture builds are needed at all

A single official PyTorch CUDA wheel is a **fat binary**: it embeds native
cubins for several compute capabilities plus PTX for the newest one (verified
by inspecting `torch-2.11.0+cu128`'s `libtorch_cuda.so`, which contains
`sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` cubins and `compute_120` PTX).
That's already broader coverage than ROCm gets from one build — no
from-scratch per-arch compile is required — but the fat binary itself is
large (900MB+ for `libtorch_cuda.so` alone) because it carries every
architecture's machine code at once, and vLLM ships several of its own CUDA
extension `.so` files on top of torch's.

To mirror llama.cpp's per-`sm_XX` download-size discipline, this repo
**prunes** the fat wheel down to one architecture per release asset using
NVIDIA's own `nvprune` tool (confirmed available as standalone apt packages
`cuda-nvprune-12-8` / `cuda-cuobjdump-12-8` — no full CUDA Toolkit install
needed) instead of recompiling anything:

1. Install the official `vllm` + `torch` wheels once (fat, all archs).
2. Run `nvprune` over every `.so` under `site-packages` that embeds
   device code, keeping only the target arch's cubin (and its PTX, for
   forward-compat within the same major generation).
3. Package the pruned tree per architecture.

### Target matrix (mirrors `llamacpp`'s CUDA support list in Lemonade)

Verified directly against the actual PyPI/PyTorch wheels (downloaded and
inspected with `cuobjdump`/`strings`), separately for each host architecture
— they don't ship the same native cubins:

| Target | Host arch | Cubin source | Notes |
|--------|-----------|-------------|-------|
| `sm_75` (Turing) | x86_64 | native | RTX 20, GTX 16, T4 |
| `sm_80` (Ampere DC) | x86_64 | native | A100 |
| `sm_86` (Ampere) | x86_64 | native | RTX 30, A40, A6000 |
| `sm_89` (Ada) | x86_64 | PTX JIT from `compute_86` | torch's x86_64 fat wheel has no native `sm_89` cubin; forward-compat PTX JIT applies (same major generation as `sm_86`). First launch pays a JIT-compile cost that vLLM/torch then cache. |
| `sm_90` (Hopper) | x86_64 **and** arm64 | native | H100, H200 (x86_64); GH200 "Grace Hopper" (arm64) |
| `sm_100` (Blackwell DC) | x86_64 **and** arm64 | native | B100, B200 (x86_64); GB200 "Grace Blackwell" (arm64) |
| `sm_120` (Blackwell) | x86_64 | native | RTX 50 |
| `sm_121` (GB10/Thor) | **arm64 only** | PTX JIT from `compute_120` | GB10 ("DGX Spark") and Jetson Thor are Grace-CPU-paired SoC/superchip modules — ARM64 hosts, not discrete GPUs in an x86_64 PC. Built by a separate `build-arm64` job on a GitHub-hosted ARM64 runner (`ubuntu-22.04-arm`), against the `manylinux_2_28_aarch64` vllm/torch wheels. Verified torch's aarch64 fat wheel embeds native `sm_80/90/100/120` cubins (a different, narrower set than the x86_64 wheel) but **no native `sm_121` cubin**, so it falls back to `compute_120` PTX, same JIT-on-first-launch situation as `sm_89` above. |

`sm_89` is therefore identical in content to `sm_86`'s pruned build, and
`sm_121` is identical to `sm_120`'s (arm64) pruned build — they exist as
separate release tags/filenames purely so Lemonade's `get_install_params()`
can key off `SystemInfo::get_cuda_arch()` uniformly, matching the llama.cpp
CUDA convention. This can be revisited once torch ships native cubins for
those architectures.

`sm_90`/`sm_100` are built on **both** host architectures: x86_64 (for
discrete H100/H200/B100/B200 PCIe cards) and arm64 (for GH200/GB200
Grace-paired superchips), each with its own native cubin from that host's
fat wheel. The `build-arm64` job runs all three ARM64 targets
(`sm_90,sm_100,sm_121`) on the same `ubuntu-22.04-arm` runner.

## Releases

The [build workflow](.github/workflows/build-release.yml) polls PyPI daily.
When a new `vllm` version appears, it builds and publishes a release tagged
`vllm<version>` (e.g. `vllm0.25.1`) with one asset per target architecture:

| Platform | Device | Asset |
|----------|--------|-------|
| Linux x64 | CUDA (`sm_75`) | `vllm-server-<tag>-linux-x64-cuda-sm_75.tar.gz` |
| Linux x64 | CUDA (`sm_80`) | `vllm-server-<tag>-linux-x64-cuda-sm_80.tar.gz` |
| Linux x64 | CUDA (`sm_86`) | `vllm-server-<tag>-linux-x64-cuda-sm_86.tar.gz` |
| Linux x64 | CUDA (`sm_89`) | `vllm-server-<tag>-linux-x64-cuda-sm_89.tar.gz` |
| Linux x64 | CUDA (`sm_90`) | `vllm-server-<tag>-linux-x64-cuda-sm_90.tar.gz` |
| Linux x64 | CUDA (`sm_100`) | `vllm-server-<tag>-linux-x64-cuda-sm_100.tar.gz` |
| Linux x64 | CUDA (`sm_120`) | `vllm-server-<tag>-linux-x64-cuda-sm_120.tar.gz` |
| **Linux arm64** | CUDA (`sm_90`) | `vllm-server-<tag>-linux-arm64-cuda-sm_90.tar.gz` |
| **Linux arm64** | CUDA (`sm_100`) | `vllm-server-<tag>-linux-arm64-cuda-sm_100.tar.gz` |
| **Linux arm64** | CUDA (`sm_121`) | `vllm-server-<tag>-linux-arm64-cuda-sm_121.tar.gz` |

Multi-GB assets are split into `.partNN-of-MM.tar.gz` + a `.partcount`
manifest (GitHub's 2 GiB per-asset limit), the format Lemonade's
split-archive installer reassembles.

Windows support is not yet included in this prototype — see the `TODO` in the
build workflow for what's needed to add `windows-x64-cuda-*` assets.

Lemonade would pin the release tag it consumes in
[`backend_versions.json`](https://github.com/lemonade-sdk/lemonade/blob/main/src/cpp/resources/backend_versions.json)
(`vllm.cuda`), analogous to the existing `vllm.rocm` pin, with
`get_install_params()` mapping `SystemInfo::get_cuda_arch()` to the matching
asset suffix (the same pattern `llamacpp` already uses for its own CUDA
builds).

Builds can also be triggered manually from the Actions tab (with an optional
explicit `vllm` version, a `force` rebuild flag, and a comma-separated
`sm_targets` override).
