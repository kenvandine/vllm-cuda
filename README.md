# vllm-cuda

Prototype build/distribution repo holding self-contained `vllm-server`
bundles for Lemonade's `vllm` backend on NVIDIA CUDA. Like
[`vllm-rocm`](https://github.com/lemonade-sdk/vllm-rocm) and
[`chatterbox-rocm`](https://github.com/lemonade-sdk/chatterbox-rocm), this
repo holds only build artifacts (via GitHub Releases) and CI to produce
them -- no vLLM source is forked or vendored here.

## Why this repo exists

Lemonade's `vllm` backend currently only ships a ROCm build
(`lemonade-sdk/vllm-rocm`), gated to specific AMD GPU families. NVIDIA CUDA
is upstream vLLM's primary supported platform: `pip install vllm` pulls in
the official CUDA-enabled PyTorch wheels as an ordinary dependency, no
special package index required (unlike ROCm, where AMD publishes its own
private wheel index because there's no official upstream CUDA-equivalent
path). This repo automates that install into a portable, self-contained
bundle Lemonade can download and run, the same shape as the other backend
asset repos.

## Why there's no per-GPU-architecture build matrix

llama.cpp ships a separate binary per GPU architecture (`sm_75`, `sm_86`,
`sm_120`, ...) because it's compiled from source for each target. We
initially tried to mirror that here by installing the official (fat,
multi-architecture) `vllm`/`torch` wheels and then pruning them down to one
target GPU architecture with NVIDIA's `nvprune` tool.

**That doesn't work.** `nvprune` operates on unlinked relocatable objects
(ELF type `ET_REL`, produced with `nvcc -rdc=true` before final linking) --
not on already-linked shared libraries (`ET_DYN`), which is exactly what
`pip install torch`/`vllm` gives you. Every `.so` in the official wheels that
actually embeds device code (`libtorch_cuda.so`, vLLM's custom CUDA
extensions, etc.) fails pruning with `nvprune fatal: Input file '...' not
relocatable`. This was confirmed two ways: it failed in real CI (every
target, 0/38 device-code-bearing libraries pruned), and reproduced locally
by downloading a real `torch+cu128` wheel and running `nvprune` against
`libtorch_cuda.so` directly -- same error, independent of CI environment.

A true per-architecture build would mean compiling PyTorch and vLLM's custom
CUDA kernels from source ourselves. That's a materially different (and much
more expensive) undertaking than what `vllm-rocm` does: `vllm-rocm` doesn't
build from source either -- it downloads AMD's own prebuilt nightly wheel
and validates it on a real self-hosted AMD GPU runner
(`dev_lab`/`ta-devlab-halo-03`). NVIDIA/PyPI has no equivalent "prebuilt
per-arch wheel" to repackage, and there's currently no self-hosted NVIDIA GPU
runner in this org's CI to validate a from-source build against. Given that
cost and risk, this repo instead ships the **unmodified, fat, official
wheel** for each host platform:

| Host platform | Asset | Native cubins embedded | PTX fallback |
|---|---|---|---|
| linux-x64 | `linux-x64` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` | `compute_120` (covers `sm_89`/Ada and any newer arch) |
| linux-arm64 | `linux-arm64` | `sm_80, sm_90, sm_100, sm_120` | `compute_120` (covers `sm_121`/GB10/Thor and any newer arch) |

(Verified directly against `torch-2.11.0+cu128` with `cuobjdump`.)

Architectures without a native cubin (`sm_89` Ada on x86_64, `sm_121`
GB10/Jetson Thor on arm64) still run correctly -- the NVIDIA driver JIT-compiles
the nearest same-generation PTX on first launch and caches the result, at the
cost of a one-time startup delay. This mirrors how every ordinary
`pip install vllm` deployment on those GPUs already behaves; Lemonade gets
no worse (and no better) compile-time behavior than a manual install would.

linux-arm64 covers NVIDIA's Grace-CPU-paired SoC/superchip platforms
(GH200 "Grace Hopper", GB200 "Grace Blackwell", GB10 "DGX Spark" /
Jetson Thor) -- ARM64 hosts, not discrete GPUs plugged into an x86_64 PC.
GB10/Thor (`sm_121`) has no x86_64 counterpart at all; GH200/GB200
(`sm_90`/`sm_100`) also exist as discrete x86_64 PCIe cards (H100/H200,
B100/B200), which is why those two architectures' native cubins appear in
*both* rows above.

This can be revisited (either full nvprune-based pruning if NVIDIA ships a
tool that supports linked libraries, or genuine from-source per-arch builds)
once there's a concrete size/performance reason and a way to validate
against real hardware.

## Releases

The [build workflow](.github/workflows/build-release.yml) polls PyPI daily.
When a new `vllm` version appears, it builds and publishes a release tagged
`vllm<version>` (e.g. `vllm0.25.1`) with one asset per host platform:

| Platform | Asset |
|----------|-------|
| Linux x64 | `vllm-server-<tag>-linux-x64-cuda.tar.gz` |
| Linux arm64 | `vllm-server-<tag>-linux-arm64-cuda.tar.gz` |

Multi-GB assets are split into `.partNN-of-MM.tar.gz` + a `.partcount`
manifest (GitHub's 2 GiB per-asset limit), the format Lemonade's
split-archive installer reassembles.

Each bundle is a self-contained, relocatable Python environment
(`python-build-standalone` + the `vllm`/`torch` wheels installed into it)
with a `bin/vllm-server` launcher shim that execs
`python3 -m vllm.entrypoints.openai.api_server`.
