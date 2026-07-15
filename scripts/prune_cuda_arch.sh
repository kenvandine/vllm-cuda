#!/usr/bin/env bash
# Prunes every CUDA fat binary (.so) under a site-packages tree down to a
# single target architecture using NVIDIA's own `nvprune`, instead of
# recompiling anything. This is what turns one upstream "all architectures"
# vLLM/PyTorch install into the narrow per-sm_XX release assets this repo
# publishes.
#
# Usage: prune_cuda_arch.sh <site-packages-dir> <native-sm> [ptx-fallback-sm]
#
#   native-sm       Cubin to keep, e.g. "sm_86". Must be an architecture the
#                    fat binary actually ships a native cubin for.
#   ptx-fallback-sm  Optional. Used for targets with no native cubin in the
#                    fat build (sm_89, sm_121 as of torch 2.11.0): pass the
#                    PTX-compatible neighbor (e.g. "sm_86" for sm_89,
#                    "sm_120" for sm_121) and this script keeps that
#                    neighbor's PTX only (-m=yes, no cubin), so the target
#                    GPU JIT-compiles it on first launch. The output is
#                    IDENTICAL to the neighbor's pruned build in that case
#                    (see README's target matrix table) -- it's a separate
#                    release asset only so Lemonade can key off
#                    SystemInfo::get_cuda_arch() uniformly.
#
# PREREQUISITE: cuda-nvprune-12-8 and cuda-cuobjdump-12-8 (or matching
# version) apt packages installed -- NOT the full CUDA Toolkit. See
# build-release.yml for the install step.
set -euo pipefail

SITE_PACKAGES="${1:?site-packages dir required}"
NATIVE_SM="${2:?native sm_XX required, e.g. sm_86}"
PTX_FALLBACK_SM="${3:-}"

NVPRUNE="$(command -v nvprune || true)"
CUOBJDUMP="$(command -v cuobjdump || true)"
[ -n "$NVPRUNE" ] || { echo "::error::nvprune not found on PATH"; exit 1; }
[ -n "$CUOBJDUMP" ] || { echo "::error::cuobjdump not found on PATH"; exit 1; }

if [ -n "$PTX_FALLBACK_SM" ]; then
  GENCODE="-gencode=arch=compute_${PTX_FALLBACK_SM#sm_},code=compute_${PTX_FALLBACK_SM#sm_}"
  echo "Target ${NATIVE_SM}: no native cubin in upstream build; keeping" \
       "${PTX_FALLBACK_SM} PTX only (JIT fallback) -- see README caveat."
else
  GENCODE="-gencode=arch=compute_${NATIVE_SM#sm_},code=${NATIVE_SM}"
fi

pruned=0
skipped=0
failed=0

# Only .so files that actually embed device code are worth touching --
# cuobjdump exits non-zero (and prints nothing useful) on pure-host .so
# files, so use that as the "has device code" probe.
while IFS= read -r -d '' so_file; do
  if ! "$CUOBJDUMP" -lelf "$so_file" >/dev/null 2>&1; then
    skipped=$((skipped + 1))
    continue
  fi

  tmp_out="${so_file}.pruned"
  if "$NVPRUNE" "$GENCODE" -o "$tmp_out" "$so_file" 2>/tmp/nvprune_err.log; then
    orig_size=$(wc -c < "$so_file")
    new_size=$(wc -c < "$tmp_out")
    mv -f "$tmp_out" "$so_file"
    pruned=$((pruned + 1))
    echo "pruned: $so_file (${orig_size} -> ${new_size} bytes)"
  else
    echo "::warning::nvprune failed on $so_file, leaving it untouched:"
    cat /tmp/nvprune_err.log
    rm -f "$tmp_out"
    failed=$((failed + 1))
  fi
done < <(find "$SITE_PACKAGES" -name "*.so*" -type f -print0)

echo "Pruning summary for ${NATIVE_SM}: pruned=${pruned} skipped(no-device-code)=${skipped} failed=${failed}"

# A failed prune is not fatal by itself (the file just stays fat), but zero
# successful prunes means something is structurally wrong (wrong site-packages
# path, nvprune broken, etc.) and the resulting asset would be no smaller
# than the unpruned build -- treat that as a hard failure.
if [ "$pruned" -eq 0 ]; then
  echo "::error::no .so files were pruned -- refusing to publish an unpruned 'per-arch' asset"
  exit 1
fi
