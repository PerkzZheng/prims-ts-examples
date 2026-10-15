# BF16 per-token sparse attention benchmark

Run `benchmark.py` to compare optimized PrimTS D256 against FlashMLA D512.
It generates the four-row performance table used in
[examples PR #2](https://github.com/PerkzZheng/prims-ts-examples/pull/2), plus
CSV/JSON results containing individual timing samples, accuracy checks, input
hashes, source revisions, and environment information.

| Phase | Batch | Queries per request | Original KV length | Top-k |
| --- | ---: | ---: | ---: | ---: |
| Full causal prefill | 1 | 16384 | 16384 | 512, 2048 |
| MTP decode | 512 | 4 | 16384 | 512, 2048 |

PrimTS uses **D256, Hq=256, Hkv=2, 1CTA**. FlashMLA uses **D512, Hq=128,
Hkv=1, 2CTA**. Both use BF16 Q/K/V/output, FP32 accumulation, shared K=V
within each KV head, and no additional RoPE dimensions. PrimTS's two KV heads
contain independent data and share each query's sparse selection pattern.
`Hq * D` and `Hkv * D` match between variants. The useful QK+PV FLOPs are equal;
D256 has twice as many score/softmax elements.

## Environment and repositories

Use an SM100 or SM103 GPU with at least 24 GiB of free device memory. The
recorded table and this workflow were validated on **GB300/SM103**, with the
PyTorch 26.08 development image, CUDA 13.4, and CUTLASS DSL 4.8.0. SM100 timings
will differ. The pinned FlashMLA source requires an **NVCC 13.1+** compiler,
not just a compatible CUDA runtime.

To use the recorded container environment (requires access to the NVIDIA
registry), start from a directory where the three repositories will live:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace \
  gitlab-master.nvidia.com/dl/dgx/pytorch:26.08-py3-devel bash
```

Inside that environment, clone and pin both kernel repositories:

```bash
git clone --branch prims-ts-d256-token-sparse \
  https://github.com/PerkzZheng/flashinfer.git
git -C flashinfer checkout d9c5811cf5530aa3969761847c0ee1399cff7f18
git -C flashinfer submodule update --init --recursive

git clone https://github.com/deepseek-ai/FlashMLA.git
git -C FlashMLA checkout 2e5429fc5653bab6e081f09477126f731882a6a9
git -C FlashMLA submodule update --init --recursive

git clone --branch examples/d256-token-sparse \
  https://github.com/PerkzZheng/prims-ts-examples.git

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'setuptools>=77' ninja packaging rich \
  'apache-tvm-ffi==0.1.13.post3' 'nvidia-cutlass-dsl[cu13]==4.8.0'

export MAX_JOBS=8 NVCC_THREADS=2 FLASHINFER_NVCC_THREADS=2
BUILD_NVEP=0 python -m pip install --no-build-isolation --no-deps -e ./flashinfer
FLASH_MLA_SKIP_REG_SPILL_CHECK=1 python -m pip install \
  --use-pep517 --no-build-isolation --no-deps -e ./FlashMLA
```

`BUILD_NVEP=0` skips FlashInfer's unrelated MoE communication build.
For this pinned FlashMLA revision under CUDA 13.4, unrelated quantized-decode
kernels trigger its blanket register-spill check. `FLASH_MLA_SKIP_REG_SPILL_CHECK=1`
disables that post-build check; it does not change kernel source or compiler
optimization flags. The comparison uses the unchanged BF16 head128 sparse-forward
kernels. FlashMLA compilation can take several minutes. Rebuild it after changing
its source revision; the runner records the loaded extension's path and SHA-256.

## Run the comparison

From the directory containing all three checkouts:

```bash
export CUDA_VISIBLE_DEVICES=0
python prims-ts-examples/benchmarks/token_sparse_attention/benchmark.py \
  --flashinfer-repo ./flashinfer \
  --flashmla-repo ./FlashMLA \
  --output ./sparse-benchmark-results
```

This runs all four configurations, each with PrimTS followed by FlashMLA in
separate processes on the selected GPU. Each process imports Python code from
the supplied checkout. Installed compiled extensions must match those sources.
Keep the GPU otherwise idle. The script does not change application clocks or
power limits; absolute timings depend on clocks, hardware, and software versions.
A message identifies repositories that differ from the pinned clean revisions,
and `manifest.json` records the actual state.

For a quick correctness/setup check before the full run:

```bash
python prims-ts-examples/benchmarks/token_sparse_attention/benchmark.py \
  --flashinfer-repo ./flashinfer --flashmla-repo ./FlashMLA \
  --quick --output ./sparse-benchmark-smoke
```

Quick mode uses B=1/Q=129/KV=512 cached-prefix prefill and B=4/Q=4/KV=512
decode. Its timings do **not** represent the full table. To rerun just one
full-size case, add `--phase decode --topk 512` to the main command.

Both PrimTS workspace sizing and planning default to `split_kv=True`, permitting
automatic selection. The four full-size configurations select one KV partition
on the validated GB300; smaller cases can split. `--no-split-kv` disables
splitting consistently for both calls. The earlier PR table disabled splitting
for prefill and allowed it for decode; all four cases resolved to one partition.
The required wrapper argument `seq_len_q=1` denotes sparse group size G, while
all 16384 prefill queries or all four MTP queries per request are submitted.

## Inputs, timing, and outputs

The script extracts and executes the actual `_randperm_batch` function from
`FlashMLA/tests/lib.py`, using seed `20261012 + topk`. It does not replace the
helper with a different sampler. Each query samples without replacement from
its causal visible prefix. Early prefill rows contain
`min(topk, query_position + 1)` valid tokens, followed by `-1` padding.

The first backend saves the logical Int32 indices and positions under
`<output>/indices/`. Every worker loads that same file, verifies its SHA-256,
shape, positions, and generator-source hash, and copies it to the GPU. The
comparison checks matching hashes, valid-pair counts, and useful FLOPs before
producing a table. Keep these files to reuse exactly the same selections in
later runs. Q/KV generation uses seed 9012 for each backend.

PrimTS loads token IDs through 64-token physical HND cache pages, passing the
same cache as K and V. FlashMLA's BF16 sparse-forward API receives B*Q flattened
queries and global indices offset by request. This is also the FlashMLA entry
used for decode in the table; its dedicated quantized-KV decode API is not used.
Index generation, request-offset conversion, cache layout conversion, planning,
and JIT compilation occur outside timing.

Each backend checks its output against sampled FP32 PyTorch attention before
and after measurement, covering all query heads of up to 73 representative
queries. The checks retain the original RMS-relative-error limit of 0.01 and
cosine-error limit of 1e-5. Cross-variant outputs are not expected to match,
because their head geometries differ.

Timing follows the original method: five eager warmup calls, capture one
attention call in a CUDA graph, at least **200 ms of graph warmup**, then the
**median of nine samples of 20 replays**. The same buffers are reused without
an L2 scrub. CUDA events measure device time. `--warmup-ms`, `--samples`, and
`--replays` override these settings and are saved in the manifest.

Outputs include:

- `comparison.md`: the two-backend table printed to the terminal.
- `comparison.csv`: latency, relative gap, achieved TFLOP/s, and index hashes.
- `comparison.json`: the table, manifest, and complete per-backend records.
- `<phase>_topk<k>_<backend>.json` / `.log`: raw samples, FP32 checks, GPU and
  compiler information, and worker diagnostics.
- `manifest.json`: source commits, tracked-diff hashes, submodule revisions,
  benchmark-script hash, split setting, and timing options.

Gap is `(PrimTS D256 / FlashMLA D512 - 1) * 100%`; a positive value means D256
is slower. TFLOP/s uses `4 * Hq * D * valid_selected_query_token_pairs` divided
by latency. The FLOP numerator counts QK+PV only; runtime includes softmax and
other work. Generated caches, indices, logs, and results should stay outside
Git. The default local `results/` directory is ignored.
