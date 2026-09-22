# PrimTS examples

Standalone examples for FlashInfer's PrimTS attention APIs. Kernels remain in
FlashInfer; this repository demonstrates their public interfaces.

## QToken-KvBlock-Sparse-Attention

Run [q_token_kv_block_sparse_attention.py](q_token_kv_block_sparse_attention.py)
to exercise both layouts:

| Example | Q layout | Group selection |
| --- | --- | --- |
| Variable-length chunked prefill | `[total_q, Hq, D]` with request-safe `qo_indptr` | Suggested once from request count, maximum Q length, selected candidates, heads, and cached SM count |
| Uniform MTP3 decode | `[B, num_query_groups, G, Hq, D]` | Suggested once; multiple groups preserve all four query tokens if G is smaller than four |

Both use `suggest_q_token_kv_block_sparse_group_size`, then keep G fixed for
the prepared metadata/attention plan. Device properties are queried once in
`main`, never in the prepared hot path. Partial prefill groups do not cross
request boundaries.

The default prefill chunks contain 1025 and 513 queries, each ending at an
8K context. These are cached-prefix chunks, not two complete 8K prefills.
The default demonstrates G5 and a partial final group on current SM100/SM103
devices. Decode uses BS8 and MTP3; it warms the plan, captures `run`, and
replays a CUDA graph. This is an API example, not a performance benchmark.

The synthetic BF16 inputs use Hq/Hkv=12/1, D256, 512 selected logical blocks,
`kv_block_size=4`, and physical page size 16. For compactness, requests share
physical K/V pages; production frameworks normally supply distinct mappings.
Only sparse block size 4 is implemented by this FlashInfer specialization.

## Setup for QToken-KvBlock-Sparse-Attention

Use a CUDA-enabled PyTorch environment and an SM100 or SM103 GPU. The example
requires the API in [FlashInfer PR #4996](https://github.com/flashinfer-ai/flashinfer/pull/4996),
not an arbitrary released FlashInfer wheel. Install the qualified feature
source and CUTLASS DSL (the commands below use a CUDA 13 environment):

```bash
git clone --recursive --branch qsa-packed-query-official-pr \
  https://github.com/PerkzZheng/flashinfer.git
git -C flashinfer checkout d138709ecbb95c3fb3892fbef355fed745fe4a5a
python -m pip install 'setuptools>=77' 'nvidia-cutlass-dsl[cu13]==4.7.1'
python -m pip install --no-build-isolation -e ./flashinfer

git clone https://github.com/PerkzZheng/prims-ts-examples.git
cd prims-ts-examples
python q_token_kv_block_sparse_attention.py
```

Keep the installed PyTorch/CUDA combination compatible with your driver.
First execution JIT-compiles the selected kernels and can take several minutes.
No vLLM checkout, model weights, or top-k dump is required.

To try a smaller workload in the same environment:

```python
import torch
from q_token_kv_block_sparse_attention import run_packed_prefill, run_fixed_mtp_decode

device = torch.device("cuda")
sm_count = torch.cuda.get_device_properties(device).multi_processor_count
run_packed_prefill(device, sm_count, request_q_lengths=(5, 3))
run_fixed_mtp_decode(device, sm_count, batch_size=1)
```

The helper may choose smaller groups for those workloads. These calls still
preserve the request boundaries and all semantic query tokens.

Validated on GB300 (SM103) with the pinned FlashInfer source: packed prefill
with G5/partial groups and G1, and MTP3 decode with G4 and G1. Sampled outputs
match an FP32 PyTorch reference; decode graph replay overwrites poisoned
outputs correctly. SM100 runtime has not been tested for this example.

## GLM 5.3 sparse MLA

[glm5_sparse_mla.py](glm5_sparse_mla.py) demonstrates the GLM 5.3 NoPE attention
backend boundary with one native latent KV pool. **Prefill and decode call the
same `run_example` function and Prims-TS `plan()` / `run()` operator.** The plan
selects tile, split and kernel-family settings for the workload; the example
has no separate dense-prefill implementation or phase-specific attention code.

| Phase | Query shape | Context | Selected tokens |
| --- | --- | --- | --- |
| Cached-prefix prefill | `[1, 8192, 64, 512]` | 32768 | 512 per query |
| MTP decode | `[4, 4, 64, 512]` | 32768 | 512 per query |

For a serving backend:

- Supply **absorbed** Q with latent dimension 512 and the native BF16/E4M3 latent
  KV cache. Output is BF16 in latent space; apply the learned value/output
  projection afterward. `--heads` is the local head count after tensor parallelism.
- Pass the checkpoint's softmax scale. This NoPE example uses QK dimension 256
  before absorption, hence `256**-0.5`, even though Q and latent KV have width 512.
- Supply expanded logical **token** indices from the GLM pooled indexer, including
  its causal-tail selection. Pooled index IDs cannot be passed as token IDs.
  Indexer pooling does not compress this attention KV cache.
- Map those tokens through the request's block table. The small Triton preparer
  compacts holes, accounts for physical page strides, and emits every prepared
  metadata field needed by automatic dispatch. It accesses no private wrapper
  state. There is no SWA/extra pool (`max_extra_topk=0`).

Synthetic inputs use unique causal indices drawn by random-score top-k, following
FlashMLA's test approach; this does not implement the learned GLM indexer. Both
phases demonstrate CUDA Graph replay with live index lists, padded cache pages,
FP8 descales when selected, and sampled FP64 reference checks. Preparation and
attention are captured together so replay refreshes dependent metadata. This
example reports correctness and tensor shapes, not benchmark timings.

### Setup for sparse MLA

Use the sparse API from [FlashInfer PR #5434](https://github.com/flashinfer-ai/flashinfer/pull/5434).
This is a separate qualified feature checkout from the QToken example above.
The commands below use CUDA 13 and the validated CUTLASS DSL version:

```bash
git clone --recursive --branch feat/prims-ts-sparse-mla \
  https://github.com/PerkzZheng/flashinfer.git flashinfer-sparse-mla
git -C flashinfer-sparse-mla checkout 8ac751ee6a09d8a349a136192dc1ca48d848ac6d
python -m pip install 'setuptools>=77' 'nvidia-cutlass-dsl[cu13]==4.7.0' triton
python -m pip install --no-build-isolation -e ./flashinfer-sparse-mla

# From this examples repository:
python glm5_sparse_mla.py
python glm5_sparse_mla.py --dtype fp8
```

A small invocation also covers short causal prefixes and compacted holes:

```bash
python glm5_sparse_mla.py --prefill-queries 128 --context 128 --topk 64 \
  --heads 16 --decode-batch 2
```

SM100/SM103 are supported by the API. Runtime validation for this example is
recorded on GB300/SM103; SM100 has not been exercised here.

## Development

```bash
python -m pip install pre-commit
pre-commit run --all-files
```

## License and provenance

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The example was moved
from FlashInfer's `examples/prims_ts/q_token_kv_block_sparse_attention.py` at
`d138709e`, then updated to suggest prefill grouping and cache the SM count.
Original copyright and attribution are retained.
