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

## Setup

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
