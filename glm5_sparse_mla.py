# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, FlashInfer Project.
"""GLM 5.3 NoPE sparse MLA: one attention path for prefill and decode.

The backend supplies absorbed Q[..., 512], a native latent KV cache, and
expanded logical TOKEN indices from the indexer. Indexer pooling is upstream;
it does not change the attention cache into a compressed KV pool. The model's
original QK dimension determines softmax_scale, not the absorbed dimension.

The test-only input generator draws unique causal token IDs using random
scores/topk, as in FlashMLA. It does not implement GLM's learned indexer.
Page mapping/stable compaction follows the vLLM sparse-preparation pattern.
"""

import argparse

import torch
import triton
import triton.language as tl
from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    SparseMLAPreparedMetadata,
)


@triton.jit
def _prepare_indices(
    logical,
    block_table,
    indices,
    lengths,
    routes,
    execution_lengths,
    scale_params,
    q_scale,
    kv_scale,
    softmax_scale,
    QUERIES: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(logical + row * TOPK + col, col < TOPK, -1)
    valid = (col < TOPK) & (token >= 0)
    page = tl.load(
        block_table + (row // QUERIES) * TABLE_STRIDE + token // PAGE,
        valid,
        0,
    )
    # TMA consumes storage-row offsets, including physical page padding.
    storage_row = page * PAGE_STRIDE_ROWS + token % PAGE
    destination = tl.cumsum(valid.to(tl.int32)) - valid.to(tl.int32)
    count = tl.sum(valid.to(tl.int32))
    tl.store(indices + row * TOPK + destination, storage_row, valid)
    tl.store(indices + row * TOPK + col, -1, (col >= count) & (col < TOPK))
    tl.store(routes + row * CAPACITY + destination, storage_row, valid)
    tl.store(
        routes + row * CAPACITY + col,
        0x7FFFFFFF,
        (col >= count) & (col < CAPACITY),
    )
    tl.store(lengths + row, count)
    tl.store(execution_lengths + row, tl.maximum(tl.cdiv(count, 128) * 128, 1))
    tl.store(scale_params + 2 + HEADS + row, count.to(tl.float32))
    if row == 0:
        ks = tl.load(kv_scale)
        tl.store(scale_params, softmax_scale * tl.load(q_scale) * ks)
        tl.store(scale_params + 1, ks)
        tl.store(scale_params + 2 + col, -float("inf"), col < HEADS)


def _make_inputs(batch, queries, heads, context, topk, dtype, device):
    page_size = 16
    pages = triton.cdiv(context, page_size)
    block_table = torch.randperm(batch * pages, device=device, dtype=torch.int32)
    block_table = block_table.view(batch, pages)
    descale = 1 / 16 if dtype == torch.float8_e4m3fn else 1.0
    q_scale = torch.tensor(descale, device=device, dtype=torch.float32)
    kv_scale = torch.tensor(descale, device=device, dtype=torch.float32)
    query = (torch.randn(batch, queries, heads, 512, device=device) * 0.2 / q_scale).to(
        dtype
    )
    storage = (
        torch.randn(batch * pages, page_size + 1, 512, device=device) * 0.2 / kv_scale
    ).to(dtype)
    storage[:, page_size] = torch.nan
    cache = storage[:, :page_size]  # A padded native cache; no KV repacking.
    rows = batch * queries
    logical = torch.empty(rows, topk, device=device, dtype=torch.int32)
    tokens = torch.arange(context, device=device)
    slots = torch.arange(topk, device=device)
    # Chunking bounds fixture memory even for 8K-query/32K-context prefill.
    for start in range(0, rows, 128):
        end = min(start + 128, rows)
        visible = (
            torch.arange(start, end, device=device) % queries + context - queries + 1
        )
        scores = torch.rand(end - start, context, device=device)
        scores.masked_fill_(tokens[None, :] >= visible[:, None], -torch.inf)
        selected = scores.topk(topk, dim=-1, sorted=True).indices.to(torch.int32)
        selected.masked_fill_(slots[None, :] >= visible[:, None], -1)
        logical[start:end] = selected
    return query, cache, logical, block_table, q_scale, kv_scale


def _check_samples(query, cache, logical, table, out, lse, q_scale, kv_scale, scale):
    """Check a few rows with independent FP64 math and native-input descales."""
    rows, heads = logical.shape[0], query.shape[-2]
    q = query.reshape(rows, heads, 512)
    for row in sorted({0, rows // 2, rows - 1}):
        tokens = logical[row].long()
        tokens = tokens[tokens >= 0]
        pages = table[row // query.shape[1], tokens // cache.shape[1]].long()
        values = cache[pages, tokens % cache.shape[1]].double() * kv_scale.double()
        scores = ((q[row].double() * q_scale.double()) @ values.T) * scale
        probabilities = scores.softmax(-1)
        expected = probabilities @ values
        actual = out.reshape(rows, heads, 512)[row].double()
        if query.dtype == torch.float8_e4m3fn:
            # E4M3 P rounding plus BF16 partial/output rounding. Referencing the
            # quantized inputs excludes Q/KV quantization error from this check.
            absolute = values.abs()
            bound = (2**-4 + 3 * 2**-8 + 1e-5) * (probabilities @ absolute)
            bound += (
                (2**-10 / 448) * probabilities.amax(-1, keepdim=True) * absolute.sum(0)
            )
            assert ((actual - expected).abs() <= bound + 1e-6).all()
        else:
            torch.testing.assert_close(actual, expected, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(
            lse.reshape(rows, heads)[row].double(),
            scores.logsumexp(-1),
            atol=2e-4,
            rtol=1e-4,
        )


def run_example(label, *, batch, queries, heads, context, topk, dtype, device):
    """The same plan/prepare/run sequence handles both phases."""
    query, cache, logical, table, q_scale, kv_scale = _make_inputs(
        batch, queries, heads, context, topk, dtype, device
    )
    # GLM 5.3 NoPE uses QK=256 before absorption and latent rank=512.
    # A serving backend should pass its checkpoint's attention scale here.
    softmax_scale = 256**-0.5
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        device,
        batch,
        heads,
        max_seq_len_q=queries,
        max_topk=topk,
        max_extra_topk=0,
        q_data_type=dtype,
        return_lse=True,
        assume_valid_prefix=True,
    )
    rows = batch * queries
    capacity = max(256, triton.cdiv(topk, 128) * 128)
    counts = torch.empty(rows, device=device, dtype=torch.int32)
    metadata = SparseMLAPreparedMetadata(
        indices=torch.empty(rows, topk, device=device, dtype=torch.int32),
        lengths=counts,
        routes=torch.empty(1, rows, capacity, device=device, dtype=torch.int32),
        execution_lengths=torch.empty(1, rows, device=device, dtype=torch.int32),
        valid_counts=counts.unsqueeze(0),
        scale_params=torch.empty(
            1, 2 + heads + rows, device=device, dtype=torch.float32
        ),
    )
    out = torch.empty_like(query, dtype=torch.bfloat16)
    lse = torch.empty(query.shape[:-1], device=device, dtype=torch.float32)

    def prepare():
        # Supplying both direct lists and packed fields supports every
        # automatic schedule; no private wrapper state is used.
        _prepare_indices[(rows,)](
            logical,
            table,
            metadata.indices,
            metadata.lengths,
            metadata.routes,
            metadata.execution_lengths,
            metadata.scale_params,
            q_scale,
            kv_scale,
            softmax_scale,
            queries,
            heads,
            topk,
            capacity,
            table.stride(0),
            cache.shape[1],
            cache.stride(0) // 512,
            triton.next_power_of_2(capacity),
        )

    def attend(validate=False):
        return wrapper.run(
            query,
            cache,
            metadata,
            q_scale=q_scale,
            kv_scale=kv_scale,
            softmax_scale=softmax_scale,
            out=out,
            lse=lse,
            validate=validate,
        )

    prepare()
    attend(validate=True)  # Compile/warm up before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare()
        attend()
    # Replays consume live input lists, including holes that preparation
    # compacts into the valid prefixes promised to attention.
    logical.copy_(logical.flip(-1))
    out.fill_(torch.nan)
    lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    _check_samples(
        query, cache, logical, table, out, lse, q_scale, kv_scale, softmax_scale
    )
    print(
        f"{label}: Q={tuple(query.shape)}, O={tuple(out.shape)}, "
        f"{str(dtype).removeprefix('torch.')}, sampled reference + graph replay passed"
    )
    # O is still in latent space. The backend applies its learned value/output
    # projection after this call; no separate dense-prefill attention is needed.
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--prefill-queries", type=int, default=8192)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--decode-queries", type=int, default=4)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--heads", type=int, default=64, help="Local query heads after TP"
    )
    args = parser.parse_args()
    if (
        min(args.prefill_queries, args.decode_batch, args.decode_queries, args.topk) < 1
        or max(args.prefill_queries, args.decode_queries, args.topk) > args.context
        or not 1 <= args.heads <= 128
    ):
        parser.error("Require positive sizes, Q/topk <= context, and 1 <= heads <= 128")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        raise RuntimeError("This example requires SM100/SM103 and CUTLASS DSL 4.7+")
    torch.manual_seed(2026)
    shared = dict(
        heads=args.heads,
        context=args.context,
        topk=args.topk,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn,
        device=torch.device("cuda"),
    )
    run_example("prefill", batch=1, queries=args.prefill_queries, **shared)
    run_example(
        "decode", batch=args.decode_batch, queries=args.decode_queries, **shared
    )


if __name__ == "__main__":
    main()
