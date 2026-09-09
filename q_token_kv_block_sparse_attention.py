# Copyright (c) 2026, FlashInfer Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted for PerkzZheng/prims-ts-examples: suggested prefill grouping,
# caller-cached SM count, and decode views that preserve the semantic SQ.

"""QToken-KvBlock-Sparse-Attention packed-prefill and MTP-decode examples.

These are the two layouts used by serving-framework integrations. Prefill keeps
queries packed as ``[total_q, Hq, D]`` and supplies request-safe route offsets.
Uniform decode uses ``[B, Nq, G, Hq, D]`` and needs no route-offset tensor.
Both examples suggest G once from the workload and a cached SM count, then fix
it for the prepared plan. Both paths prepare metadata and attention so the
hot-path call accepts only current semantic inputs and a framework-owned output.
The workspace-owned intermediate metadata is the triple
``(q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens)``. Page indices are plain
Int32 cache locators. Grouped routes store four 8-bit query-membership masks per
Int32 word; Q1 has a zero-width membership table. ``seq_lens`` selects each
route's live locator prefix. Spare bytes in the last live membership word
are zero; unused locator entries and whole membership words beyond the live
prefix are unspecified. The combined calls below keep this metadata
hidden and do not add it to the public attention signature.

For each valid query at zero-based position ``p``, the indexer supplies
``min(block_topk, (p + 1) // kv_block_size)`` distinct completed-block IDs
as a valid prefix, in any order. Q1 does not compact arbitrary missing entries
inside that prefix. Later columns are ignored; metadata derives the incomplete
causal tail directly from the query position.

The caller fixes ``G`` for a prepared plan. The optional pure-host group-size
suggestion helper uses a caller-cached SM count and never queries the device or
reads a tensor. QToken-KvBlock-Sparse-Attention normally chooses the smallest qualified TileQ in
8/16/32/64 that can hold ``G * (Hq / Hkv)`` rows and always uses a 128-token
K/V tile. FP8 Q1 uses its qualified Q64/Keeps profile. Packed prefill is
nonsplit; fixed decode fills, but does not cross, the first active-CTA service
wave while retaining useful K/V work per split. Q1 is capped at split eight;
grouped routes use the shared reducer's supported fanout. The current
production route is causal and non-windowed.
``kv_block_size`` is an explicit power-of-two API parameter
so integrations do not bake the current specialization into their interface,
although only block size four is implemented today.
``max_seq_len_kv`` is the plan-time upper bound on the logical context length,
including the current query tokens. It validates logical block IDs and may be
smaller than the reserved dense block-table capacity. A CUDA graph may replay
only inputs whose visible K/V lengths stay within this fixed bound.

Run on SM100 or SM103 after installing FlashInfer with PrimTS support. To keep
the example compact, requests share the same physical cache pages; a serving
framework normally provides distinct physical mappings.
"""

from __future__ import annotations

from itertools import accumulate

import torch
from flashinfer.decode import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
    make_q_token_kv_block_sparse_qo_indptr,
    suggest_q_token_kv_block_sparse_group_size,
)

_NUM_QO_HEADS = 12
_NUM_KV_HEADS = 1
_HEAD_DIM = 256
_BLOCK_TOPK = 512
_KV_BLOCK_SIZE = 4
_STORAGE_PAGE_SIZE = 16
_CONTEXT_LENGTH = 8192


def _make_cache_and_block_table(
    num_requests: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_storage_pages = (_CONTEXT_LENGTH + _STORAGE_PAGE_SIZE - 1) // (
        _STORAGE_PAGE_SIZE
    )
    k_cache = torch.randn(
        num_storage_pages,
        _NUM_KV_HEADS,
        _STORAGE_PAGE_SIZE,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        num_storage_pages,
        dtype=torch.int32,
        device=device,
    ).repeat(num_requests, 1)
    return k_cache, v_cache, block_table


def _make_indexer_block_ids(
    num_query_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    base_blocks = torch.arange(_BLOCK_TOPK, dtype=torch.int32, device=device)
    return torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(num_query_tokens)]
    ).contiguous()


def run_packed_prefill(
    device: torch.device,
    multi_processor_count: int,
    *,
    request_q_lengths: tuple[int, ...] = (1025, 513),
) -> None:
    """Run variable-length chunked prefill with a suggested, then fixed, G.

    Each chunk ends at context position 8191 and attends to an existing cache.
    The default has enough query routes to illustrate G5 on SM100/SM103 and
    includes a partial final group. This is an API example, not a benchmark.
    """

    num_requests = len(request_q_lengths)
    num_query_tokens = sum(request_q_lengths)
    query = torch.randn(
        num_query_tokens,
        _NUM_QO_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache, v_cache, block_table = _make_cache_and_block_table(num_requests, device)
    indexer_block_ids = _make_indexer_block_ids(num_query_tokens, device)
    token_to_request = torch.tensor(
        [
            request
            for request, length in enumerate(request_q_lengths)
            for _ in range(length)
        ],
        dtype=torch.int32,
        device=device,
    )
    query_positions = torch.tensor(
        [
            position
            for length in request_q_lengths
            for position in range(_CONTEXT_LENGTH - length, _CONTEXT_LENGTH)
        ],
        dtype=torch.int64,
        device=device,
    )
    query_start_loc_cpu = torch.tensor(
        (0, *accumulate(request_q_lengths)),
        dtype=torch.int32,
        device="cpu",
    )

    group_size = suggest_q_token_kv_block_sparse_group_size(
        batch_size=num_requests,
        seq_len_q=max(request_q_lengths),
        selected_seq_len_kv=_BLOCK_TOPK * _KV_BLOCK_SIZE + (_KV_BLOCK_SIZE - 1),
        num_qo_heads=_NUM_QO_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        multi_processor_count=multi_processor_count,
    )
    # Maximum request length is only a scheduling hint. Build actual routes
    # from each request's boundaries so partial groups never cross requests.
    qo_indptr = make_q_token_kv_block_sparse_qo_indptr(
        query_start_loc_cpu,
        num_query_tokens,
        group_size=group_size,
        device=device,
    )
    output = torch.empty_like(query)
    workspace_bytes = get_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=_BLOCK_TOPK,
        max_seq_len_kv=_CONTEXT_LENGTH,
        kv_block_size=_KV_BLOCK_SIZE,
        o_data_type=output.dtype,
        qo_indptr=qo_indptr,
        seq_len_q=group_size,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        qo_indptr.numel() - 1,
        group_size,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _KV_BLOCK_SIZE,
        _STORAGE_PAGE_SIZE,
        _BLOCK_TOPK,
        _CONTEXT_LENGTH,
        device=device,
        workspace_buffer=workspace,
        use_packed_q=True,
        q_data_type=query.dtype,
        kv_data_type=k_cache.dtype,
        o_data_type=output.dtype,
    )
    wrapper.run(
        query,
        (k_cache, v_cache),
        block_table,
        indexer_block_ids,
        token_to_request,
        query_positions,
        qo_indptr=qo_indptr,
        out=output,
    )
    torch.cuda.synchronize()
    print(
        "packed prefill: "
        f"query={tuple(query.shape)}, routes={qo_indptr.numel() - 1}, "
        f"group_size<={group_size}, workspace_bytes={workspace_bytes}"
    )


def run_fixed_mtp_decode(
    device: torch.device,
    multi_processor_count: int,
    *,
    batch_size: int = 8,
) -> None:
    """Run causal MTP decode with one suggested, then caller-fixed, group."""

    mtp_num_speculative_tokens = 3
    seq_len_q = mtp_num_speculative_tokens + 1
    group_size = suggest_q_token_kv_block_sparse_group_size(
        batch_size,
        seq_len_q,
        _BLOCK_TOPK * _KV_BLOCK_SIZE + (_KV_BLOCK_SIZE - 1),
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        multi_processor_count,
    )
    # SQ4 permits each suggested G (1, 2, or 4) without padding. Preserve all
    # four semantic tokens even when the suggestion uses several small groups.
    num_query_groups = seq_len_q // group_size
    num_query_tokens = batch_size * num_query_groups * group_size

    # vLLM owns flat token storage and exposes this zero-copy fixed decode view.
    flat_query = torch.randn(
        num_query_tokens,
        _NUM_QO_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query = flat_query.view(
        batch_size,
        num_query_groups,
        group_size,
        _NUM_QO_HEADS,
        _HEAD_DIM,
    )
    flat_output = torch.empty_like(flat_query)
    output = flat_output.view_as(query)
    k_cache, v_cache, block_table = _make_cache_and_block_table(batch_size, device)
    indexer_block_ids = _make_indexer_block_ids(num_query_tokens, device)
    token_to_request = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(num_query_groups * group_size)
    query_positions = torch.arange(
        _CONTEXT_LENGTH - seq_len_q,
        _CONTEXT_LENGTH,
        dtype=torch.int64,
        device=device,
    ).repeat(batch_size)

    workspace_bytes = get_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=_BLOCK_TOPK,
        max_seq_len_kv=_CONTEXT_LENGTH,
        kv_block_size=_KV_BLOCK_SIZE,
        o_data_type=output.dtype,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        batch_size * num_query_groups,
        group_size,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _KV_BLOCK_SIZE,
        _STORAGE_PAGE_SIZE,
        _BLOCK_TOPK,
        _CONTEXT_LENGTH,
        device=device,
        workspace_buffer=workspace,
        q_data_type=query.dtype,
        kv_data_type=k_cache.dtype,
        o_data_type=output.dtype,
    )

    # Compile and initialize outside capture, then replay only the hot path.
    wrapper.run(
        query,
        (k_cache, v_cache),
        block_table,
        indexer_block_ids,
        token_to_request,
        query_positions,
        out=output,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            query,
            (k_cache, v_cache),
            block_table,
            indexer_block_ids,
            token_to_request,
            query_positions,
            out=output,
        )
    graph.replay()
    torch.cuda.synchronize()

    print(
        "fixed MTP decode: "
        f"query={tuple(query.shape)}, group_size={group_size}, "
        f"kv_block_size={_KV_BLOCK_SIZE}, "
        f"workspace_bytes={workspace_bytes}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "PrimTS QToken-KvBlock-Sparse-Attention currently requires SM100 or SM103"
        )

    torch.manual_seed(42)
    device = torch.device("cuda")
    multi_processor_count = torch.cuda.get_device_properties(
        device
    ).multi_processor_count
    run_packed_prefill(device, multi_processor_count)
    run_fixed_mtp_decode(device, multi_processor_count)


if __name__ == "__main__":
    main()
