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

"""BF16 D256 per-token sparse attention with shared K=V and CUDA Graphs.

Q has 256 heads; the contiguous HND cache has two independent KV heads.
Each query selects logical token IDs, shared across the two KV heads.
The application tensors use [B, seq_len_q, Hq, D]: seq_len_q=16384 for
prefill and seq_len_q=4 for MTP decode. A zero-copy view adds the wrapper's
G=1 sparse-group axis. All query rows are submitted in one wrapper call.

Plan and warm up outside capture. Keep all captured storage alive, and update
its contents in place before replay. One wrapper/workspace serves one ordered
stream; concurrent graphs need separate wrappers and workspaces.
"""

from __future__ import annotations

import argparse

import torch
from flashinfer.decode import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
)


def make_token_indices(
    positions: torch.Tensor, topk: int, context: int
) -> torch.Tensor:
    """Sample unique causal tokens; pad only the end of each row with -1.

    Random-score top-k follows the sampling approach in FlashMLA's unit tests.
    Chunking bounds temporary storage for the full 16K-query prefill example.
    A serving application supplies its indexer's token IDs instead.
    """
    result = torch.full(
        (positions.numel(), topk), -1, dtype=torch.int32, device=positions.device
    )
    tokens = torch.arange(context, device=positions.device)
    width = min(topk, context)
    for start in range(0, positions.numel(), 128):
        visible = positions[start : start + 128, None]
        scores = torch.rand(visible.shape[0], context, device=positions.device)
        scores.masked_fill_(tokens[None, :] > visible, -torch.inf)
        values, ids = scores.topk(width, dim=-1)
        result[start : start + 128, :width] = ids.masked_fill(
            ~values.isfinite(), -1
        ).to(torch.int32)
    return result


def check_output(q, kv, table, indices, requests, positions, output) -> None:
    """Check representative queries against FP32 attention over selected tokens."""
    flat_q = q.flatten(0, 1)
    flat_out = output.flatten(0, 1)
    rows = sorted({0, 1, 63, 127, flat_q.shape[0] // 2, flat_q.shape[0] - 1})
    for row in (r for r in rows if r < flat_q.shape[0]):
        ids = indices[row].long()
        ids = ids[(ids >= 0) & (ids <= positions[row])]
        pages = table[requests[row].long(), ids // kv.shape[2]].long()
        selected = kv[pages, :, ids % kv.shape[2], :].permute(1, 0, 2).float()
        query = flat_q[row].view(2, 128, 256).float()
        scores = torch.matmul(query, selected.transpose(-1, -2)) * 256**-0.5
        reference = torch.matmul(scores.softmax(-1), selected).reshape(256, 256)
        torch.testing.assert_close(
            flat_out[row].float(), reference, rtol=0.02, atol=0.02
        )
    assert torch.isfinite(output).all(), "Output contains nonfinite values"


@torch.inference_mode()
def run_example(*, batch_size: int, seq_len_q: int, context: int, topk: int) -> None:
    device = torch.device("cuda")
    page_size, heads_q, heads_kv, head_dim = 64, 256, 2, 256
    pages_per_request = (context + page_size - 1) // page_size
    # Each request owns separate physical pages. Shuffle the mapping to show
    # that sparse IDs are logical token positions, not physical cache offsets.
    table = torch.randperm(
        batch_size * pages_per_request, device=device, dtype=torch.int32
    ).view(batch_size, pages_per_request)
    kv = torch.randn(
        batch_size * pages_per_request,
        heads_kv,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = torch.randn(
        batch_size, seq_len_q, heads_q, head_dim, device=device, dtype=torch.bfloat16
    )
    requests = torch.arange(
        batch_size, device=device, dtype=torch.int32
    ).repeat_interleave(seq_len_q)
    positions = torch.arange(
        context - seq_len_q, context, device=device, dtype=torch.int32
    ).repeat(batch_size)
    indices = make_token_indices(positions, topk, context)
    output = torch.empty_like(q)
    # This wrapper takes [B, num_groups, G, Hq, D]. Each query has its own
    # token list, so the token-sparse kernel uses G=1 and num_groups=seq_len_q.
    # unsqueeze is a view: it neither copies nor drops any query/output rows.
    query_group_size = 1
    q_groups = q.unsqueeze(2)
    output_groups = output.unsqueeze(2)
    num_query_routes = batch_size * seq_len_q
    # One KV partition matches the measured configurations. The same flag must
    # be used for workspace sizing and planning.
    workspace_bytes = get_q_token_kv_block_sparse_workspace_size(
        q_groups,
        kv,
        table,
        block_topk=topk,
        max_seq_len_kv=context,
        kv_block_size=1,
        split_kv=False,
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        batch_size=num_query_routes,
        # Upstream main / PR #5351 require this argument (no default).
        # It denotes sparse group size G=1. The application's seq_len_q is
        # 16384 for prefill or 4 for MTP; all B * seq_len_q routes run together.
        seq_len_q=query_group_size,
        num_qo_heads=heads_q,
        num_kv_heads=heads_kv,
        head_dim=head_dim,
        kv_block_size=1,  # Each index selects one token, not a four-token block.
        page_size=page_size,
        block_topk=topk,
        max_seq_len_kv=context,
        device=device,
        workspace_buffer=workspace,
        split_kv=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )

    def run():
        wrapper.run(
            q_groups,
            (kv, kv),  # Identical storage enables the shared-KV load schedule.
            table,
            indices,
            requests,
            positions,
            sm_scale=head_dim**-0.5,
            out=output_groups,
        )
        return output  # Application layout: [B, seq_len_q, Hq, D].

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    check_output(q, kv, table, indices, requests, positions, output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(torch.nan)
    for _ in range(10):
        graph.replay()
    check_output(q, kv, table, indices, requests, positions, output)

    # Refresh inputs without changing captured addresses or planned capacities.
    q.normal_()
    indices.copy_(make_token_indices(positions, topk, context))
    output.fill_(torch.nan)
    for _ in range(10):
        graph.replay()
    check_output(q, kv, table, indices, requests, positions, output)
    print(
        f"PASS B={batch_size} seq_len_q={seq_len_q} KV={context} topk={topk}: "
        "BF16 Hq=256 Hkv=2 D=256; eager, graph replay, and updated indices"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prefill", "decode", "both"), default="both"
    )
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--prefill-queries", type=int, default=16384)
    parser.add_argument("--decode-batch", type=int, default=512)
    parser.add_argument("--decode-queries", type=int, default=4)
    parser.add_argument("--topk", type=int, choices=(512, 2048), default=512)
    args = parser.parse_args()
    if (
        min(args.context, args.prefill_queries, args.decode_batch, args.decode_queries)
        <= 0
    ):
        parser.error("context, batch, and query counts must be positive")
    if args.phase in ("prefill", "both") and args.prefill_queries > args.context:
        parser.error("prefill queries must not exceed context")
    if args.phase in ("decode", "both") and args.decode_queries > args.context:
        parser.error("decode queries must not exceed context")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        raise RuntimeError("This example requires an SM100, SM103, or SM107 CUDA GPU")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.phase in ("prefill", "both"):
        run_example(
            batch_size=1,
            seq_len_q=args.prefill_queries,
            context=args.context,
            topk=args.topk,
        )
    if args.phase in ("decode", "both"):
        run_example(
            batch_size=args.decode_batch,
            seq_len_q=args.decode_queries,
            context=args.context,
            topk=args.topk,
        )


if __name__ == "__main__":
    main()
