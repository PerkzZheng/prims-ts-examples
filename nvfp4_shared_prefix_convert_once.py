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

"""Convert-once split decode for an NVFP4 KV cache with a shared prefix.

Prefix caching makes many requests point their first pages at the same physical
KV pages. PrimTS decodes NVFP4 K/V by converting every tile to FP8 inside the
attention kernel, so a plain decode call converts the shared pages once per
request. This example converts them once per step instead and splits the call:

1. ``convert_shared_prefix``: one small CUDA kernel (below) converts the shared
   NVFP4 pages to an FP8 scratch copy, ``E4M3(e2m1 * block_scale)``. These are
   exactly the FP8 values the NVFP4 decode kernel computes with.
2. Prefix pass: ``BatchDecodePagedTSWrapper`` on FP8 K/V over the scratch pages,
   dense mask (every query token sees the whole prefix), base-2 LSE.
3. Suffix pass: ``BatchDecodePagedTSWrapper`` on the NVFP4 cache over each
   request's private pages, causal mask, base-2 LSE. It runs on a side stream so
   it fills SMs that the prefix pass leaves idle.
4. ``flashinfer.merge_state_in_place`` combines both passes.

The output equals a single NVFP4 decode call over the full sequence up to
floating-point reordering. The conversion costs a fixed amount per step
(about 35 us for a 30720-token prefix at Hkv 4, D512 on B300), while the saving
grows with the number of requests that share the prefix. With q_len 8 on B300
(CUPTI kernel time; the timing printed below uses CUDA events and reads a few us
higher) it wins from batch 16 (201 vs 225 us single call, trtllm-gen FP8 226 us) and
clearly at batch 32 (319 vs 432 us, trtllm-gen 445 us); at batch 8 the single
call is faster (121 vs 140 us). Dispatch on the number of sharers and the
prefix length, not on batch size alone.

Only the conversion kernel lives here; attention and merging are FlashInfer
public APIs. Requires an SM100/SM103 GPU and FlashInfer with PrimTS D512 FP8-KV
decode support (see README).
"""

from __future__ import annotations

import argparse
import math
import os

import flashinfer
import torch
from flashinfer.attention.prims_ts import BatchDecodePagedTSWrapper
from torch.utils.cpp_extension import load_inline

FP8 = torch.float8_e4m3fn
NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 4, 512, 64

# --------------------------------------------------------------------------
# NVFP4 -> FP8 conversion of the shared prefix pages
# --------------------------------------------------------------------------
_CONVERT_SRC = r"""
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// One E4M3 block scale broadcast to f16x2.
__device__ __forceinline__ uint32_t scale_f16x2(uint32_t sf_byte) {
  uint32_t r;
  uint16_t two = (uint16_t)(sf_byte | (sf_byte << 8));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(two));
  return r;
}

// 4 packed bytes (8 e2m1 values) times one scale -> 8 E4M3 bytes. e2m1 * e4m3
// is exact in f16 (2-bit x 3-bit mantissas), so the result is E4M3(x * s).
__device__ __forceinline__ uint2 convert8(uint32_t in, uint32_t s) {
  uint32_t h0, h1, h2, h3;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3) : "r"(in));
  asm("mul.rn.f16x2 %0, %0, %4;\n mul.rn.f16x2 %1, %1, %4;\n"
      " mul.rn.f16x2 %2, %2, %4;\n mul.rn.f16x2 %3, %3, %4;"
      : "+r"(h0), "+r"(h1), "+r"(h2), "+r"(h3) : "r"(s));
  uint16_t o0, o1, o2, o3;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %4;\n cvt.rn.satfinite.e4m3x2.f16x2 %1, %5;\n"
      " cvt.rn.satfinite.e4m3x2.f16x2 %2, %6;\n cvt.rn.satfinite.e4m3x2.f16x2 %3, %7;"
      : "=h"(o0), "=h"(o1), "=h"(o2), "=h"(o3)
      : "r"(h0), "r"(h1), "r"(h2), "r"(h3));
  return make_uint2((uint32_t)o0 | ((uint32_t)o1 << 16),
                    (uint32_t)o2 | ((uint32_t)o3 << 16));
}

// Inputs use the PrimTS decode NVFP4 layout (HND):
//   data [pages, Hkv, PS, D/2] uint8, scales [pages, Hkv, PS, D/16] E4M3;
//   K scales are row-major, V scales are 4x4-transposed inside each
//   4-token group. Output: [num_prefix_pages, Hkv, PS, D] E4M3.
// One thread converts 16 packed bytes (32 values, two block scales).
template <int D, int PS>
__global__ void __launch_bounds__(256) convert_kernel(
    const uint8_t* __restrict__ k_data, const uint8_t* __restrict__ k_sf,
    const uint8_t* __restrict__ v_data, const uint8_t* __restrict__ v_sf,
    const int32_t* __restrict__ pages, uint8_t* __restrict__ k_out,
    uint8_t* __restrict__ v_out, int num_kv_heads) {
  constexpr int kChunks = D / 32;            // 16-byte chunks per token row
  constexpr int kTokensPerBlock = 256 / kChunks;
  constexpr int kScales = D / 16;
  const bool is_v = blockIdx.y;
  const int slot_head = blockIdx.x / (PS / kTokensPerBlock);
  const int token = (blockIdx.x % (PS / kTokensPerBlock)) * kTokensPerBlock +
                    threadIdx.x / kChunks;
  const int chunk = threadIdx.x % kChunks;
  const int slot = slot_head / num_kv_heads, head = slot_head % num_kv_heads;
  const int64_t page = pages[slot];
  const int64_t src = (page * num_kv_heads + head) * PS + token;
  const int64_t dst = ((int64_t)slot * num_kv_heads + head) * PS + token;
  const uint8_t* data = is_v ? v_data : k_data;
  const uint4 in = __ldg(reinterpret_cast<const uint4*>(data + src * (D / 2)) + chunk);
  uint32_t sa, sb;
  if (!is_v) {
    const uint16_t s2 =
        __ldg(reinterpret_cast<const uint16_t*>(k_sf + src * kScales) + chunk);
    sa = s2 & 0xff;
    sb = s2 >> 8;
  } else {
    const uint8_t* base = v_sf + (page * num_kv_heads + head) * PS * kScales;
    const int group = token >> 2, lane = token & 3;
    auto at = [&](int s) {
      return (uint32_t)__ldg(
          base + (((group * 4 + s / (kScales / 4)) * (kScales / 4) +
                   s % (kScales / 4)) * 4 + lane));
    };
    sa = at(2 * chunk);
    sb = at(2 * chunk + 1);
  }
  const uint32_t fa = scale_f16x2(sa), fb = scale_f16x2(sb);
  const uint2 o0 = convert8(in.x, fa), o1 = convert8(in.y, fa);
  const uint2 o2 = convert8(in.z, fb), o3 = convert8(in.w, fb);
  uint4* out = reinterpret_cast<uint4*>((is_v ? v_out : k_out) + dst * D) + 2 * chunk;
  out[0] = make_uint4(o0.x, o0.y, o1.x, o1.y);
  out[1] = make_uint4(o2.x, o2.y, o3.x, o3.y);
}

void convert(torch::Tensor k_data, torch::Tensor k_sf, torch::Tensor v_data,
             torch::Tensor v_sf, torch::Tensor pages, torch::Tensor k_out,
             torch::Tensor v_out) {
  const int num_pages = pages.numel(), heads = k_data.size(1);
  TORCH_CHECK(k_data.size(2) == 64 && k_data.size(3) == 256,
              "this example instantiates D=512, page size 64");
  constexpr int kTokensPerBlock = 256 / (512 / 32);
  dim3 grid(num_pages * heads * (64 / kTokensPerBlock), 2);
  convert_kernel<512, 64><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      k_data.data_ptr<uint8_t>(), (const uint8_t*)k_sf.data_ptr(),
      v_data.data_ptr<uint8_t>(), (const uint8_t*)v_sf.data_ptr(),
      pages.data_ptr<int32_t>(), (uint8_t*)k_out.data_ptr(),
      (uint8_t*)v_out.data_ptr(), heads);
}
"""

_convert_module = None


def _converter():
    global _convert_module
    if _convert_module is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a"  # e2m1/e4m3 paired converts need sm_100a/sm_103a
        _convert_module = load_inline(
            name=f"prims_ts_example_nvfp4_prefix_convert_sm{arch}",
            cpp_sources=(
                "void convert(torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
            ),
            cuda_sources=_CONVERT_SRC,
            functions=["convert"],
            extra_cuda_cflags=[
                "-O3",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
                "-std=c++20",
            ],
            extra_cflags=["-std=c++20"],
            build_directory=os.environ.get("PRIMS_TS_EXAMPLE_BUILD_DIR") or None,
        )
    return _convert_module


def convert_shared_prefix(nvfp4_kv, kv_scale_factors, prefix_pages, fp8_k, fp8_v):
    """Write FP8 copies of ``prefix_pages`` into ``fp8_k``/``fp8_v`` (graph-safe)."""
    (k_data, v_data), (k_sf, v_sf) = nvfp4_kv, kv_scale_factors
    _converter().convert(k_data, k_sf, v_data, v_sf, prefix_pages, fp8_k, fp8_v)


# --------------------------------------------------------------------------
# The split decode
# --------------------------------------------------------------------------
class SharedPrefixNvfp4Decode:
    """Convert-once split decode for requests that share their first pages.

    Plans once for a fixed batch, query length, shared-prefix length and
    maximum suffix length; ``run`` is allocation-free and CUDA-graph safe.
    """

    def __init__(
        self,
        device,
        *,
        batch_size,
        q_len,
        prefix_len,
        max_suffix_len,
        softmax_scale,
        output_scale,
    ):
        assert prefix_len % PAGE_SIZE == 0, "the shared prefix must be whole pages"
        self.batch_size, self.q_len = batch_size, q_len
        self.num_prefix_pages = prefix_len // PAGE_SIZE
        self.softmax_scale = torch.tensor(
            softmax_scale, dtype=torch.float32, device=device
        )
        self.output_scale = torch.tensor(
            output_scale, dtype=torch.float32, device=device
        )
        scratch_shape = (self.num_prefix_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
        self.fp8_k = torch.empty(scratch_shape, dtype=FP8, device=device)
        self.fp8_v = torch.empty_like(self.fp8_k)
        # Every request reads the same scratch pages in the prefix pass.
        self.prefix_block_table = (
            torch.arange(self.num_prefix_pages, dtype=torch.int32, device=device)
            .expand(batch_size, -1)
            .contiguous()
        )
        self.prefix_lens = torch.full(
            (batch_size,), prefix_len, dtype=torch.int32, device=device
        )
        common = dict(
            max_q_len=q_len,
            q_dtype=FP8,
            out_dtype=torch.bfloat16,
            lse_mode="base2",
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
        )
        nh = (batch_size, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
        self.prefix = BatchDecodePagedTSWrapper(kv_layout="HND")
        self.prefix.plan(
            device, *nh, prefix_len, kv_dtype=FP8, mask_type="dense", **common
        )
        self.suffix = BatchDecodePagedTSWrapper(kv_layout="HND")
        self.suffix.plan(
            device,
            *nh,
            max_suffix_len,
            kv_dtype=torch.uint8,
            mask_type="causal",
            **common,
        )
        # Decode queries are [B, Hq, D] for one token and [B, Q, Hq, D] otherwise.
        shape = (
            (batch_size, NUM_QO_HEADS)
            if q_len == 1
            else (batch_size, q_len, NUM_QO_HEADS)
        )
        self.prefix_out = torch.empty(
            *shape, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.prefix_lse = torch.empty(shape, dtype=torch.float32, device=device)
        self.suffix_out = torch.empty_like(self.prefix_out)
        self.suffix_lse = torch.empty_like(self.prefix_lse)
        self.side_stream = torch.cuda.Stream(device)

    def run(
        self,
        q,
        nvfp4_kv,
        kv_scale_factors,
        prefix_pages,
        suffix_block_table,
        suffix_lens,
    ):
        """Return attention over prefix + suffix in ``self.prefix_out``."""
        main = torch.cuda.current_stream()
        # 1. Convert alone: it and the NVFP4 suffix pass are both DRAM-bound.
        convert_shared_prefix(
            nvfp4_kv, kv_scale_factors, prefix_pages, self.fp8_k, self.fp8_v
        )
        self.side_stream.wait_stream(main)
        # 2. Prefix pass (FP8 K/V, dense) on the main stream ...
        self.prefix.run(
            q,
            (self.fp8_k, self.fp8_v),
            self.prefix_lens,
            self.prefix_block_table,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            out=self.prefix_out,
            lse=self.prefix_lse,
            validate=False,
        )
        # 3. ... while the suffix pass (NVFP4 K/V, causal) fills the idle SMs.
        with torch.cuda.stream(self.side_stream):
            self.suffix.run(
                q,
                nvfp4_kv,
                suffix_lens,
                suffix_block_table,
                kv_scale_factors=kv_scale_factors,
                softmax_scale=self.softmax_scale,
                output_scale=self.output_scale,
                out=self.suffix_out,
                lse=self.suffix_lse,
                validate=False,
            )
        main.wait_stream(self.side_stream)
        # 4. Merge (both LSEs are base 2).
        n = self.batch_size * self.q_len
        flashinfer.merge_state_in_place(
            self.prefix_out.view(n, NUM_QO_HEADS, HEAD_DIM),
            self.prefix_lse.view(n, NUM_QO_HEADS),
            self.suffix_out.view(n, NUM_QO_HEADS, HEAD_DIM),
            self.suffix_lse.view(n, NUM_QO_HEADS),
        )
        return self.prefix_out


# --------------------------------------------------------------------------
# Synthetic NVFP4 cache with a shared prefix
# --------------------------------------------------------------------------
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _quantize_nvfp4(x):
    """Return packed e2m1 [..., D/2] and E4M3 block scales [..., D/16]."""
    blocks = x.float().unflatten(-1, (-1, 16))
    scales = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0).to(FP8)
    r = (blocks / scales.float().clamp(min=1e-12)[..., None]).clamp(-6.0, 6.0)
    bounds = torch.tensor(_E2M1_BOUNDS, device=x.device)
    idx = torch.bucketize(r.abs(), bounds).to(torch.uint8)
    idx = (idx | (((r < 0) & (idx > 0)).to(torch.uint8) << 3)).flatten(-2)
    return idx[..., 0::2] | (idx[..., 1::2] << 4), scales


def _swizzle_v_scales(sf):
    """PrimTS V scale layout: 4x4 (token, scale) transpose per 4-token group."""
    p, h, t, s = sf.shape
    return (
        sf.reshape(p, h, t // 4, 4, 4, s // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape(p, h, t, s)
    )


def make_shared_prefix_inputs(
    batch_size, q_len, prefix_len, suffix_len, device, seed=0
):
    """NVFP4 cache where every request's first ``prefix_len`` tokens share pages."""
    g = torch.Generator(device=device).manual_seed(seed)
    n_prefix, n_suffix = prefix_len // PAGE_SIZE, -(-suffix_len // PAGE_SIZE)
    num_pages = n_prefix + batch_size * n_suffix
    perm = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed))
    prefix_pages = perm[:n_prefix].to(torch.int32).to(device)
    suffix_table = perm[n_prefix:].view(batch_size, n_suffix).to(torch.int32).to(device)
    k_global, v_global = 448.0 / 6.0 / 2.0, 448.0 / 6.0  # NVFP4 per-tensor scales
    shape = (num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
    k = torch.randn(shape, generator=g, device=device) * k_global
    v = torch.randn(shape, generator=g, device=device) * v_global
    (k_data, k_sf), (v_data, v_sf) = _quantize_nvfp4(k), _quantize_nvfp4(v)
    del k, v
    q_shape = (batch_size, NUM_QO_HEADS, HEAD_DIM)
    if q_len > 1:
        q_shape = (batch_size, q_len, NUM_QO_HEADS, HEAD_DIM)
    q = 1.5 * torch.randn(q_shape, generator=g, device=device)
    return dict(
        q=q.clamp(-448, 448).to(FP8),
        nvfp4_kv=(k_data, v_data),
        kv_scale_factors=(k_sf, _swizzle_v_scales(v_sf).contiguous()),
        row_major_v_scales=v_sf,
        prefix_pages=prefix_pages,
        suffix_block_table=suffix_table,
        suffix_lens=torch.full(
            (batch_size,), suffix_len, dtype=torch.int32, device=device
        ),
        full_block_table=torch.cat(
            (prefix_pages.expand(batch_size, -1), suffix_table), dim=1
        ).contiguous(),
        full_lens=torch.full(
            (batch_size,), prefix_len + suffix_len, dtype=torch.int32, device=device
        ),
        softmax_scale=1.0 / k_global / math.sqrt(HEAD_DIM),
        output_scale=1.0 / v_global,
    )


# --------------------------------------------------------------------------
# Check against an FP32 reference and a single NVFP4 decode call; time both
# --------------------------------------------------------------------------
_E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _dequantize_e4m3(packed, scales):
    """The FP8 operand the kernels compute with: E4M3(e2m1 * block scale)."""
    nibbles = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    e2m1 = torch.tensor(_E2M1_VALUES, device=packed.device)
    return (e2m1[nibbles] * scales.float().repeat_interleave(16, -1)).to(FP8)


def _reference_request(inp, batch_idx, q_len, length):
    """FP32 attention of one request over its full (prefix + suffix) context."""
    pages = inp["full_block_table"][batch_idx, : -(-length // PAGE_SIZE)].long()
    (k_data, v_data), k_sf = inp["nvfp4_kv"], inp["kv_scale_factors"][0]
    v_sf = inp["row_major_v_scales"]

    def gather(data, scales):
        x = _dequantize_e4m3(data[pages], scales[pages]).float()
        return x.transpose(1, 2).reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:length]

    k, v = gather(k_data, k_sf), gather(v_data, v_sf)  # [length, Hkv, D]
    group = NUM_QO_HEADS // NUM_KV_HEADS
    q = inp["q"][batch_idx].float().reshape(q_len, NUM_KV_HEADS, group, HEAD_DIM)
    scores = torch.einsum("qhgd,khd->hgqk", q, k) * inp["softmax_scale"]
    ends = torch.arange(q_len, device=q.device) + length - q_len  # causal, bottom-right
    visible = torch.arange(length, device=q.device)[None] <= ends[:, None]
    scores = scores.masked_fill(~visible, -torch.inf)
    out = torch.einsum("hgqk,khd->qhgd", scores.softmax(-1), v) * inp["output_scale"]
    out = out.reshape(q_len, NUM_QO_HEADS, HEAD_DIM)
    return out if q_len > 1 else out[0]


def _graph_time_us(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    flush = torch.empty(
        2 * torch.cuda.get_device_properties(0).L2_cache_size,
        dtype=torch.int8,
        device="cuda",
    )
    times = []
    for _ in range(iters):
        flush.zero_()  # cold L2, as in a real decode step
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--q-len", type=int, default=8)
    ap.add_argument("--prefix-len", type=int, default=30720)
    ap.add_argument("--suffix-len", type=int, default=2048)
    args = ap.parse_args()
    device = torch.device("cuda", torch.cuda.current_device())
    inp = make_shared_prefix_inputs(
        args.batch, args.q_len, args.prefix_len, args.suffix_len, device
    )

    split = SharedPrefixNvfp4Decode(
        device,
        batch_size=args.batch,
        q_len=args.q_len,
        prefix_len=args.prefix_len,
        max_suffix_len=args.suffix_len,
        softmax_scale=inp["softmax_scale"],
        output_scale=inp["output_scale"],
    )

    def run_split():
        return split.run(
            inp["q"],
            inp["nvfp4_kv"],
            inp["kv_scale_factors"],
            inp["prefix_pages"],
            inp["suffix_block_table"],
            inp["suffix_lens"],
        )

    # Reference: one NVFP4 decode call over prefix + suffix.
    full_len = args.prefix_len + args.suffix_len
    single = BatchDecodePagedTSWrapper(kv_layout="HND")
    single.plan(
        device,
        args.batch,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        full_len,
        max_q_len=args.q_len,
        q_dtype=FP8,
        kv_dtype=torch.uint8,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        softmax_scale=split.softmax_scale,
        output_scale=split.output_scale,
    )
    single_out = torch.empty_like(split.prefix_out)

    def run_single():
        return single.run(
            inp["q"],
            inp["nvfp4_kv"],
            inp["full_lens"],
            inp["full_block_table"],
            kv_scale_factors=inp["kv_scale_factors"],
            softmax_scale=split.softmax_scale,
            output_scale=split.output_scale,
            out=single_out,
            validate=False,
        )

    # The conversion reproduces the FP8 values the NVFP4 kernel computes with.
    convert_shared_prefix(
        inp["nvfp4_kv"],
        inp["kv_scale_factors"],
        inp["prefix_pages"],
        split.fp8_k,
        split.fp8_v,
    )
    page = int(inp["prefix_pages"][0])
    expected_k = _dequantize_e4m3(
        inp["nvfp4_kv"][0][page], inp["kv_scale_factors"][0][page]
    )
    assert torch.equal(split.fp8_k[0].view(torch.uint8), expected_k.view(torch.uint8))

    out = run_split().float()
    ref = run_single().float()
    torch.cuda.synchronize()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0).item()

    print(
        f"split vs single call: cos {cos(out, ref):.6f}, "
        f"max |diff| {(out - ref).abs().max().item():.4g}"
    )
    # Both paths round to BF16 and use FP8 P; the split also rounds each pass's
    # output to BF16 before merging. Require the split to be as close to an FP32
    # reference as the single call is, up to one more BF16 rounding.
    for b in sorted({0, args.batch // 2, args.batch - 1}):
        exact = _reference_request(inp, b, args.q_len, full_len)
        err_split = (out[b] - exact).abs().max().item()
        err_single = (ref[b] - exact).abs().max().item()
        print(
            f"  request {b:2d} vs FP32: split cos {cos(out[b], exact):.6f} "
            f"max {err_split:.4g} | single cos {cos(ref[b], exact):.6f} "
            f"max {err_single:.4g}"
        )
        assert err_split <= 2 * err_single + 4e-3, (err_split, err_single)

    t_split, t_single = _graph_time_us(run_split), _graph_time_us(run_single)
    print(
        f"B{args.batch} q_len {args.q_len}, {args.prefix_len}-token shared prefix + "
        f"{args.suffix_len} private tokens per request:\n"
        f"  single NVFP4 call   {t_single:8.1f} us\n"
        f"  convert-once split  {t_split:8.1f} us  ({t_single / t_split:.2f}x)"
    )


if __name__ == "__main__":
    main()
