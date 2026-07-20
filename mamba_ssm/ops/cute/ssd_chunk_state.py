# Copyright (c) 2024, Tri Dao, Albert Gu.

from __future__ import annotations

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_NUM_WARPS = 8
_NUM_THREADS = _NUM_WARPS * 32


@cute.jit
def _softplus(value: Float32) -> Float32:
    if value <= Float32(20.0):
        exponent = cute.math.exp2(value * Float32(_LOG2E), fastmath=True)
        value = cute.math.log2(exponent + Float32(1.0), fastmath=True) * Float32(_LN2)
    return value


@cache
def _compile_chunk_cumsum_fwd(
    dt_dtype,
    a_dtype,
    bias_dtype,
    chunk_size: int,
    *,
    has_bias: bool,
    dt_softplus: bool,
):
    items_per_lane = math.ceil(chunk_size / 32)

    class ChunkCumsumFwd:
        @cute.jit
        def __call__(
            self,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mBias: cute.Tensor | None,
            mDtOut: cute.Tensor,
            mDACumsum: cute.Tensor,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            stride_dt_batch: Int64,
            stride_dt_seqlen: Int64,
            stride_dt_head: Int64,
            stride_a_head: Int64,
            stride_bias_head: Int64,
            dt_min: Float32,
            dt_max: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDt,
                mA,
                mBias,
                mDtOut,
                mDACumsum,
                seqlen,
                nheads,
                nchunks,
                stride_dt_batch,
                stride_dt_seqlen,
                stride_dt_head,
                stride_a_head,
                stride_bias_head,
                dt_min,
                dt_max,
            ).launch(
                grid=[batch, nchunks, cute.ceil_div(nheads, _NUM_WARPS)],
                block=[_NUM_THREADS, 1, 1],
                smem=_NUM_WARPS * chunk_size * 4,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mBias: cute.Tensor | None,
            mDtOut: cute.Tensor,
            mDACumsum: cute.Tensor,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            stride_dt_batch: Int64,
            stride_dt_seqlen: Int64,
            stride_dt_head: Int64,
            stride_a_head: Int64,
            stride_bias_head: Int64,
            dt_min: Float32,
            dt_max: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            batch_idx, chunk, head_block = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            head = head_block * _NUM_WARPS + warp

            smem = cutlass.utils.SmemAllocator()
            sDt = smem.allocate_tensor(
                Float32,
                cute.make_ordered_layout((_NUM_WARPS, chunk_size), order=(1, 0)),
                byte_alignment=16,
            )
            for item in cutlass.range_constexpr(items_per_lane):
                linear = tidx + item * _NUM_THREADS
                pos = linear // _NUM_WARPS
                local_head = linear - pos * _NUM_WARPS
                load_head = head_block * _NUM_WARPS + local_head
                token = chunk * chunk_size + pos
                dt = Float32(0.0)
                if pos < chunk_size and load_head < nheads and token < seqlen:
                    dt_idx = (
                        Int64(batch_idx) * stride_dt_batch
                        + Int64(token) * stride_dt_seqlen
                        + Int64(load_head) * stride_dt_head
                    )
                    dt = Float32(mDt[dt_idx])
                    if const_expr(has_bias):
                        dt += Float32(mBias[Int64(load_head) * stride_bias_head])
                    if const_expr(dt_softplus):
                        dt = _softplus(dt)
                    if dt < dt_min:
                        dt = dt_min
                    if dt > dt_max:
                        dt = dt_max
                if pos < chunk_size:
                    sDt[local_head, pos] = dt
            cute.arch.sync_threads()

            a = Float32(0.0)
            if head < nheads:
                a = Float32(mA[Int64(head) * stride_a_head])

            carry = Float32(0.0)
            for item in cutlass.range_constexpr(items_per_lane):
                pos = item * 32 + lane
                dt = Float32(0.0)
                if head < nheads and pos < chunk_size:
                    dt = sDt[warp, pos]

                value = dt * a
                for offset in (1, 2, 4, 8, 16):
                    addend = cute.arch.shuffle_sync_up(value, offset, mask_and_clamp=0)
                    if lane >= offset:
                        value += addend
                value += carry

                if head < nheads and pos < chunk_size:
                    out_idx = (
                        ((Int64(batch_idx) * nheads + head) * nchunks + chunk)
                        * chunk_size
                        + pos
                    )
                    mDtOut[out_idx] = dt
                    mDACumsum[out_idx] = value
                carry = cute.arch.shuffle_sync(value, 31)

    fake = lambda dtype: (
        cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
        if dtype is not None
        else None
    )
    return cute.compile(
        ChunkCumsumFwd(),
        fake(dt_dtype),
        fake(a_dtype),
        fake(bias_dtype) if has_bias else None,
        fake(Float32),
        fake(Float32),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_cumsum_fwd_cute(
    dt,
    A,
    chunk_size,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    """CuTe implementation of ``_chunk_cumsum_fwd``."""
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    if not 0 < chunk_size <= 256:
        raise ValueError("CuTe chunk cumsum supports chunk sizes from 1 through 256")
    if not all(t.is_cuda for t in (dt, A) if t is not None):
        raise ValueError("CuTe chunk cumsum requires CUDA tensors")
    if any(t is not None and not t.is_contiguous() for t in (dt, A, dt_bias)):
        raise ValueError("CuTe chunk cumsum requires contiguous tensors")

    nchunks = math.ceil(seqlen / chunk_size)
    shape = (batch, nheads, nchunks, chunk_size)
    dt_out = torch.empty(shape, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty_like(dt_out)
    compiled = _compile_chunk_cumsum_fwd(
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[A.dtype],
        torch2cute_dtype_map[dt_bias.dtype] if dt_bias is not None else None,
        chunk_size,
        has_bias=dt_bias is not None,
        dt_softplus=dt_softplus,
    )
    compiled(
        dt.view(-1),
        A.view(-1),
        dt_bias.view(-1) if dt_bias is not None else None,
        dt_out.view(-1),
        dA_cumsum.view(-1),
        batch,
        seqlen,
        nheads,
        nchunks,
        dt.stride(0),
        dt.stride(1),
        dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,
        dt_limit[0],
        dt_limit[1],
    )
    return dA_cumsum, dt_out


__all__ = ["chunk_cumsum_fwd_cute"]
