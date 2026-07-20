# Copyright (c) 2024, Tri Dao, Albert Gu.

from __future__ import annotations

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import sigmoid
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce


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
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size + pos
                    mDtOut[out_idx] = dt
                    mDACumsum[out_idx] = value
                carry = cute.arch.shuffle_sync(value, 31)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
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


@cache
def _compile_chunk_cumsum_bwd(
    ddA_dtype,
    ddt_out_dtype,
    dt_dtype,
    a_dtype,
    bias_dtype,
    chunk_size: int,
    *,
    has_bias: bool,
    dt_softplus: bool,
):
    items_per_lane = math.ceil(chunk_size / 32)

    class ChunkCumsumBwd:
        @cute.jit
        def __call__(
            self,
            mDDA: cute.Tensor,
            mDDtOut: cute.Tensor,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mBias: cute.Tensor | None,
            mDDt: cute.Tensor,
            mDAPartial: cute.Tensor,
            mDBiasPartial: cute.Tensor | None,
            mDA: cute.Tensor,
            mDBias: cute.Tensor | None,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            dt_min: Float32,
            dt_max: Float32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDDA,
                mDDtOut,
                mDt,
                mA,
                mBias,
                mDDt,
                mDAPartial,
                mDBiasPartial,
                seqlen,
                nheads,
                nchunks,
                dt_min,
                dt_max,
            ).launch(
                grid=[batch, nchunks, cute.ceil_div(nheads, _NUM_WARPS)],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )
            self.finalize_kernel(
                mDAPartial, mDBiasPartial, mDA, mDBias, batch * nchunks, nheads
            ).launch(
                grid=[cute.ceil_div(nheads, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDDA: cute.Tensor,
            mDDtOut: cute.Tensor,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mBias: cute.Tensor | None,
            mDDt: cute.Tensor,
            mDAPartial: cute.Tensor,
            mDBiasPartial: cute.Tensor | None,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            dt_min: Float32,
            dt_max: Float32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            batch_idx, chunk, head_block = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            head = head_block * _NUM_WARPS + warp
            dA_sum = Float32(0.0)
            dbias_sum = Float32(0.0)
            if head < nheads:
                a = Float32(mA[head])
                bias = Float32(mBias[head]) if const_expr(has_bias) else Float32(0.0)
                for item in cutlass.range_constexpr(items_per_lane):
                    pos = item * 32 + lane
                    token = chunk * chunk_size + pos
                    if pos < chunk_size and token < seqlen:
                        chunk_idx = (
                            (Int64(batch_idx) * nheads + head) * nchunks + chunk
                        ) * chunk_size + pos
                        dt_idx = (Int64(batch_idx) * seqlen + token) * nheads + head
                        ddA = Float32(mDDA[chunk_idx])
                        ddt = ddA * a + Float32(mDDtOut[chunk_idx])
                        dt = Float32(mDt[dt_idx]) + bias
                        dt_pre_softplus = dt
                        if const_expr(dt_softplus):
                            dt = _softplus(dt)
                        if dt < dt_min:
                            dt = dt_min
                            ddt = Float32(0.0)
                        if dt > dt_max:
                            dt = dt_max
                            ddt = Float32(0.0)
                        if const_expr(dt_softplus):
                            if dt_pre_softplus <= Float32(20.0):
                                ddt *= sigmoid(dt_pre_softplus)
                        mDDt[dt_idx] = dt_dtype(ddt)
                        dA_sum += ddA * dt
                        dbias_sum += ddt

            for offset in (16, 8, 4, 2, 1):
                dA_sum += cute.arch.shuffle_sync_down(dA_sum, offset)
                dbias_sum += cute.arch.shuffle_sync_down(dbias_sum, offset)
            if lane == 0 and head < nheads:
                partial_idx = (Int64(batch_idx) * nchunks + chunk) * nheads + head
                mDAPartial[partial_idx] = dA_sum
                if const_expr(has_bias):
                    mDBiasPartial[partial_idx] = dbias_sum

        @cute.kernel
        def finalize_kernel(
            self,
            mDAPartial: cute.Tensor,
            mDBiasPartial: cute.Tensor | None,
            mDA: cute.Tensor,
            mDBias: cute.Tensor | None,
            outer: Int32,
            nheads: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            head = block * _NUM_THREADS + tidx
            if head < nheads:
                dA = Float32(0.0)
                dbias = Float32(0.0)
                for idx in cutlass.range(0, outer, 1, unroll=1):
                    partial_idx = Int64(idx) * nheads + head
                    dA += Float32(mDAPartial[partial_idx])
                    if const_expr(has_bias):
                        dbias += Float32(mDBiasPartial[partial_idx])
                mDA[head] = dA
                if const_expr(has_bias):
                    mDBias[head] = dbias

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkCumsumBwd(),
        fake(ddA_dtype),
        fake(ddt_out_dtype),
        fake(dt_dtype),
        fake(a_dtype),
        fake(bias_dtype) if has_bias else None,
        fake(dt_dtype),
        fake(Float32),
        fake(Float32) if has_bias else None,
        fake(Float32),
        fake(Float32) if has_bias else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_cumsum_bwd_cute(
    ddA,
    ddt_out,
    dt,
    A,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    ddt=None,
):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    if ddA.shape != (batch, nheads, nchunks, chunk_size) or ddt_out.shape != ddA.shape:
        raise ValueError("CuTe chunk cumsum gradients have incompatible shapes")
    if A.shape != (nheads,) or (dt_bias is not None and dt_bias.shape != (nheads,)):
        raise ValueError("CuTe chunk cumsum parameters have incompatible shapes")
    tensors = (ddA, ddt_out, dt, A, dt_bias, ddt)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_bwd

        return _chunk_cumsum_bwd(
            ddA, ddt_out, dt, A, dt_bias, dt_softplus, dt_limit, ddt
        )
    if ddt is None:
        ddt = torch.empty_like(dt)
    outer = batch * nchunks
    dA_partial = torch.empty(outer, nheads, device=dt.device, dtype=torch.float32)
    dbias_partial = torch.empty_like(dA_partial) if dt_bias is not None else None
    dA = torch.empty_like(A, dtype=torch.float32)
    dbias = (
        torch.empty_like(dt_bias, dtype=torch.float32) if dt_bias is not None else None
    )
    compiled = _compile_chunk_cumsum_bwd(
        torch2cute_dtype_map[ddA.dtype],
        torch2cute_dtype_map[ddt_out.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[A.dtype],
        torch2cute_dtype_map[dt_bias.dtype] if dt_bias is not None else None,
        chunk_size,
        has_bias=dt_bias is not None,
        dt_softplus=dt_softplus,
    )
    compiled(
        ddA.view(-1),
        ddt_out.view(-1),
        dt.view(-1),
        A.view(-1),
        dt_bias.view(-1) if dt_bias is not None else None,
        ddt.view(-1),
        dA_partial.view(-1),
        dbias_partial.view(-1) if dbias_partial is not None else None,
        dA.view(-1),
        dbias.view(-1) if dbias is not None else None,
        batch,
        seqlen,
        nheads,
        nchunks,
        dt_limit[0],
        dt_limit[1],
    )
    return ddt, dA, dbias


@cache
def _compile_chunk_state_fwd(
    b_dtype,
    x_dtype,
    dt_dtype,
    da_dtype,
    output_dtype,
    index_dtype,
    *,
    has_seq_idx: bool,
):
    class ChunkStateFwd:
        @cute.jit
        def __call__(
            self,
            mB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mStates: cute.Tensor,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
            stream: cuda.CUstream,
        ):
            count = Int64(batch) * nchunks * nheads * headdim * dstate
            self.kernel(
                mB,
                mX,
                mDt,
                mDA,
                mSeqIdx,
                mStates,
                count,
                seqlen,
                nheads,
                ngroups,
                nchunks,
                chunk_size,
                headdim,
                dstate,
            ).launch(
                grid=[cute.ceil_div(count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mStates: cute.Tensor,
            count: Int64,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < count:
                state_feature = linear % dstate
                outer = linear // dstate
                head_feature = outer % headdim
                outer = outer // headdim
                head = Int32(outer % nheads)
                outer = outer // nheads
                chunk = Int32(outer % nchunks)
                batch_idx = Int32(outer // nchunks)
                group = head // (nheads // ngroups)
                chunk_start = Int32(chunk * chunk_size)
                chunk_end = min(chunk_start + chunk_size, seqlen)
                last_seq = Int32(0)
                if const_expr(has_seq_idx):
                    last_seq = Int32(mSeqIdx[Int64(batch_idx) * seqlen + chunk_end - 1])

                da_base = (
                    (Int64(batch_idx) * nheads + head) * nchunks + chunk
                ) * chunk_size
                da_last = Float32(mDA[da_base + chunk_size - 1])
                value = Float32(0.0)
                for token in cutlass.range(chunk_start, chunk_end, 1, unroll=1):
                    pos = token - chunk_start
                    use_token = True
                    if const_expr(has_seq_idx):
                        use_token = (
                            Int32(mSeqIdx[Int64(batch_idx) * seqlen + token])
                            == last_seq
                        )
                    if use_token:
                        exponent = da_last - Float32(mDA[da_base + pos])
                        if exponent > Float32(0.0):
                            exponent = Float32(0.0)
                        scale = cute.math.exp2(
                            exponent * Float32(_LOG2E), fastmath=True
                        )
                        scale *= Float32(mDt[da_base + pos])
                        x_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * headdim + head_feature
                        b_idx = (
                            (Int64(batch_idx) * seqlen + token) * ngroups + group
                        ) * dstate + state_feature
                        value += Float32(mX[x_idx]) * Float32(mB[b_idx]) * scale
                mStates[linear] = output_dtype(value)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkStateFwd(),
        fake(b_dtype),
        fake(x_dtype),
        fake(dt_dtype),
        fake(da_dtype),
        fake(index_dtype) if has_seq_idx else None,
        fake(output_dtype),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_state_fwd_cute(
    B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    expected_state_shape = (batch, nchunks, nheads, headdim, dstate)
    if nheads % ngroups != 0 or B.shape[:2] != (batch, seqlen):
        raise ValueError("CuTe chunk state input shapes are incompatible")
    if dA_cumsum.shape != dt.shape or (
        seq_idx is not None and seq_idx.shape != (batch, seqlen)
    ):
        raise ValueError("CuTe chunk state metadata shapes are incompatible")
    if states is not None and states.shape != expected_state_shape:
        raise ValueError("CuTe chunk state output has an incompatible shape")
    tensors = (B, x, dt, dA_cumsum, seq_idx, states)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd

        return _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32)
    if states is None:
        states = torch.empty(
            expected_state_shape,
            device=x.device,
            dtype=torch.float32 if states_in_fp32 else B.dtype,
        )
    compiled = _compile_chunk_state_fwd(
        torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[states.dtype],
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        B.view(-1),
        x.view(-1),
        dt.view(-1),
        dA_cumsum.view(-1),
        seq_idx.view(-1) if seq_idx is not None else None,
        states.view(-1),
        batch,
        seqlen,
        nheads,
        ngroups,
        nchunks,
        chunk_size,
        headdim,
        dstate,
    )
    return states


@cache
def _compile_chunk_state_bwd_dx(
    b_dtype, x_dtype, dt_dtype, da_dtype, dstate_dtype, dx_dtype
):
    class ChunkStateBwdDx:
        @cute.jit
        def __call__(
            self,
            mB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mDStates: cute.Tensor,
            mDX: cute.Tensor,
            mDDt: cute.Tensor,
            mDDA: cute.Tensor,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
            stream: cuda.CUstream,
        ):
            rows = Int64(batch) * nchunks * nheads * chunk_size
            self.kernel(
                mB,
                mX,
                mDt,
                mDA,
                mDStates,
                mDX,
                mDDt,
                mDDA,
                rows,
                seqlen,
                nheads,
                ngroups,
                nchunks,
                chunk_size,
                headdim,
                dstate,
            ).launch(
                grid=[rows, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=_NUM_WARPS * 4,
                stream=stream,
            )
            outer = Int64(batch) * nchunks * nheads
            self.finalize_kernel(mDDA, outer, chunk_size).launch(
                grid=[cute.ceil_div(outer, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mDStates: cute.Tensor,
            mDX: cute.Tensor,
            mDDt: cute.Tensor,
            mDDA: cute.Tensor,
            rows: Int64,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if Int64(row) < rows:
                pos = Int32(row % chunk_size)
                outer = row // chunk_size
                head = Int32(outer % nheads)
                outer = outer // nheads
                chunk = Int32(outer % nchunks)
                batch_idx = Int32(outer // nchunks)
                token = chunk * chunk_size + pos
                group = head // (nheads // ngroups)
                value = Float32(0.0)
                if token < seqlen and tidx < headdim:
                    for state_feature in cutlass.range(0, dstate, 1, unroll=1):
                        b_idx = (
                            (Int64(batch_idx) * seqlen + token) * ngroups + group
                        ) * dstate + state_feature
                        ds_idx = (
                            ((Int64(batch_idx) * nchunks + chunk) * nheads + head)
                            * headdim
                            + tidx
                        ) * dstate + state_feature
                        value += Float32(mB[b_idx]) * Float32(mDStates[ds_idx])
                    da_base = (
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size
                    da_idx = da_base + pos
                    exponent = Float32(mDA[da_base + chunk_size - 1]) - Float32(
                        mDA[da_idx]
                    )
                    if exponent > Float32(0.0):
                        exponent = Float32(0.0)
                    scale = cute.math.exp2(exponent * Float32(_LOG2E), fastmath=True)
                    dt = Float32(mDt[da_idx])
                    x_idx = (
                        (Int64(batch_idx) * seqlen + token) * nheads + head
                    ) * headdim + tidx
                    mDX[x_idx] = dx_dtype(value * scale * dt)
                    value = value * scale * Float32(mX[x_idx])

                smem = cutlass.utils.SmemAllocator()
                reduction_buffer = smem.allocate_tensor(
                    Float32, cute.make_layout((1, (_NUM_WARPS, 1))), byte_alignment=16
                )
                total = row_reduce(
                    value,
                    cute.ReductionOp.ADD,
                    _NUM_THREADS,
                    reduction_buffer=reduction_buffer,
                )
                if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
                    da_idx = (
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size + pos
                    if token < seqlen:
                        mDDt[da_idx] = total
                        mDDA[da_idx] = -total * Float32(mDt[da_idx])
                    else:
                        mDDt[da_idx] = Float32(0.0)
                        mDDA[da_idx] = Float32(0.0)

        @cute.kernel
        def finalize_kernel(self, mDDA: cute.Tensor, outer: Int64, chunk_size: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            idx = Int64(block) * _NUM_THREADS + tidx
            if idx < outer:
                base = idx * chunk_size
                total = Float32(0.0)
                for pos in cutlass.range(0, chunk_size, 1, unroll=1):
                    total += Float32(mDDA[base + pos])
                mDDA[base + chunk_size - 1] -= total

    def fake(dtype):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        ChunkStateBwdDx(),
        fake(b_dtype),
        fake(x_dtype),
        fake(dt_dtype),
        fake(da_dtype),
        fake(dstate_dtype),
        fake(dx_dtype),
        fake(Float32),
        fake(Float32),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_state_bwd_dx_cute(B, x, dt, dA_cumsum, dstates, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    if headdim > _NUM_THREADS:
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_dx

        return _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx)
    tensors = (B, x, dt, dA_cumsum, dstates, dx)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_dx

        return _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx)
    if dx is None:
        dx = torch.empty_like(x)
    ddt = torch.empty_like(dt, dtype=torch.float32)
    ddA = torch.empty_like(dA_cumsum, dtype=torch.float32)
    compiled = _compile_chunk_state_bwd_dx(
        torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[dstates.dtype],
        torch2cute_dtype_map[dx.dtype],
    )
    compiled(
        B.view(-1),
        x.view(-1),
        dt.view(-1),
        dA_cumsum.view(-1),
        dstates.view(-1),
        dx.view(-1),
        ddt.view(-1),
        ddA.view(-1),
        batch,
        seqlen,
        nheads,
        ngroups,
        nchunks,
        chunk_size,
        headdim,
        dstate,
    )
    return dx, ddt, ddA


@cache
def _compile_chunk_state_bwd_db(
    x_dtype,
    dt_dtype,
    da_dtype,
    dstate_dtype,
    b_dtype,
    index_dtype,
    *,
    has_b: bool,
    has_seq_idx: bool,
):
    class ChunkStateBwdDb:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mDStates: cute.Tensor,
            mB: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mDB: cute.Tensor,
            mDDA: cute.Tensor | None,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
            stream: cuda.CUstream,
        ):
            db_rows = Int64(batch) * nchunks * ngroups * chunk_size
            self.db_kernel(
                mX,
                mDt,
                mDA,
                mDStates,
                mSeqIdx,
                mDB,
                db_rows,
                seqlen,
                nheads,
                ngroups,
                nchunks,
                chunk_size,
                headdim,
                dstate,
            ).launch(
                grid=[db_rows, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )
            if const_expr(has_b):
                dda_rows = Int64(batch) * nchunks * nheads * chunk_size
                self.dda_kernel(
                    mX,
                    mDt,
                    mDA,
                    mDStates,
                    mB,
                    mSeqIdx,
                    mDDA,
                    dda_rows,
                    seqlen,
                    nheads,
                    ngroups,
                    nchunks,
                    chunk_size,
                    headdim,
                    dstate,
                ).launch(
                    grid=[dda_rows, 1, 1],
                    block=[_NUM_THREADS, 1, 1],
                    smem=_NUM_WARPS * 4,
                    stream=stream,
                )
                outer = Int64(batch) * nchunks * nheads
                self.prefix_kernel(mDDA, outer, chunk_size).launch(
                    grid=[cute.ceil_div(outer, _NUM_THREADS), 1, 1],
                    block=[_NUM_THREADS, 1, 1],
                    stream=stream,
                )

        @cute.kernel
        def db_kernel(
            self,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mDStates: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mDB: cute.Tensor,
            rows: Int64,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
        ):
            state_feature, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if Int64(row) < rows:
                pos = Int32(row % chunk_size)
                outer = row // chunk_size
                group = Int32(outer % ngroups)
                outer = outer // ngroups
                chunk = Int32(outer % nchunks)
                batch_idx = Int32(outer // nchunks)
                token = chunk * chunk_size + pos
                if token < seqlen and state_feature < dstate:
                    chunk_end = min((chunk + 1) * chunk_size, seqlen)
                    use_token = True
                    if const_expr(has_seq_idx):
                        use_token = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + token]
                        ) == Int32(mSeqIdx[Int64(batch_idx) * seqlen + chunk_end - 1])
                    value = Float32(0.0)
                    if use_token:
                        heads_per_group = nheads // ngroups
                        for local_head in cutlass.range(
                            0, heads_per_group, 1, unroll=1
                        ):
                            head = group * heads_per_group + local_head
                            da_base = (
                                (Int64(batch_idx) * nheads + head) * nchunks + chunk
                            ) * chunk_size
                            exponent = Float32(mDA[da_base + chunk_size - 1]) - Float32(
                                mDA[da_base + pos]
                            )
                            if exponent > Float32(0.0):
                                exponent = Float32(0.0)
                            scale = cute.math.exp2(
                                exponent * Float32(_LOG2E), fastmath=True
                            ) * Float32(mDt[da_base + pos])
                            contraction = Float32(0.0)
                            for feature in cutlass.range(0, headdim, 1, unroll=1):
                                x_idx = (
                                    (Int64(batch_idx) * seqlen + token) * nheads + head
                                ) * headdim + feature
                                ds_idx = (
                                    (
                                        (Int64(batch_idx) * nchunks + chunk) * nheads
                                        + head
                                    )
                                    * headdim
                                    + feature
                                ) * dstate + state_feature
                                contraction += Float32(mX[x_idx]) * Float32(
                                    mDStates[ds_idx]
                                )
                            value += contraction * scale
                    db_idx = (
                        (Int64(batch_idx) * seqlen + token) * ngroups + group
                    ) * dstate + state_feature
                    mDB[db_idx] = value

        @cute.kernel
        def dda_kernel(
            self,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mDStates: cute.Tensor,
            mB: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mDDA: cute.Tensor,
            rows: Int64,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
        ):
            state_feature, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if Int64(row) < rows:
                pos = Int32(row % chunk_size)
                outer = row // chunk_size
                head = Int32(outer % nheads)
                outer = outer // nheads
                chunk = Int32(outer % nchunks)
                batch_idx = Int32(outer // nchunks)
                token = chunk * chunk_size + pos
                group = head // (nheads // ngroups)
                value = Float32(0.0)
                if token < seqlen and state_feature < dstate:
                    chunk_end = min((chunk + 1) * chunk_size, seqlen)
                    use_token = True
                    if const_expr(has_seq_idx):
                        use_token = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + token]
                        ) == Int32(mSeqIdx[Int64(batch_idx) * seqlen + chunk_end - 1])
                    if use_token:
                        contraction = Float32(0.0)
                        for feature in cutlass.range(0, headdim, 1, unroll=1):
                            x_idx = (
                                (Int64(batch_idx) * seqlen + token) * nheads + head
                            ) * headdim + feature
                            ds_idx = (
                                ((Int64(batch_idx) * nchunks + chunk) * nheads + head)
                                * headdim
                                + feature
                            ) * dstate + state_feature
                            contraction += Float32(mX[x_idx]) * Float32(
                                mDStates[ds_idx]
                            )
                        da_base = (
                            (Int64(batch_idx) * nheads + head) * nchunks + chunk
                        ) * chunk_size
                        exponent = Float32(mDA[da_base + chunk_size - 1]) - Float32(
                            mDA[da_base + pos]
                        )
                        if exponent > Float32(0.0):
                            exponent = Float32(0.0)
                        scale = cute.math.exp2(
                            exponent * Float32(_LOG2E), fastmath=True
                        ) * Float32(mDt[da_base + pos])
                        b_idx = (
                            (Int64(batch_idx) * seqlen + token) * ngroups + group
                        ) * dstate + state_feature
                        value = contraction * scale * Float32(mB[b_idx])

                smem = cutlass.utils.SmemAllocator()
                reduction_buffer = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((1, (_NUM_WARPS, 1))),
                    byte_alignment=16,
                )
                total = row_reduce(
                    value,
                    cute.ReductionOp.ADD,
                    _NUM_THREADS,
                    reduction_buffer=reduction_buffer,
                )
                if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
                    da_base = (
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size
                    if token < seqlen and pos < chunk_size - 1:
                        mDDA[da_base + pos + 1] = total

        @cute.kernel
        def prefix_kernel(self, mDDA: cute.Tensor, outer: Int64, chunk_size: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            idx = Int64(block) * _NUM_THREADS + tidx
            if idx < outer:
                base = idx * chunk_size
                total = Float32(0.0)
                for pos in cutlass.range(0, chunk_size, 1, unroll=1):
                    total += Float32(mDDA[base + pos])
                    mDDA[base + pos] = total

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkStateBwdDb(),
        fake(x_dtype),
        fake(dt_dtype),
        fake(da_dtype),
        fake(dstate_dtype),
        fake(b_dtype) if has_b else None,
        fake(index_dtype) if has_seq_idx else None,
        fake(Float32),
        fake(Float32) if has_b else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_state_bwd_db_cute(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    if dstate > _NUM_THREADS or nheads % ngroups != 0:
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_db

        return _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx, B, ngroups)
    tensors = (x, dt, dA_cumsum, dstates, seq_idx, B)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_db

        return _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx, B, ngroups)
    dB = torch.empty(
        batch, seqlen, ngroups, dstate, device=x.device, dtype=torch.float32
    )
    ddA = torch.zeros_like(dA_cumsum, dtype=torch.float32) if B is not None else None
    compiled = _compile_chunk_state_bwd_db(
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[dstates.dtype],
        torch2cute_dtype_map[B.dtype] if B is not None else None,
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_b=B is not None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        x.view(-1),
        dt.view(-1),
        dA_cumsum.view(-1),
        dstates.view(-1),
        B.view(-1) if B is not None else None,
        seq_idx.view(-1) if seq_idx is not None else None,
        dB.view(-1),
        ddA.view(-1) if ddA is not None else None,
        batch,
        seqlen,
        nheads,
        ngroups,
        nchunks,
        chunk_size,
        headdim,
        dstate,
    )
    return dB if B is None else (dB, ddA)


def chunk_state_bwd_ddAcs_stable_cute(B, x, dt, dA_cumsum, dstates, seq_idx=None):
    _, ddA_cumsum = chunk_state_bwd_db_cute(
        x, dt, dA_cumsum, dstates, seq_idx, B, B.shape[2]
    )
    return ddA_cumsum


class ChunkStateCuteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        if B.stride(-1) != 1:
            B = B.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        states = chunk_state_fwd_cute(
            B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32
        )
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = chunk_state_bwd_dx_cute(B, x, dt, dA_cumsum, dstates)
        dB = chunk_state_bwd_db_cute(x, dt, dA_cumsum, dstates, ngroups=B.shape[2])
        return dB.to(B.dtype), dx, ddt, ddA_cumsum, None


def chunk_state_cute(B, x, dt, dA_cumsum, states_in_fp32=True):
    return ChunkStateCuteFn.apply(B, x, dt, dA_cumsum, states_in_fp32)


__all__ = [
    "chunk_cumsum_bwd_cute",
    "chunk_cumsum_fwd_cute",
    "chunk_state_bwd_db_cute",
    "chunk_state_bwd_ddAcs_stable_cute",
    "chunk_state_bwd_dx_cute",
    "chunk_state_cute",
    "chunk_state_fwd_cute",
]
