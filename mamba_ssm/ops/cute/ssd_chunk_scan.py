# Copyright (c) 2026, NVIDIA CORPORATION.

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import sigmoid
from quack.cute_dsl_utils import torch2cute_dtype_map


_LOG2E = 1.4426950408889634
_NUM_THREADS = 256


@cache
def _compile_chunk_scan_fwd(
    cb_dtype,
    x_dtype,
    dt_dtype,
    da_dtype,
    c_dtype,
    state_dtype,
    d_dtype,
    z_dtype,
    index_dtype,
    *,
    has_d: bool,
    d_has_hdim: bool,
    has_z: bool,
    has_seq_idx: bool,
):
    class ChunkScanFwd:
        @cute.jit
        def __call__(
            self,
            mCB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mC: cute.Tensor,
            mStates: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mOut: cute.Tensor,
            mOutX: cute.Tensor | None,
            count: Int64,
            seqlen: Int32,
            nheads: Int32,
            ngroups: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            dstate: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mCB,
                mX,
                mDt,
                mDA,
                mC,
                mStates,
                mD,
                mZ,
                mSeqIdx,
                mOut,
                mOutX,
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
            mCB: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mDA: cute.Tensor,
            mC: cute.Tensor,
            mStates: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mOut: cute.Tensor,
            mOutX: cute.Tensor | None,
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
                feature = Int32(linear % headdim)
                outer = linear // headdim
                head = Int32(outer % nheads)
                outer = outer // nheads
                token = Int32(outer % seqlen)
                batch_idx = Int32(outer // seqlen)
                chunk = token // chunk_size
                pos = token - chunk * chunk_size
                group = head // (nheads // ngroups)
                da_base = (
                    (Int64(batch_idx) * nheads + head) * nchunks + chunk
                ) * chunk_size
                da_pos = Float32(mDA[da_base + pos])
                value = Float32(0.0)

                use_state = True
                if const_expr(has_seq_idx):
                    prev_seq = Int32(0)
                    if chunk > 0:
                        prev_seq = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + chunk * chunk_size - 1]
                        )
                    use_state = (
                        Int32(mSeqIdx[Int64(batch_idx) * seqlen + token]) == prev_seq
                    )
                if use_state:
                    state_value = Float32(0.0)
                    for state_feature in cutlass.range(0, dstate, 1, unroll=1):
                        c_idx = (
                            (Int64(batch_idx) * seqlen + token) * ngroups + group
                        ) * dstate + state_feature
                        state_idx = (
                            ((Int64(batch_idx) * nchunks + chunk) * nheads + head)
                            * headdim
                            + feature
                        ) * dstate + state_feature
                        state_value += Float32(mC[c_idx]) * Float32(mStates[state_idx])
                    value += state_value * cute.math.exp2(
                        da_pos * Float32(_LOG2E), fastmath=True
                    )

                cb_base = (
                    ((Int64(batch_idx) * nchunks + chunk) * ngroups + group)
                    * chunk_size
                    + pos
                ) * chunk_size
                chunk_token_start = chunk * chunk_size
                for source_pos in cutlass.range(0, pos + 1, 1, unroll=1):
                    exponent = da_pos - Float32(mDA[da_base + source_pos])
                    if exponent > Float32(0.0):
                        exponent = Float32(0.0)
                    scale = cute.math.exp2(
                        exponent * Float32(_LOG2E), fastmath=True
                    ) * Float32(mDt[da_base + source_pos])
                    source_token = chunk_token_start + source_pos
                    x_idx = (
                        (Int64(batch_idx) * seqlen + source_token) * nheads + head
                    ) * headdim + feature
                    value += (
                        Float32(mCB[cb_base + source_pos]) * Float32(mX[x_idx]) * scale
                    )

                x_idx = (
                    (Int64(batch_idx) * seqlen + token) * nheads + head
                ) * headdim + feature
                if const_expr(has_d):
                    d_idx = Int64(head)
                    if const_expr(d_has_hdim):
                        d_idx = d_idx * headdim + feature
                    value += Float32(mX[x_idx]) * Float32(mD[d_idx])
                if const_expr(has_z):
                    mOutX[linear] = x_dtype(value)
                    z = Float32(mZ[linear])
                    value *= z * sigmoid(z)
                mOut[linear] = x_dtype(value)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkScanFwd(),
        fake(cb_dtype),
        fake(x_dtype),
        fake(dt_dtype),
        fake(da_dtype),
        fake(c_dtype),
        fake(state_dtype),
        fake(d_dtype) if has_d else None,
        fake(z_dtype) if has_z else None,
        fake(index_dtype) if has_seq_idx else None,
        fake(x_dtype),
        fake(x_dtype) if has_z else None,
        Int64(0),
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


def chunk_scan_fwd_cute(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    tensors = (cb, x, dt, dA_cumsum, C, states, D, z, seq_idx)
    if nheads % ngroups != 0 or any(
        t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors
    ):
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd

        return _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D, z, seq_idx)
    out = torch.empty_like(x)
    out_x = torch.empty_like(x) if z is not None else None
    compiled = _compile_chunk_scan_fwd(
        torch2cute_dtype_map[cb.dtype],
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[C.dtype],
        torch2cute_dtype_map[states.dtype],
        torch2cute_dtype_map[D.dtype] if D is not None else None,
        torch2cute_dtype_map[z.dtype] if z is not None else None,
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_d=D is not None,
        d_has_hdim=D is not None and D.ndim == 2,
        has_z=z is not None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        cb.view(-1),
        x.view(-1),
        dt.view(-1),
        dA_cumsum.view(-1),
        C.view(-1),
        states.view(-1),
        D.view(-1) if D is not None else None,
        z.view(-1) if z is not None else None,
        seq_idx.view(-1) if seq_idx is not None else None,
        out.view(-1),
        out_x.view(-1) if out_x is not None else None,
        out.numel(),
        seqlen,
        nheads,
        ngroups,
        nchunks,
        chunk_size,
        headdim,
        dstate,
    )
    return out, out_x


__all__ = ["chunk_scan_fwd_cute"]
