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
from quack.reduce import row_reduce


_LOG2E = 1.4426950408889634
_NUM_THREADS = 256
_NUM_WARPS = _NUM_THREADS // 32


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


@cache
def _compile_chunk_scan_bwd_dz(
    x_dtype,
    z_dtype,
    out_dtype,
    dout_dtype,
    d_dtype,
    dz_dtype,
    *,
    has_ddacs: bool,
    has_d: bool,
    d_has_hdim: bool,
    recompute_output: bool,
):
    class ChunkScanBwdDz:
        @cute.jit
        def __call__(
            self,
            mX: cute.Tensor,
            mZ: cute.Tensor,
            mOut: cute.Tensor,
            mDOut: cute.Tensor,
            mD: cute.Tensor | None,
            mDz: cute.Tensor,
            mDOutX: cute.Tensor,
            mDDA: cute.Tensor | None,
            mOutZ: cute.Tensor | None,
            rows: Int64,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mX,
                mZ,
                mOut,
                mDOut,
                mD,
                mDz,
                mDOutX,
                mDDA,
                mOutZ,
                rows,
                seqlen,
                nheads,
                nchunks,
                chunk_size,
                headdim,
            ).launch(
                grid=[rows, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                smem=_NUM_WARPS * 4,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX: cute.Tensor,
            mZ: cute.Tensor,
            mOut: cute.Tensor,
            mDOut: cute.Tensor,
            mD: cute.Tensor | None,
            mDz: cute.Tensor,
            mDOutX: cute.Tensor,
            mDDA: cute.Tensor | None,
            mOutZ: cute.Tensor | None,
            rows: Int64,
            seqlen: Int32,
            nheads: Int32,
            nchunks: Int32,
            chunk_size: Int32,
            headdim: Int32,
        ):
            feature, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            if Int64(row) < rows:
                head = Int32(row % nheads)
                outer = row // nheads
                token = Int32(outer % seqlen)
                batch_idx = Int32(outer // seqlen)
                value = Float32(0.0)
                if feature < headdim:
                    idx = (
                        (Int64(batch_idx) * seqlen + token) * nheads + head
                    ) * headdim + feature
                    dout = Float32(mDOut[idx])
                    out = Float32(mOut[idx])
                    z = Float32(mZ[idx])
                    z_sigmoid = sigmoid(z)
                    mDz[idx] = dz_dtype(
                        dout
                        * out
                        * z_sigmoid
                        * (Float32(1.0) + z * (Float32(1.0) - z_sigmoid))
                    )
                    if const_expr(recompute_output):
                        mOutZ[idx] = out_dtype(out * z * z_sigmoid)
                    dout_x = dout * z * z_sigmoid
                    mDOutX[idx] = dout_dtype(dout_x)
                    if const_expr(has_d):
                        d_idx = Int64(head)
                        if const_expr(d_has_hdim):
                            d_idx = d_idx * headdim + feature
                        out -= Float32(mX[idx]) * Float32(mD[d_idx])
                    value = dout_x * out

                if const_expr(has_ddacs):
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
                        chunk = token // chunk_size
                        pos = token - chunk * chunk_size
                        dda_idx = (
                            (Int64(batch_idx) * nheads + head) * nchunks + chunk
                        ) * chunk_size + pos
                        mDDA[dda_idx] = total

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkScanBwdDz(),
        fake(x_dtype),
        fake(z_dtype),
        fake(out_dtype),
        fake(dout_dtype),
        fake(d_dtype) if has_d else None,
        fake(dz_dtype),
        fake(dout_dtype),
        fake(Float32) if has_ddacs else None,
        fake(out_dtype) if recompute_output else None,
        Int64(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def chunk_scan_bwd_dz_cute(
    x,
    z,
    out,
    dout,
    chunk_size,
    has_ddAcs=True,
    D=None,
    dz=None,
    recompute_output=False,
):
    batch, seqlen, nheads, headdim = x.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    tensors = (x, z, out, dout, D, dz)
    if headdim > _NUM_THREADS or any(
        t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors
    ):
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dz

        return _chunk_scan_bwd_dz(
            x, z, out, dout, chunk_size, has_ddAcs, D, dz, recompute_output
        )
    if dz is None:
        dz = torch.empty_like(z)
    dout_x = torch.empty_like(dout)
    ddA = (
        torch.empty(batch, nheads, nchunks, chunk_size, device=x.device)
        if has_ddAcs
        else None
    )
    outz = torch.empty_like(x) if recompute_output else None
    compiled = _compile_chunk_scan_bwd_dz(
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[z.dtype],
        torch2cute_dtype_map[out.dtype],
        torch2cute_dtype_map[dout.dtype],
        torch2cute_dtype_map[D.dtype] if D is not None else None,
        torch2cute_dtype_map[dz.dtype],
        has_ddacs=has_ddAcs,
        has_d=D is not None,
        d_has_hdim=D is not None and D.ndim == 2,
        recompute_output=recompute_output,
    )
    compiled(
        x.view(-1),
        z.view(-1),
        out.view(-1),
        dout.view(-1),
        D.view(-1) if D is not None else None,
        dz.view(-1),
        dout_x.view(-1),
        ddA.view(-1) if ddA is not None else None,
        outz.view(-1) if outz is not None else None,
        batch * seqlen * nheads,
        seqlen,
        nheads,
        nchunks,
        chunk_size,
        headdim,
    )
    if D is not None:
        dD = (dout_x.float() * x.float()).sum(dim=(0, 1))
        if D.ndim == 1:
            dD = dD.sum(dim=-1)
        dD = dD.to(D.dtype)
    else:
        dD = None
    result = (dz, dout_x, dD, ddA) if has_ddAcs else (dz, dout_x, dD)
    return (*result, outz) if recompute_output else result


__all__ = ["chunk_scan_bwd_dz_cute", "chunk_scan_fwd_cute"]
