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


@cache
def _compile_chunk_scan_bwd_dstates(
    c_dtype, da_dtype, dout_dtype, index_dtype, output_dtype, *, has_seq_idx: bool
):
    class ChunkScanBwdDStates:
        @cute.jit
        def __call__(
            self,
            mC: cute.Tensor,
            mDA: cute.Tensor,
            mDOut: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mDStates: cute.Tensor,
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
                mC,
                mDA,
                mDOut,
                mSeqIdx,
                mDStates,
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
            mC: cute.Tensor,
            mDA: cute.Tensor,
            mDOut: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mDStates: cute.Tensor,
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
                state_feature = Int32(linear % dstate)
                outer = linear // dstate
                feature = Int32(outer % headdim)
                outer = outer // headdim
                head = Int32(outer % nheads)
                outer = outer // nheads
                chunk = Int32(outer % nchunks)
                batch_idx = Int32(outer // nchunks)
                group = head // (nheads // ngroups)
                chunk_start = chunk * chunk_size
                chunk_end = min(chunk_start + chunk_size, seqlen)
                prev_seq = Int32(0)
                if const_expr(has_seq_idx):
                    if chunk > 0:
                        prev_seq = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + chunk_start - 1]
                        )
                da_base = (
                    (Int64(batch_idx) * nheads + head) * nchunks + chunk
                ) * chunk_size
                value = Float32(0.0)
                for token in cutlass.range(chunk_start, chunk_end, 1, unroll=1):
                    use_token = True
                    if const_expr(has_seq_idx):
                        use_token = (
                            Int32(mSeqIdx[Int64(batch_idx) * seqlen + token])
                            == prev_seq
                        )
                    if use_token:
                        pos = token - chunk_start
                        scale = cute.math.exp2(
                            Float32(mDA[da_base + pos]) * Float32(_LOG2E),
                            fastmath=True,
                        )
                        c_idx = (
                            (Int64(batch_idx) * seqlen + token) * ngroups + group
                        ) * dstate + state_feature
                        dout_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * headdim + feature
                        value += Float32(mC[c_idx]) * Float32(mDOut[dout_idx]) * scale
                mDStates[linear] = output_dtype(value)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkScanBwdDStates(),
        fake(c_dtype),
        fake(da_dtype),
        fake(dout_dtype),
        fake(index_dtype) if has_seq_idx else None,
        fake(output_dtype),
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


def chunk_scan_bwd_dstates_cute(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    output_dtype = C.dtype if dtype is None else dtype
    tensors = (C, dA_cumsum, dout, seq_idx)
    if nheads % ngroups != 0 or any(
        t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors
    ):
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates

        return _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx, dtype)
    dstates = torch.empty(
        batch,
        nchunks,
        nheads,
        headdim,
        dstate,
        device=C.device,
        dtype=output_dtype,
    )
    compiled = _compile_chunk_scan_bwd_dstates(
        torch2cute_dtype_map[C.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[dout.dtype],
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        torch2cute_dtype_map[output_dtype],
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        C.view(-1),
        dA_cumsum.view(-1),
        dout.view(-1),
        seq_idx.view(-1) if seq_idx is not None else None,
        dstates.view(-1),
        dstates.numel(),
        seqlen,
        nheads,
        ngroups,
        nchunks,
        chunk_size,
        headdim,
        dstate,
    )
    return dstates


@cache
def _compile_chunk_scan_bwd_dc(
    state_dtype,
    da_dtype,
    dout_dtype,
    c_dtype,
    index_dtype,
    *,
    has_c: bool,
    has_seq_idx: bool,
):
    class ChunkScanBwdDC:
        @cute.jit
        def __call__(
            self,
            mStates: cute.Tensor,
            mDA: cute.Tensor,
            mDOut: cute.Tensor,
            mC: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mDC: cute.Tensor,
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
            dc_rows = Int64(batch) * seqlen * ngroups
            self.dc_kernel(
                mStates,
                mDA,
                mDOut,
                mSeqIdx,
                mDC,
                dc_rows,
                seqlen,
                nheads,
                ngroups,
                nchunks,
                chunk_size,
                headdim,
                dstate,
            ).launch(
                grid=[dc_rows, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )
            if const_expr(has_c):
                dda_rows = Int64(batch) * seqlen * nheads
                self.dda_kernel(
                    mStates,
                    mDA,
                    mDOut,
                    mC,
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

        @cute.kernel
        def dc_kernel(
            self,
            mStates: cute.Tensor,
            mDA: cute.Tensor,
            mDOut: cute.Tensor,
            mSeqIdx: cute.Tensor | None,
            mDC: cute.Tensor,
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
                group = Int32(row % ngroups)
                outer = row // ngroups
                token = Int32(outer % seqlen)
                batch_idx = Int32(outer // seqlen)
                chunk = token // chunk_size
                pos = token - chunk * chunk_size
                use_token = True
                if const_expr(has_seq_idx):
                    prev_seq = Int32(0)
                    if chunk > 0:
                        prev_seq = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + chunk * chunk_size - 1]
                        )
                    use_token = (
                        Int32(mSeqIdx[Int64(batch_idx) * seqlen + token]) == prev_seq
                    )
                if state_feature < dstate:
                    value = Float32(0.0)
                    if use_token:
                        heads_per_group = nheads // ngroups
                        for local_head in cutlass.range(
                            0, heads_per_group, 1, unroll=1
                        ):
                            head = group * heads_per_group + local_head
                            contraction = Float32(0.0)
                            for feature in cutlass.range(0, headdim, 1, unroll=1):
                                dout_idx = (
                                    (Int64(batch_idx) * seqlen + token) * nheads + head
                                ) * headdim + feature
                                state_idx = (
                                    (
                                        (Int64(batch_idx) * nchunks + chunk) * nheads
                                        + head
                                    )
                                    * headdim
                                    + feature
                                ) * dstate + state_feature
                                contraction += Float32(mDOut[dout_idx]) * Float32(
                                    mStates[state_idx]
                                )
                            da_idx = (
                                (Int64(batch_idx) * nheads + head) * nchunks + chunk
                            ) * chunk_size + pos
                            value += contraction * cute.math.exp2(
                                Float32(mDA[da_idx]) * Float32(_LOG2E),
                                fastmath=True,
                            )
                    dc_idx = (
                        (Int64(batch_idx) * seqlen + token) * ngroups + group
                    ) * dstate + state_feature
                    mDC[dc_idx] = value

        @cute.kernel
        def dda_kernel(
            self,
            mStates: cute.Tensor,
            mDA: cute.Tensor,
            mDOut: cute.Tensor,
            mC: cute.Tensor,
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
                head = Int32(row % nheads)
                outer = row // nheads
                token = Int32(outer % seqlen)
                batch_idx = Int32(outer // seqlen)
                chunk = token // chunk_size
                pos = token - chunk * chunk_size
                group = head // (nheads // ngroups)
                use_token = True
                if const_expr(has_seq_idx):
                    prev_seq = Int32(0)
                    if chunk > 0:
                        prev_seq = Int32(
                            mSeqIdx[Int64(batch_idx) * seqlen + chunk * chunk_size - 1]
                        )
                    use_token = (
                        Int32(mSeqIdx[Int64(batch_idx) * seqlen + token]) == prev_seq
                    )
                value = Float32(0.0)
                if use_token and state_feature < dstate:
                    contraction = Float32(0.0)
                    for feature in cutlass.range(0, headdim, 1, unroll=1):
                        dout_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * headdim + feature
                        state_idx = (
                            ((Int64(batch_idx) * nchunks + chunk) * nheads + head)
                            * headdim
                            + feature
                        ) * dstate + state_feature
                        contraction += Float32(mDOut[dout_idx]) * Float32(
                            mStates[state_idx]
                        )
                    da_idx = (
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size + pos
                    c_idx = (
                        (Int64(batch_idx) * seqlen + token) * ngroups + group
                    ) * dstate + state_feature
                    value = (
                        contraction
                        * cute.math.exp2(
                            Float32(mDA[da_idx]) * Float32(_LOG2E), fastmath=True
                        )
                        * Float32(mC[c_idx])
                    )
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
                    da_idx = (
                        (Int64(batch_idx) * nheads + head) * nchunks + chunk
                    ) * chunk_size + pos
                    mDDA[da_idx] = total

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        ChunkScanBwdDC(),
        fake(state_dtype),
        fake(da_dtype),
        fake(dout_dtype),
        fake(c_dtype) if has_c else None,
        fake(index_dtype) if has_seq_idx else None,
        fake(Float32),
        fake(Float32) if has_c else None,
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


def chunk_scan_bwd_dC_cute(
    prev_states, dA_cumsum, dout, seq_idx=None, C=None, ngroups=1
):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    chunk_size = dA_cumsum.shape[-1]
    tensors = (prev_states, dA_cumsum, dout, seq_idx, C)
    if (
        dstate > _NUM_THREADS
        or nheads % ngroups != 0
        or any(
            t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors
        )
    ):
        from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC

        return _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx, C, ngroups)
    dC = torch.empty(
        batch, seqlen, ngroups, dstate, device=dout.device, dtype=torch.float32
    )
    ddA = torch.empty_like(dA_cumsum, dtype=torch.float32) if C is not None else None
    compiled = _compile_chunk_scan_bwd_dc(
        torch2cute_dtype_map[prev_states.dtype],
        torch2cute_dtype_map[dA_cumsum.dtype],
        torch2cute_dtype_map[dout.dtype],
        torch2cute_dtype_map[C.dtype] if C is not None else None,
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_c=C is not None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        prev_states.view(-1),
        dA_cumsum.view(-1),
        dout.view(-1),
        C.view(-1) if C is not None else None,
        seq_idx.view(-1) if seq_idx is not None else None,
        dC.view(-1),
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
    return dC if C is None else (dC, ddA)


__all__ = [
    "chunk_scan_bwd_dC_cute",
    "chunk_scan_bwd_dstates_cute",
    "chunk_scan_bwd_dz_cute",
    "chunk_scan_fwd_cute",
]
