# Copyright (c) 2024, Tri Dao, Albert Gu.

from __future__ import annotations

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import silu
from quack.cute_dsl_utils import torch2cute_dtype_map


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_NUM_THREADS = 128
_WARPS_PER_CTA = _NUM_THREADS // 32


@cute.jit
def _softplus(value: Float32) -> Float32:
    if value <= Float32(20.0):
        exponent = cute.math.exp2(value * Float32(_LOG2E), fastmath=True)
        value = cute.math.log2(exponent + Float32(1.0), fastmath=True) * Float32(_LN2)
    return value


@cache
def _compile_selective_state_update(
    state_dtype,
    x_dtype,
    dt_dtype,
    a_dtype,
    b_dtype,
    c_dtype,
    d_dtype,
    z_dtype,
    dt_bias_dtype,
    index_dtype,
    dstate: int,
    *,
    has_d: bool,
    has_z: bool,
    has_dt_bias: bool,
    has_state_batch_indices: bool,
    dt_softplus: bool,
):
    lanes_per_feature = min(32, max(4, dstate // 4))
    features_per_warp = 32 // lanes_per_feature
    features_per_cta = _WARPS_PER_CTA * features_per_warp
    items_per_lane = math.ceil(dstate / lanes_per_feature)
    reduction_offsets = tuple(1 << i for i in range(int(math.log2(lanes_per_feature))))

    class SelectiveStateUpdate:
        @cute.jit
        def __call__(
            self,
            mState: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mOut: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mDtBias: cute.Tensor | None,
            mStateBatchIndices: cute.Tensor | None,
            batch: Int32,
            nheads: Int32,
            dim: Int32,
            ngroups: Int32,
            state_batch_stride: Int64,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mState,
                mX,
                mDt,
                mA,
                mB,
                mC,
                mOut,
                mD,
                mZ,
                mDtBias,
                mStateBatchIndices,
                nheads,
                dim,
                ngroups,
                state_batch_stride,
            ).launch(
                grid=[cute.ceil_div(dim, features_per_cta), batch, nheads],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mState: cute.Tensor,
            mX: cute.Tensor,
            mDt: cute.Tensor,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mOut: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mDtBias: cute.Tensor | None,
            mStateBatchIndices: cute.Tensor | None,
            nheads: Int32,
            dim: Int32,
            ngroups: Int32,
            state_batch_stride: Int64,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block_d, batch_idx, head = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            feature_group = lane // lanes_per_feature
            state_lane = lane % lanes_per_feature
            feature = block_d * features_per_cta + warp * features_per_warp + feature_group

            if feature < dim:
                state_batch = Int32(batch_idx)
                if const_expr(has_state_batch_indices):
                    state_batch = Int32(mStateBatchIndices[batch_idx])
                if state_batch < Int32(0):
                    if state_lane == 0:
                        out_idx = (Int64(batch_idx) * nheads + head) * dim + feature
                        mOut[out_idx] = x_dtype(0.0)
                else:
                    token_idx = (Int64(batch_idx) * nheads + head) * dim + feature
                    param_idx = Int64(head) * dim + feature
                    group = head // (nheads // ngroups)
                    bc_base = (Int64(batch_idx) * ngroups + group) * dstate
                    state_base = Int64(state_batch) * state_batch_stride + (Int64(head) * dim + feature) * dstate

                    x = Float32(0.0)
                    dt = Float32(0.0)
                    if state_lane == 0:
                        x = Float32(mX[token_idx])
                        dt = Float32(mDt[token_idx])
                        if const_expr(has_dt_bias):
                            dt += Float32(mDtBias[param_idx])
                        if const_expr(dt_softplus):
                            dt = _softplus(dt)
                    source_lane = feature_group * lanes_per_feature
                    x = cute.arch.shuffle_sync(x, source_lane)
                    dt = cute.arch.shuffle_sync(dt, source_lane)

                    output = Float32(0.0)
                    for item in cutlass.range_constexpr(items_per_lane):
                        state_col = state_lane + item * lanes_per_feature
                        if state_col < dstate:
                            state_idx = state_base + state_col
                            state = Float32(mState[state_idx])
                            a = Float32(mA[param_idx * dstate + state_col])
                            b = Float32(mB[bc_base + state_col])
                            c = Float32(mC[bc_base + state_col])
                            decay = cute.math.exp2(a * dt * Float32(_LOG2E), fastmath=True)
                            state = state * decay + b * dt * x
                            mState[state_idx] = state_dtype(state)
                            output += state * c

                    for offset in reduction_offsets:
                        output += cute.arch.shuffle_sync_bfly(output, offset)
                    if state_lane == 0:
                        if const_expr(has_d):
                            output += x * Float32(mD[param_idx])
                        if const_expr(has_z):
                            output *= silu(Float32(mZ[token_idx]))
                        mOut[token_idx] = x_dtype(output)

    fake = lambda dtype, align=16: (
        cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=align)
        if dtype is not None
        else None
    )
    return cute.compile(
        SelectiveStateUpdate(),
        fake(state_dtype),
        fake(x_dtype),
        fake(dt_dtype),
        fake(a_dtype),
        fake(b_dtype),
        fake(c_dtype),
        fake(x_dtype),
        fake(d_dtype) if has_d else None,
        fake(z_dtype) if has_z else None,
        fake(dt_bias_dtype) if has_dt_bias else None,
        fake(index_dtype, 4) if has_state_batch_indices else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def selective_state_update_cute(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
):
    """CuTe implementation of the selective_state_update API."""
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    _, nheads, dim, dstate = state.shape
    if dstate not in (16, 32, 64, 128):
        raise ValueError("CuTe selective_state_update supports dstate 16, 32, 64, or 128")
    batch = x.shape[0]
    ngroups = B.shape[1]
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    assert nheads % ngroups == 0
    assert B.shape == C.shape == (batch, ngroups, dstate)
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)

    tensors = (state, x, dt, A, B, C, D, z, dt_bias)
    if any(t is not None and not t.is_contiguous() for t in tensors):
        raise ValueError("CuTe selective_state_update requires contiguous tensors")
    out = torch.empty_like(x)
    compiled = _compile_selective_state_update(
        torch2cute_dtype_map[state.dtype],
        torch2cute_dtype_map[x.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[A.dtype],
        torch2cute_dtype_map[B.dtype],
        torch2cute_dtype_map[C.dtype],
        torch2cute_dtype_map[D.dtype] if D is not None else None,
        torch2cute_dtype_map[z.dtype] if z is not None else None,
        torch2cute_dtype_map[dt_bias.dtype] if dt_bias is not None else None,
        torch2cute_dtype_map[state_batch_indices.dtype] if state_batch_indices is not None else None,
        dstate,
        has_d=D is not None,
        has_z=z is not None,
        has_dt_bias=dt_bias is not None,
        has_state_batch_indices=state_batch_indices is not None,
        dt_softplus=dt_softplus,
    )
    compiled(
        state.view(-1),
        x.view(-1),
        dt.view(-1),
        A.view(-1),
        B.view(-1),
        C.view(-1),
        out.view(-1),
        D.view(-1) if D is not None else None,
        z.view(-1) if z is not None else None,
        dt_bias.view(-1) if dt_bias is not None else None,
        state_batch_indices.view(-1) if state_batch_indices is not None else None,
        batch,
        nheads,
        dim,
        ngroups,
        state.stride(0),
    )
    return out if has_heads else out.squeeze(1)


__all__ = ["selective_state_update_cute"]
