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

from mamba_ssm.ops.cute.mamba3.packing import bf16x2_to_i32, i32_to_bf16x2


_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_NUM_THREADS = 128


@cache
def _is_hopper_or_newer(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] >= 9


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
    pack_bf16_state: bool,
):
    num_threads = 256 if dstate == 128 else _NUM_THREADS
    warps_per_cta = num_threads // 32
    state_values_per_lane = 4 if pack_bf16_state else 1
    lanes_per_feature = min(32, max(4, dstate // (4 * state_values_per_lane)))
    features_per_warp = 32 // lanes_per_feature
    features_per_cta = warps_per_cta * features_per_warp
    items_per_lane = math.ceil(dstate / (lanes_per_feature * state_values_per_lane))
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
                block=[num_threads, 1, 1],
                smem=2 * dstate * 4,
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
            feature = (
                block_d * features_per_cta + warp * features_per_warp + feature_group
            )
            group = head // (nheads // ngroups)
            bc_base = (Int64(batch_idx) * ngroups + group) * dstate

            smem = cutlass.utils.SmemAllocator()
            sB = smem.allocate_tensor(
                Float32, cute.make_layout((dstate,)), byte_alignment=16
            )
            sC = smem.allocate_tensor(
                Float32, cute.make_layout((dstate,)), byte_alignment=16
            )
            if const_expr(dstate >= 64):
                if tidx < dstate:
                    sB[tidx] = Float32(mB[bc_base + tidx])
                    sC[tidx] = Float32(mC[bc_base + tidx])
                cute.arch.sync_threads()

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
                    state_base = (
                        Int64(state_batch) * state_batch_stride
                        + (Int64(head) * dim + feature) * dstate
                    )

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
                    if const_expr(pack_bf16_state):
                        mStatePacked = cute.recast_tensor(mState, Int32)
                        for item in cutlass.range_constexpr(items_per_lane):
                            state_col = (
                                state_lane * state_values_per_lane
                                + item * lanes_per_feature * state_values_per_lane
                            )
                            packed_state_idx = state_base + state_col
                            state0_raw, state1_raw = i32_to_bf16x2(
                                mStatePacked[packed_state_idx // 2]
                            )
                            state2_raw, state3_raw = i32_to_bf16x2(
                                mStatePacked[packed_state_idx // 2 + 1]
                            )
                            state0 = Float32(state0_raw)
                            state1 = Float32(state1_raw)
                            state2 = Float32(state2_raw)
                            state3 = Float32(state3_raw)
                            a0 = Float32(mA[param_idx * dstate + state_col])
                            a1 = Float32(mA[param_idx * dstate + state_col + 1])
                            a2 = Float32(mA[param_idx * dstate + state_col + 2])
                            a3 = Float32(mA[param_idx * dstate + state_col + 3])
                            if const_expr(dstate >= 64):
                                b0 = sB[state_col]
                                b1 = sB[state_col + 1]
                                b2 = sB[state_col + 2]
                                b3 = sB[state_col + 3]
                                c0 = sC[state_col]
                                c1 = sC[state_col + 1]
                                c2 = sC[state_col + 2]
                                c3 = sC[state_col + 3]
                            else:
                                b0 = Float32(mB[bc_base + state_col])
                                b1 = Float32(mB[bc_base + state_col + 1])
                                b2 = Float32(mB[bc_base + state_col + 2])
                                b3 = Float32(mB[bc_base + state_col + 3])
                                c0 = Float32(mC[bc_base + state_col])
                                c1 = Float32(mC[bc_base + state_col + 1])
                                c2 = Float32(mC[bc_base + state_col + 2])
                                c3 = Float32(mC[bc_base + state_col + 3])
                            decay0 = cute.math.exp2(
                                a0 * dt * Float32(_LOG2E), fastmath=True
                            )
                            decay1 = cute.math.exp2(
                                a1 * dt * Float32(_LOG2E), fastmath=True
                            )
                            decay2 = cute.math.exp2(
                                a2 * dt * Float32(_LOG2E), fastmath=True
                            )
                            decay3 = cute.math.exp2(
                                a3 * dt * Float32(_LOG2E), fastmath=True
                            )
                            state0 = state0 * decay0 + b0 * dt * x
                            state1 = state1 * decay1 + b1 * dt * x
                            state2 = state2 * decay2 + b2 * dt * x
                            state3 = state3 * decay3 + b3 * dt * x
                            mStatePacked[packed_state_idx // 2] = bf16x2_to_i32(
                                state_dtype(state0), state_dtype(state1)
                            )
                            mStatePacked[packed_state_idx // 2 + 1] = bf16x2_to_i32(
                                state_dtype(state2), state_dtype(state3)
                            )
                            output += (
                                state0 * c0 + state1 * c1 + state2 * c2 + state3 * c3
                            )
                    else:
                        for item in cutlass.range_constexpr(items_per_lane):
                            state_col = state_lane + item * lanes_per_feature
                            if state_col < dstate:
                                scalar_state_idx = state_base + state_col
                                state = Float32(mState[scalar_state_idx])
                                a = Float32(mA[param_idx * dstate + state_col])
                                if const_expr(dstate >= 64):
                                    b = sB[state_col]
                                    c = sC[state_col]
                                else:
                                    b = Float32(mB[bc_base + state_col])
                                    c = Float32(mC[bc_base + state_col])
                                decay = cute.math.exp2(
                                    a * dt * Float32(_LOG2E), fastmath=True
                                )
                                state = state * decay + b * dt * x
                                mState[scalar_state_idx] = state_dtype(state)
                                output += state * c

                    for offset in reduction_offsets:
                        output += cute.arch.shuffle_sync_bfly(output, offset)
                    if state_lane == 0:
                        if const_expr(has_d):
                            output += x * Float32(mD[param_idx])
                        if const_expr(has_z):
                            output *= silu(Float32(mZ[token_idx]))
                        mOut[token_idx] = x_dtype(output)

    def fake(dtype, align=16):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=align
            )
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


def _selective_state_update_cute(
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
        raise ValueError(
            "CuTe selective_state_update supports dstate 16, 32, 64, or 128"
        )
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
        torch2cute_dtype_map[state_batch_indices.dtype]
        if state_batch_indices is not None
        else None,
        dstate,
        has_d=D is not None,
        has_z=z is not None,
        has_dt_bias=dt_bias is not None,
        has_state_batch_indices=state_batch_indices is not None,
        dt_softplus=dt_softplus,
        pack_bf16_state=state.dtype == torch.bfloat16,
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
    """Run the faster CuTe route and preserve the Triton API elsewhere."""
    use_cute = (
        _is_hopper_or_newer(state.device.index)
        and state.dtype == torch.bfloat16
        and state.shape[-1] == 128
        and x.shape[0] >= 16
        and x[0].numel() >= 4096
        and D is not None
        and z is not None
        and dt_bias is not None
        and dt_softplus
    )
    if use_cute:
        return _selective_state_update_cute(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            state_batch_indices,
        )

    from mamba_ssm.ops.triton.selective_state_update import selective_state_update

    return selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
    )


__all__ = ["selective_state_update_cute"]
