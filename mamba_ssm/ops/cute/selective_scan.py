from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.activation import silu, softplus
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce


@cache
def _compile_selective_scan_inference(
    input_dtype,
    delta_dtype,
    a_dtype,
    b_dtype,
    c_dtype,
    d_dtype,
    z_dtype,
    bias_dtype,
    dstate: int,
    num_threads: int,
    b_rank: int,
    c_rank: int,
    *,
    has_d: bool,
    has_z: bool,
    has_bias: bool,
    delta_softplus: bool,
    return_last_state: bool,
):
    class SelectiveScanInference:
        @cute.jit
        def __call__(
            self,
            mU: cute.Tensor,
            mDelta: cute.Tensor,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mBias: cute.Tensor | None,
            mOut: cute.Tensor,
            mLast: cute.Tensor | None,
            batch: Int32,
            dim: Int32,
            seqlen: Int32,
            b_groups: Int32,
            c_groups: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mU,
                mDelta,
                mA,
                mB,
                mC,
                mD,
                mZ,
                mBias,
                mOut,
                mLast,
                dim,
                seqlen,
                b_groups,
                c_groups,
            ).launch(
                grid=[batch, dim, 1],
                block=[num_threads, 1, 1],
                smem=(num_threads // 32) * 4,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mU: cute.Tensor,
            mDelta: cute.Tensor,
            mA: cute.Tensor,
            mB: cute.Tensor,
            mC: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mBias: cute.Tensor | None,
            mOut: cute.Tensor,
            mLast: cute.Tensor | None,
            dim: Int32,
            seqlen: Int32,
            b_groups: Int32,
            c_groups: Int32,
        ):
            state, _, _ = cute.arch.thread_idx()
            batch_idx, channel, _ = cute.arch.block_idx()
            value = Float32(0.0)
            a = Float32(0.0)
            if state < dstate:
                a = Float32(mA[Int64(channel) * dstate + state])

            allocator = cutlass.utils.SmemAllocator()
            reduction = allocator.allocate_tensor(
                Float32,
                cute.make_layout((1, (num_threads // 32, 1))),
                byte_alignment=16,
            )
            group_b = channel // (dim // b_groups)
            group_c = channel // (dim // c_groups)
            for token in cutlass.range(0, seqlen, 1, unroll=1):
                token_idx = (Int64(batch_idx) * dim + channel) * seqlen + token
                u = Float32(mU[token_idx])
                dt = Float32(mDelta[token_idx])
                if const_expr(has_bias):
                    dt += Float32(mBias[channel])
                if const_expr(delta_softplus):
                    dt = softplus(dt)

                b = Float32(0.0)
                c = Float32(0.0)
                if state < dstate:
                    if const_expr(b_rank == 2):
                        b = Float32(mB[Int64(channel) * dstate + state])
                    elif const_expr(b_rank == 3):
                        b = Float32(
                            mB[(Int64(batch_idx) * dstate + state) * seqlen + token]
                        )
                    else:
                        b = Float32(
                            mB[
                                (
                                    (Int64(batch_idx) * b_groups + group_b) * dstate
                                    + state
                                )
                                * seqlen
                                + token
                            ]
                        )
                    if const_expr(c_rank == 2):
                        c = Float32(mC[Int64(channel) * dstate + state])
                    elif const_expr(c_rank == 3):
                        c = Float32(
                            mC[(Int64(batch_idx) * dstate + state) * seqlen + token]
                        )
                    else:
                        c = Float32(
                            mC[
                                (
                                    (Int64(batch_idx) * c_groups + group_c) * dstate
                                    + state
                                )
                                * seqlen
                                + token
                            ]
                        )
                    value = cute.math.exp(a * dt, fastmath=True) * value + dt * b * u

                output = row_reduce(
                    value * c,
                    cute.ReductionOp.ADD,
                    num_threads,
                    reduction_buffer=reduction,
                )
                if state == 0:
                    if const_expr(has_d):
                        output += Float32(mD[channel]) * u
                    if const_expr(has_z):
                        output *= silu(Float32(mZ[token_idx]))
                    mOut[token_idx] = input_dtype(output)

            if const_expr(return_last_state):
                if state < dstate:
                    last_idx = (Int64(batch_idx) * dim + channel) * dstate + state
                    mLast[last_idx] = a_dtype(value)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        SelectiveScanInference(),
        fake(input_dtype),
        fake(delta_dtype),
        fake(a_dtype),
        fake(b_dtype),
        fake(c_dtype),
        fake(d_dtype) if has_d else None,
        fake(z_dtype) if has_z else None,
        fake(bias_dtype) if has_bias else None,
        fake(input_dtype),
        fake(a_dtype) if return_last_state else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _can_use_selective_scan_inference(u, delta, A, B, C, D, z, delta_bias):
    tensors = (u, delta, A, B, C, D, z, delta_bias)
    return (
        not torch.is_grad_enabled()
        and u.is_cuda
        and not A.is_complex()
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and u.shape[-1] <= 16
        and A.shape[1] <= 256
        and B.ndim in (2, 3, 4)
        and C.ndim in (2, 3, 4)
        and all(t is None or (t.is_cuda and t.is_contiguous()) for t in tensors)
    )


def selective_scan_cute(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    if _can_use_selective_scan_inference(u, delta, A, B, C, D, z, delta_bias):
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        b_groups = B.shape[1] if B.ndim == 4 else dim
        c_groups = C.shape[1] if C.ndim == 4 else dim
        if dim % b_groups == 0 and dim % c_groups == 0:
            out = torch.empty_like(u)
            last = (
                torch.empty((batch, dim, dstate), device=u.device, dtype=A.dtype)
                if return_last_state
                else None
            )
            num_threads = max(32, 1 << (dstate - 1).bit_length())
            compiled = _compile_selective_scan_inference(
                torch2cute_dtype_map[u.dtype],
                torch2cute_dtype_map[delta.dtype],
                torch2cute_dtype_map[A.dtype],
                torch2cute_dtype_map[B.dtype],
                torch2cute_dtype_map[C.dtype],
                torch2cute_dtype_map[D.dtype] if D is not None else None,
                torch2cute_dtype_map[z.dtype] if z is not None else None,
                torch2cute_dtype_map[delta_bias.dtype]
                if delta_bias is not None
                else None,
                dstate,
                num_threads,
                B.ndim,
                C.ndim,
                has_d=D is not None,
                has_z=z is not None,
                has_bias=delta_bias is not None,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
            )
            compiled(
                u.view(-1),
                delta.view(-1),
                A.view(-1),
                B.view(-1),
                C.view(-1),
                D.view(-1) if D is not None else None,
                z.view(-1) if z is not None else None,
                delta_bias.view(-1) if delta_bias is not None else None,
                out.view(-1),
                last.view(-1) if last is not None else None,
                batch,
                dim,
                seqlen,
                b_groups,
                c_groups,
            )
            return (out, last) if return_last_state else out

    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    result = selective_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        return_last_state,
    )
    if not return_last_state:
        return result
    output, last_state = result
    return output, last_state.detach()


def mamba_inner_cute(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
    checkpoint_lvl=1,
    b_rms_weight=None,
    c_rms_weight=None,
    dt_rms_weight=None,
    b_c_dt_rms_eps=1e-6,
):
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn

    return mamba_inner_fn(
        xz,
        conv1d_weight,
        conv1d_bias,
        x_proj_weight,
        delta_proj_weight,
        out_proj_weight,
        out_proj_bias,
        A,
        B,
        C,
        D,
        delta_bias,
        B_proj_bias,
        C_proj_bias,
        delta_softplus,
        checkpoint_lvl,
        b_rms_weight,
        c_rms_weight,
        dt_rms_weight,
        b_c_dt_rms_eps,
    )


__all__ = ["mamba_inner_cute", "selective_scan_cute"]
