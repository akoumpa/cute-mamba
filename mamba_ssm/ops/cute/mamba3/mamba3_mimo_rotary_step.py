from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.utils import cos_approx, sin_approx, tanh_approx


_NUM_THREADS = 256
_PI = 3.141592653589793


@cache
def _compile_rotary_qk_step(
    q_dtype,
    k_dtype,
    angle_dtype,
    proj_dtype,
    dt_dtype,
    bias_q_dtype,
    bias_k_dtype,
    *,
    has_bias_q: bool,
    has_bias_k: bool,
    conjugate: bool,
    rotate_pairwise: bool,
):
    class RotaryQKStep:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mAngleState: cute.Tensor,
            mAngleProj: cute.Tensor,
            mDt: cute.Tensor,
            mBiasQ: cute.Tensor | None,
            mBiasK: cute.Tensor | None,
            mOutQ: cute.Tensor,
            mOutK: cute.Tensor,
            mOutAngle: cute.Tensor,
            count: Int64,
            mimo_dim: Int32,
            nheads: Int32,
            headdim: Int32,
            rotary_half: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mQ,
                mK,
                mAngleState,
                mAngleProj,
                mDt,
                mBiasQ,
                mBiasK,
                mOutQ,
                mOutK,
                mOutAngle,
                count,
                mimo_dim,
                nheads,
                headdim,
                rotary_half,
            ).launch(
                grid=[cute.ceil_div(count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mAngleState: cute.Tensor,
            mAngleProj: cute.Tensor,
            mDt: cute.Tensor,
            mBiasQ: cute.Tensor | None,
            mBiasK: cute.Tensor | None,
            mOutQ: cute.Tensor,
            mOutK: cute.Tensor,
            mOutAngle: cute.Tensor,
            count: Int64,
            mimo_dim: Int32,
            nheads: Int32,
            headdim: Int32,
            rotary_half: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < count:
                feature = Int32(linear % headdim)
                outer = linear // headdim
                head = Int32(outer % nheads)
                outer = outer // nheads
                mimo = Int32(outer % mimo_dim)
                batch_idx = Int32(outer // mimo_dim)
                bias_idx = (Int64(mimo) * nheads + head) * headdim + feature
                q = Float32(mQ[linear])
                k = Float32(mK[linear])
                if const_expr(has_bias_q):
                    q += Float32(mBiasQ[bias_idx])
                if const_expr(has_bias_k):
                    k += Float32(mBiasK[bias_idx])

                angle_feature = Int32(0)
                counterpart = feature
                rotate = False
                lower = False
                if const_expr(rotate_pairwise):
                    angle_feature = feature // 2
                    rotate = feature < rotary_half * 2
                    lower = feature % 2 == 0
                    counterpart = feature + 1 if lower else feature - 1
                else:
                    half = headdim // 2
                    lower = feature < half
                    angle_feature = feature if lower else feature - half
                    rotate = angle_feature < rotary_half
                    counterpart = feature + half if lower else feature - half

                if rotate:
                    pair_idx = (
                        (Int64(batch_idx) * mimo_dim + mimo) * nheads + head
                    ) * headdim + counterpart
                    pair_bias_idx = (
                        Int64(mimo) * nheads + head
                    ) * headdim + counterpart
                    q_pair = Float32(mQ[pair_idx])
                    k_pair = Float32(mK[pair_idx])
                    if const_expr(has_bias_q):
                        q_pair += Float32(mBiasQ[pair_bias_idx])
                    if const_expr(has_bias_k):
                        k_pair += Float32(mBiasK[pair_bias_idx])
                    angle_idx = (
                        Int64(batch_idx) * nheads + head
                    ) * rotary_half + angle_feature
                    angle = Float32(mAngleState[angle_idx]) + tanh_approx(
                        Float32(mAngleProj[angle_idx])
                    ) * Float32(mDt[Int64(batch_idx) * nheads + head]) * Float32(_PI)
                    cosine = cos_approx(angle)
                    sine = sin_approx(angle)
                    if const_expr(conjugate):
                        sine = -sine
                    if lower:
                        q = q * cosine - q_pair * sine
                        k = k * cosine - k_pair * sine
                    else:
                        q = q * cosine + q_pair * sine
                        k = k * cosine + k_pair * sine
                    if mimo == 0 and lower:
                        mOutAngle[angle_idx] = angle_dtype(angle)

                mOutQ[linear] = q_dtype(q)
                mOutK[linear] = k_dtype(k)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        RotaryQKStep(),
        fake(q_dtype),
        fake(k_dtype),
        fake(angle_dtype),
        fake(proj_dtype),
        fake(dt_dtype),
        fake(bias_q_dtype) if has_bias_q else None,
        fake(bias_k_dtype) if has_bias_k else None,
        fake(q_dtype),
        fake(k_dtype),
        fake(angle_dtype),
        Int64(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def apply_rotary_qk_inference_fwd_cute(
    q,
    k,
    angle_state,
    angle_proj,
    dt,
    bias_q=None,
    bias_k=None,
    inplace=False,
    conjugate=False,
    rotate_pairwise=True,
):
    batch, mimo_dim, nheads, headdim = q.shape
    rotary_half = angle_state.shape[-1]
    if k.shape != q.shape or headdim % 2 != 0 or headdim > 256:
        raise AssertionError("invalid Q/K rotary shape")
    if angle_state.shape != (batch, nheads, rotary_half):
        raise AssertionError("invalid angle state shape")
    if angle_proj.shape != angle_state.shape or dt.shape != (batch, nheads):
        raise AssertionError("invalid angle projection or dt shape")
    if rotary_half * 2 > headdim:
        raise AssertionError("rotary_dim must be <= headdim")
    if bias_q is not None and bias_q.shape != (mimo_dim, nheads, headdim):
        raise AssertionError("invalid Q bias shape")
    if bias_k is not None and bias_k.shape != (mimo_dim, nheads, headdim):
        raise AssertionError("invalid K bias shape")
    tensors = (q, k, angle_state, angle_proj, dt, bias_q, bias_k)
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
            apply_rotary_qk_inference_fwd,
        )

        return apply_rotary_qk_inference_fwd(
            q,
            k,
            angle_state,
            angle_proj,
            dt,
            bias_q,
            bias_k,
            inplace,
            conjugate,
            rotate_pairwise,
        )

    q_input = q.contiguous()
    k_input = k.contiguous()
    angle_input = angle_state.contiguous()
    proj_input = angle_proj.contiguous()
    dt_input = dt.contiguous()
    bias_q_input = bias_q.contiguous() if bias_q is not None else None
    bias_k_input = bias_k.contiguous() if bias_k is not None else None
    out_q = torch.empty_like(q_input)
    out_k = torch.empty_like(k_input)
    out_angle = torch.empty_like(angle_input)
    compiled = _compile_rotary_qk_step(
        torch2cute_dtype_map[q_input.dtype],
        torch2cute_dtype_map[k_input.dtype],
        torch2cute_dtype_map[angle_input.dtype],
        torch2cute_dtype_map[proj_input.dtype],
        torch2cute_dtype_map[dt_input.dtype],
        torch2cute_dtype_map[bias_q_input.dtype] if bias_q_input is not None else None,
        torch2cute_dtype_map[bias_k_input.dtype] if bias_k_input is not None else None,
        has_bias_q=bias_q_input is not None,
        has_bias_k=bias_k_input is not None,
        conjugate=conjugate,
        rotate_pairwise=rotate_pairwise,
    )
    compiled(
        q_input.view(-1),
        k_input.view(-1),
        angle_input.view(-1),
        proj_input.view(-1),
        dt_input.view(-1),
        bias_q_input.view(-1) if bias_q_input is not None else None,
        bias_k_input.view(-1) if bias_k_input is not None else None,
        out_q.view(-1),
        out_k.view(-1),
        out_angle.view(-1),
        q_input.numel(),
        mimo_dim,
        nheads,
        headdim,
        rotary_half,
    )
    if inplace:
        q.copy_(out_q)
        k.copy_(out_k)
        angle_state.copy_(out_angle)
        return q, k, angle_state
    return out_q, out_k, out_angle


__all__ = ["apply_rotary_qk_inference_fwd_cute"]
