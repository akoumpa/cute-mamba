from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.utils import cos_approx, sin_approx


@cache
def _compile_dk_state_post(
    state_dtype,
    angle_dtype,
    k_dtype,
    bias_dtype,
    dk_dtype,
    dk_bias_dtype,
    dangle_dtype,
    index_dtype,
    *,
    qk_dim: int,
    angle_dim: int,
    is_varlen: bool,
):
    num_threads = max(32, min(256, qk_dim))

    class DkStatePost:
        @cute.jit
        def __call__(
            self,
            mState: cute.Tensor,
            mAngles: cute.Tensor,
            mK: cute.Tensor,
            mKBias: cute.Tensor,
            mDk: cute.Tensor,
            mDkBias: cute.Tensor,
            mDangles: cute.Tensor,
            mCuSeqlens: cute.Tensor | None,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            nheads_qk: Int32,
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mState,
                mAngles,
                mK,
                mKBias,
                mDk,
                mDkBias,
                mDangles,
                mCuSeqlens,
                seqlen,
                nheads,
                nheads_qk,
            ).launch(
                grid=[nheads, num_sequences, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mState: cute.Tensor,
            mAngles: cute.Tensor,
            mK: cute.Tensor,
            mKBias: cute.Tensor,
            mDk: cute.Tensor,
            mDkBias: cute.Tensor,
            mDangles: cute.Tensor,
            mCuSeqlens: cute.Tensor | None,
            seqlen: Int32,
            nheads: Int32,
            nheads_qk: Int32,
        ):
            feature, _, _ = cute.arch.thread_idx()
            head, sequence, _ = cute.arch.block_idx()
            batch_idx = sequence
            last_pos = seqlen - 1
            if const_expr(is_varlen):
                batch_idx = Int32(0)
                last_pos = Int32(mCuSeqlens[sequence + 1]) - 1
            qk_head = head // (nheads // nheads_qk)
            state_base = (Int64(sequence) * nheads + head) * qk_dim
            pair = feature // 2
            angle = Float32(0.0)
            if pair < angle_dim:
                angle_idx = (
                    (Int64(batch_idx) * seqlen + last_pos) * nheads + head
                ) * angle_dim + pair
                angle = Float32(mAngles[angle_idx])
            cosine = cos_approx(angle)
            sine = sin_approx(angle)

            if feature < qk_dim:
                pair_feature = feature + 1 if feature % 2 == 0 else feature - 1
                state = Float32(mState[state_base + feature])
                pair_state = Float32(mState[state_base + pair_feature])
                rotated = state * cosine + pair_state * sine
                if feature % 2 != 0:
                    rotated = state * cosine - pair_state * sine
                dk_idx = (
                    (Int64(batch_idx) * seqlen + last_pos) * nheads_qk + qk_head
                ) * qk_dim + feature
                cute.arch.atomic_add(ptr=mDk.iterator + dk_idx, val=rotated)
                bias_idx = Int64(head) * qk_dim + feature
                cute.arch.atomic_add(ptr=mDkBias.iterator + bias_idx, val=rotated)

            if feature < angle_dim:
                first = feature * 2
                second = first + 1
                dangle_idx = (
                    (Int64(batch_idx) * seqlen + last_pos) * nheads + head
                ) * angle_dim + feature
                dangle = Float32(mAngles[dangle_idx])
                dangle_cosine = cos_approx(dangle)
                dangle_sine = sin_approx(dangle)
                state0 = Float32(mState[state_base + first])
                state1 = Float32(mState[state_base + second])
                k_base = (
                    (Int64(batch_idx) * seqlen + last_pos) * nheads_qk + qk_head
                ) * qk_dim
                bias_base = Int64(head) * qk_dim
                k0 = Float32(mK[k_base + first]) + Float32(mKBias[bias_base + first])
                k1 = Float32(mK[k_base + second]) + Float32(mKBias[bias_base + second])
                dtheta = state0 * (-k0 * dangle_sine - k1 * dangle_cosine)
                dtheta += state1 * (k0 * dangle_cosine - k1 * dangle_sine)
                mDangles[dangle_idx] = dangle_dtype(
                    Float32(mDangles[dangle_idx]) + dtheta
                )

    def fake(dtype, align=16):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=align
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        DkStatePost(),
        fake(state_dtype),
        fake(angle_dtype),
        fake(k_dtype),
        fake(bias_dtype),
        fake(dk_dtype),
        fake(dk_bias_dtype),
        fake(dangle_dtype),
        fake(index_dtype, 4) if is_varlen else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def apply_dk_state_post_cute_impl(
    d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, cu_seqlens=None
):
    batch, seqlen, nheads, angle_dim = angles.shape
    num_sequences = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
    qk_dim = d_ok_state.shape[-1]
    nheads_qk = k.shape[2]
    tensors = (d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, cu_seqlens)
    if any(tensor is not None and not tensor.is_cuda for tensor in tensors):
        raise AssertionError("Mamba3 state-post tensors must be CUDA tensors")
    d_ok_state, angles, k, k_bias = (
        tensor.contiguous() for tensor in (d_ok_state, angles, k, k_bias)
    )
    compiled = _compile_dk_state_post(
        torch2cute_dtype_map[d_ok_state.dtype],
        torch2cute_dtype_map[angles.dtype],
        torch2cute_dtype_map[k.dtype],
        torch2cute_dtype_map[k_bias.dtype],
        torch2cute_dtype_map[dk.dtype],
        torch2cute_dtype_map[dk_bias.dtype],
        torch2cute_dtype_map[dangles.dtype],
        torch2cute_dtype_map[cu_seqlens.dtype] if cu_seqlens is not None else None,
        qk_dim=qk_dim,
        angle_dim=angle_dim,
        is_varlen=cu_seqlens is not None,
    )
    compiled(
        d_ok_state.view(-1),
        angles.view(-1),
        k.view(-1),
        k_bias.view(-1),
        dk.view(-1),
        dk_bias.view(-1),
        dangles.view(-1),
        cu_seqlens.view(-1) if cu_seqlens is not None else None,
        batch,
        seqlen,
        nheads,
        nheads_qk,
        num_sequences,
    )


__all__ = ["apply_dk_state_post_cute_impl"]
