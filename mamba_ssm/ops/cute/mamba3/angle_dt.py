from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.utils import tanh_approx


_NUM_THREADS = 256
_VEC_SIZE = 4
_PI = 3.141592653589793
_TWO_PI = 2.0 * _PI


@cache
def _compile_angle_dt_fwd(
    angle_dtype,
    dt_dtype,
    initial_dtype,
    index_dtype,
    *,
    has_initial: bool,
    return_state: bool,
    is_varlen: bool,
):
    class AngleDtFwd:
        @cute.jit
        def __call__(
            self,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mInitial: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mOut: cute.Tensor,
            mFinal: cute.Tensor | None,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            dim: Int32,
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            feature_blocks = cute.ceil_div(dim, _VEC_SIZE)
            sequences = num_sequences if const_expr(is_varlen) else batch
            work = Int64(sequences) * nheads * feature_blocks
            self.kernel(
                mAngle,
                mDt,
                mInitial,
                mCuSeqlens,
                mOut,
                mFinal,
                work,
                seqlen,
                nheads,
                dim,
                feature_blocks,
            ).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mInitial: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mOut: cute.Tensor,
            mFinal: cute.Tensor | None,
            work: Int64,
            seqlen: Int32,
            nheads: Int32,
            dim: Int32,
            feature_blocks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < work:
                feature_block = linear % feature_blocks
                outer = linear // feature_blocks
                head = outer % nheads
                sequence = outer // nheads
                feature = feature_block * _VEC_SIZE
                begin = Int32(0)
                end = seqlen
                batch_idx = sequence
                if const_expr(is_varlen):
                    begin = Int32(mCuSeqlens[sequence])
                    end = Int32(mCuSeqlens[sequence + 1])
                    batch_idx = Int32(0)

                states = [Float32(0.0) for _ in range(_VEC_SIZE)]
                final_base = (Int64(sequence) * nheads + head) * dim + feature
                if const_expr(has_initial):
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            states[item] = Float32(mInitial[final_base + item])

                for token in cutlass.range(begin, end, 1, unroll=1):
                    dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                    if const_expr(is_varlen):
                        dt_idx = Int64(head) * seqlen + token
                    dt_value = Float32(mDt[dt_idx])
                    angle_base = (
                        (Int64(batch_idx) * seqlen + token) * nheads + head
                    ) * dim + feature
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            angle = tanh_approx(
                                Float32(mAngle[angle_base + item])
                            ) * Float32(_PI)
                            value = states[item] + angle * dt_value
                            quotient = value / Float32(_TWO_PI)
                            turns = Int32(quotient)
                            if Float32(turns) > quotient:
                                turns -= Int32(1)
                            value -= Float32(_TWO_PI) * Float32(turns)
                            states[item] = value
                            mOut[angle_base + item] = angle_dtype(value)

                if const_expr(return_state):
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            mFinal[final_base + item] = angle_dtype(states[item])

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        AngleDtFwd(),
        fake(angle_dtype),
        fake(dt_dtype),
        fake(initial_dtype) if has_initial else None,
        fake(index_dtype) if is_varlen else None,
        fake(angle_dtype),
        fake(angle_dtype) if return_state else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def angle_dt_fwd_cute(
    angle,
    dt,
    init_state=None,
    chunk_size=64,
    return_output_state=False,
    cu_seqlens=None,
):
    del chunk_size
    batch, seqlen, nheads, dim = angle.shape
    assert dt.shape == (batch, nheads, seqlen)
    is_varlen = cu_seqlens is not None
    num_sequences = cu_seqlens.shape[0] - 1 if is_varlen else batch
    if is_varlen and batch != 1:
        raise ValueError("Varlen mode requires batch=1")
    if init_state is not None:
        assert init_state.shape == (num_sequences, nheads, dim)
    tensors = (angle, dt, init_state, cu_seqlens)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        raise ValueError("CuTe angle_dt requires contiguous CUDA tensors")

    out = torch.empty_like(angle)
    final = (
        torch.empty(num_sequences, nheads, dim, device=angle.device, dtype=angle.dtype)
        if return_output_state
        else None
    )
    compiled = _compile_angle_dt_fwd(
        torch2cute_dtype_map[angle.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[init_state.dtype] if init_state is not None else None,
        torch2cute_dtype_map[cu_seqlens.dtype] if is_varlen else None,
        has_initial=init_state is not None,
        return_state=return_output_state,
        is_varlen=is_varlen,
    )
    compiled(
        angle.view(-1),
        dt.view(-1),
        init_state.view(-1) if init_state is not None else None,
        cu_seqlens.view(-1) if is_varlen else None,
        out.view(-1),
        final.view(-1) if final is not None else None,
        batch,
        seqlen,
        nheads,
        dim,
        num_sequences,
    )
    return (out, final) if return_output_state else out


__all__ = ["angle_dt_fwd_cute"]
