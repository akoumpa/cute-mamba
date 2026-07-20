from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map


_LOG2E = 1.4426950408889634
_NUM_THREADS = 256
_VEC_SIZE = 8


@cache
def _compile_state_passing_fwd(
    state_dtype,
    decay_dtype,
    initial_dtype,
    output_dtype,
    index_dtype,
    *,
    has_initial: bool,
    has_seq_idx: bool,
):
    class StatePassingFwd:
        @cute.jit
        def __call__(
            self,
            mStates: cute.Tensor,
            mDecay: cute.Tensor,
            mInitial: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mOut: cute.Tensor,
            mFinal: cute.Tensor,
            batch: Int32,
            nchunks: Int32,
            nheads: Int32,
            dim: Int32,
            seqlen: Int32,
            chunk_size: Int32,
            stream: cuda.CUstream,
        ):
            feature_blocks = cute.ceil_div(dim, _VEC_SIZE)
            work = Int64(batch) * nheads * feature_blocks
            self.kernel(
                mStates,
                mDecay,
                mInitial,
                mSeqIdx,
                mOut,
                mFinal,
                work,
                nchunks,
                nheads,
                dim,
                seqlen,
                chunk_size,
                feature_blocks,
            ).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mStates: cute.Tensor,
            mDecay: cute.Tensor,
            mInitial: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mOut: cute.Tensor,
            mFinal: cute.Tensor,
            work: Int64,
            nchunks: Int32,
            nheads: Int32,
            dim: Int32,
            seqlen: Int32,
            chunk_size: Int32,
            feature_blocks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < work:
                feature_block = linear % feature_blocks
                outer = linear // feature_blocks
                head = outer % nheads
                batch_idx = outer // nheads
                feature = feature_block * _VEC_SIZE
                values = [Float32(0.0) for _ in range(_VEC_SIZE)]
                initial_base = (batch_idx * nheads + head) * dim + feature
                if const_expr(has_initial):
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            values[item] = Float32(mInitial[initial_base + item])

                seq = Int32(0)
                for chunk in cutlass.range(0, nchunks, 1, unroll=1):
                    state_base = ((batch_idx * nchunks + chunk) * nheads + head) * dim + feature
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            mOut[state_base + item] = output_dtype(values[item])

                    scale = cute.math.exp2(
                        Float32(mDecay[(batch_idx * nheads + head) * nchunks + chunk])
                        * Float32(_LOG2E),
                        fastmath=True,
                    )
                    if const_expr(has_seq_idx):
                        token = min((chunk + 1) * chunk_size, seqlen) - 1
                        seq_new = Int32(mSeqIdx[batch_idx * seqlen + token])
                        if seq_new != seq:
                            scale = Float32(0.0)
                        seq = seq_new
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            values[item] = (
                                scale * values[item] + Float32(mStates[state_base + item])
                            )

                for item in cutlass.range_constexpr(_VEC_SIZE):
                    if feature + item < dim:
                        mFinal[initial_base + item] = values[item]

    fake = lambda dtype: (
        cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,), assumed_align=16)
        if dtype is not None
        else None
    )
    return cute.compile(
        StatePassingFwd(),
        fake(state_dtype),
        fake(decay_dtype),
        fake(initial_dtype) if has_initial else None,
        fake(index_dtype) if has_seq_idx else None,
        fake(output_dtype),
        fake(Float32),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def state_passing_fwd_cute(
    states,
    dA_chunk_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    if seq_idx is not None:
        if chunk_size is None:
            raise ValueError("chunk_size is required with seq_idx")
        assert seq_idx.shape[0] == batch
    tensors = (states, dA_chunk_cumsum, initial_states, seq_idx)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        raise ValueError("CuTe state passing requires contiguous CUDA tensors")

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    compiled = _compile_state_passing_fwd(
        torch2cute_dtype_map[states.dtype],
        torch2cute_dtype_map[dA_chunk_cumsum.dtype],
        torch2cute_dtype_map[initial_states.dtype] if initial_states is not None else None,
        torch2cute_dtype_map[out_dtype],
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_initial=initial_states is not None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        states.view(-1),
        dA_chunk_cumsum.view(-1),
        initial_states.view(-1) if initial_states is not None else None,
        seq_idx.view(-1) if seq_idx is not None else None,
        out.view(-1),
        final.view(-1),
        batch,
        nchunks,
        nheads,
        dim,
        seq_idx.shape[1] if seq_idx is not None else 0,
        chunk_size if seq_idx is not None else 0,
    )
    return out, final


__all__ = ["state_passing_fwd_cute"]
