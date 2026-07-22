from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce


_LOG2E = 1.4426950408889634
_NUM_THREADS = 256


@cache
def _compile_state_passing_fwd(
    state_dtype,
    decay_dtype,
    initial_dtype,
    output_dtype,
    index_dtype,
    vec_size: int,
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
            feature_blocks = cute.ceil_div(dim, vec_size)
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
                feature = feature_block * vec_size
                values = [Float32(0.0) for _ in range(vec_size)]
                initial_base = (batch_idx * nheads + head) * dim + feature
                if const_expr(has_initial):
                    for item in cutlass.range_constexpr(vec_size):
                        if feature + item < dim:
                            values[item] = Float32(mInitial[initial_base + item])

                seq = Int32(0)
                for chunk in cutlass.range(0, nchunks, 1, unroll=1):
                    state_base = (
                        (batch_idx * nchunks + chunk) * nheads + head
                    ) * dim + feature
                    for item in cutlass.range_constexpr(vec_size):
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
                    for item in cutlass.range_constexpr(vec_size):
                        if feature + item < dim:
                            values[item] = scale * values[item] + Float32(
                                mStates[state_base + item]
                            )

                for item in cutlass.range_constexpr(vec_size):
                    if feature + item < dim:
                        mFinal[initial_base + item] = values[item]

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
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
    out = torch.empty(
        (batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype
    )
    final = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    compiled = _compile_state_passing_fwd(
        torch2cute_dtype_map[states.dtype],
        torch2cute_dtype_map[dA_chunk_cumsum.dtype],
        torch2cute_dtype_map[initial_states.dtype]
        if initial_states is not None
        else None,
        torch2cute_dtype_map[out_dtype],
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        4 if states.dtype == torch.float32 else 8,
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


@cache
def _compile_state_passing_bwd(
    state_dtype,
    decay_dtype,
    dout_dtype,
    final_dtype,
    dstates_dtype,
    converted_dtype,
    index_dtype,
    *,
    has_final: bool,
    has_initial: bool,
    convert_states: bool,
    has_seq_idx: bool,
):
    class StatePassingBwd:
        @cute.jit
        def __call__(
            self,
            mStates: cute.Tensor,
            mDecay: cute.Tensor,
            mDOut: cute.Tensor,
            mDFinal: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mDStates: cute.Tensor,
            mDDecayPartial: cute.Tensor,
            mDInitial: cute.Tensor | None,
            mStatesConverted: cute.Tensor | None,
            batch: Int32,
            nchunks: Int32,
            nheads: Int32,
            dim: Int32,
            seqlen: Int32,
            chunk_size: Int32,
            stream: cuda.CUstream,
        ):
            nblocks = cute.ceil_div(dim, _NUM_THREADS)
            self.kernel(
                mStates,
                mDecay,
                mDOut,
                mDFinal,
                mSeqIdx,
                mDStates,
                mDDecayPartial,
                mDInitial,
                mStatesConverted,
                nchunks,
                nheads,
                dim,
                seqlen,
                chunk_size,
                nblocks,
            ).launch(
                grid=[nblocks, batch, nheads],
                block=[_NUM_THREADS, 1, 1],
                smem=8 * 4,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mStates: cute.Tensor,
            mDecay: cute.Tensor,
            mDOut: cute.Tensor,
            mDFinal: cute.Tensor | None,
            mSeqIdx: cute.Tensor | None,
            mDStates: cute.Tensor,
            mDDecayPartial: cute.Tensor,
            mDInitial: cute.Tensor | None,
            mStatesConverted: cute.Tensor | None,
            nchunks: Int32,
            nheads: Int32,
            dim: Int32,
            seqlen: Int32,
            chunk_size: Int32,
            nblocks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            feature_block, batch_idx, head = cute.arch.block_idx()
            feature = feature_block * _NUM_THREADS + tidx
            outer_base = (Int64(batch_idx) * nheads + head) * dim
            dstates = Float32(0.0)
            if const_expr(has_final):
                if feature < dim:
                    dstates = Float32(mDFinal[outer_base + feature])

            smem = cutlass.utils.SmemAllocator()
            reduction_buffer = smem.allocate_tensor(
                Float32, cute.make_layout((1, (8, 1))), byte_alignment=16
            )
            seq = Int32(0)
            if const_expr(has_seq_idx):
                seq = Int32(mSeqIdx[Int64(batch_idx) * seqlen + seqlen - 1])

            for reverse_idx in cutlass.range(0, nchunks, 1, unroll=1):
                chunk = nchunks - 1 - reverse_idx
                state_idx = (
                    (Int64(batch_idx) * nchunks + chunk) * nheads + head
                ) * dim + feature
                if feature < dim:
                    mDStates[state_idx] = dstates_dtype(dstates)
                    if const_expr(convert_states):
                        mStatesConverted[state_idx] = converted_dtype(
                            Float32(mStates[state_idx])
                        )

                if chunk > 0:
                    scale = cute.math.exp2(
                        Float32(
                            mDecay[(Int64(batch_idx) * nheads + head) * nchunks + chunk]
                        )
                        * Float32(_LOG2E),
                        fastmath=True,
                    )
                    if const_expr(has_seq_idx):
                        token = chunk * chunk_size - 1
                        seq_new = Int32(mSeqIdx[Int64(batch_idx) * seqlen + token])
                        if seq_new != seq:
                            scale = Float32(0.0)
                        seq = seq_new
                    partial = Float32(0.0)
                    if feature < dim:
                        partial = Float32(mStates[state_idx]) * dstates * scale
                    total = row_reduce(
                        partial,
                        cute.ReductionOp.ADD,
                        _NUM_THREADS,
                        reduction_buffer=reduction_buffer,
                    )
                    if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
                        partial_idx = (
                            (Int64(batch_idx) * nheads + head) * nchunks + chunk
                        ) * nblocks + feature_block
                        mDDecayPartial[partial_idx] = total
                    if feature < dim:
                        dstates = scale * dstates + Float32(mDOut[state_idx])

            if const_expr(has_initial):
                scale = cute.math.exp2(
                    Float32(mDecay[(Int64(batch_idx) * nheads + head) * nchunks])
                    * Float32(_LOG2E),
                    fastmath=True,
                )
                if const_expr(has_seq_idx):
                    if seq != Int32(0):
                        scale = Float32(0.0)
                state0_idx = (
                    (Int64(batch_idx) * nchunks) * nheads + head
                ) * dim + feature
                partial = Float32(0.0)
                if feature < dim:
                    partial = Float32(mStates[state0_idx]) * dstates * scale
                total = row_reduce(
                    partial,
                    cute.ReductionOp.ADD,
                    _NUM_THREADS,
                    reduction_buffer=reduction_buffer,
                )
                if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
                    partial_idx = (
                        (Int64(batch_idx) * nheads + head) * nchunks
                    ) * nblocks + feature_block
                    mDDecayPartial[partial_idx] = total
                if feature < dim:
                    dstates = scale * dstates + Float32(mDOut[state0_idx])
                    mDInitial[outer_base + feature] = dstates_dtype(dstates)
            else:
                if cute.arch.warp_idx() == 0 and cute.arch.lane_idx() == 0:
                    partial_idx = (
                        (Int64(batch_idx) * nheads + head) * nchunks
                    ) * nblocks + feature_block
                    mDDecayPartial[partial_idx] = Float32(0.0)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        StatePassingBwd(),
        fake(state_dtype),
        fake(decay_dtype),
        fake(dout_dtype),
        fake(final_dtype) if has_final else None,
        fake(index_dtype) if has_seq_idx else None,
        fake(dstates_dtype),
        fake(Float32),
        fake(dstates_dtype) if has_initial else None,
        fake(converted_dtype) if convert_states else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def state_passing_bwd_cute(
    states,
    dA_chunk_cumsum,
    dout,
    dfinal_states=None,
    seq_idx=None,
    has_initial_states=None,
    dstates_dtype=None,
    states_dtype=None,
    chunk_size=None,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    assert dout.shape == states.shape
    if seq_idx is not None and chunk_size is None:
        raise ValueError("chunk_size is required with seq_idx")
    dstates_dtype = dout.dtype if dstates_dtype is None else dstates_dtype
    dstates = torch.empty_like(dout, dtype=dstates_dtype)
    dinitial = torch.empty_like(dstates[:, 0]) if has_initial_states else None
    converted = (
        torch.empty_like(states, dtype=dstates_dtype)
        if states_dtype is not None and states_dtype != states.dtype
        else None
    )
    nblocks = (dim + _NUM_THREADS - 1) // _NUM_THREADS
    partial = torch.empty(
        batch, nheads, nchunks, nblocks, device=states.device, dtype=torch.float32
    )
    compiled = _compile_state_passing_bwd(
        torch2cute_dtype_map[states.dtype],
        torch2cute_dtype_map[dA_chunk_cumsum.dtype],
        torch2cute_dtype_map[dout.dtype],
        torch2cute_dtype_map[dfinal_states.dtype]
        if dfinal_states is not None
        else None,
        torch2cute_dtype_map[dstates_dtype],
        torch2cute_dtype_map[dstates_dtype] if converted is not None else None,
        torch2cute_dtype_map[seq_idx.dtype] if seq_idx is not None else None,
        has_final=dfinal_states is not None,
        has_initial=bool(has_initial_states),
        convert_states=converted is not None,
        has_seq_idx=seq_idx is not None,
    )
    compiled(
        states.view(-1),
        dA_chunk_cumsum.view(-1),
        dout.view(-1),
        dfinal_states.view(-1) if dfinal_states is not None else None,
        seq_idx.view(-1) if seq_idx is not None else None,
        dstates.view(-1),
        partial.view(-1),
        dinitial.view(-1) if dinitial is not None else None,
        converted.view(-1) if converted is not None else None,
        batch,
        nchunks,
        nheads,
        dim,
        seq_idx.shape[1] if seq_idx is not None else 0,
        chunk_size if seq_idx is not None else 0,
    )
    ddecay = partial.sum(dim=-1).to(dA_chunk_cumsum.dtype)
    if states_dtype is not None and states_dtype == states.dtype:
        converted = states
    result = (dstates, ddecay, dinitial)
    return result if states_dtype is None else (*result, converted)


class StatePassingCuteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states, dA_chunk_cumsum, initial_states=None):
        if states.stride(-1) != 1:
            states = states.contiguous()
        out, final = state_passing_fwd_cute(states, dA_chunk_cumsum, initial_states)
        ctx.save_for_backward(out, dA_chunk_cumsum)
        ctx.has_initial_states = initial_states is not None
        return out, final

    @staticmethod
    def backward(ctx, dout, dfinal_states):
        out, dA_chunk_cumsum = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dstates, ddecay, dinitial = state_passing_bwd_cute(
            out,
            dA_chunk_cumsum,
            dout,
            dfinal_states=dfinal_states,
            has_initial_states=ctx.has_initial_states,
        )
        return dstates, ddecay, dinitial


def state_passing_cute(states, dA_chunk_cumsum, initial_states=None):
    return StatePassingCuteFn.apply(states, dA_chunk_cumsum, initial_states)


__all__ = [
    "StatePassingCuteFn",
    "state_passing_bwd_cute",
    "state_passing_cute",
    "state_passing_fwd_cute",
]
