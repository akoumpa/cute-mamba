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


@cute.jit
def _wrap_angle(value: Float32) -> Float32:
    quotient = value / Float32(_TWO_PI)
    turns = Int32(quotient)
    if Float32(turns) > quotient:
        turns -= Int32(1)
    return value - Float32(_TWO_PI) * Float32(turns)


@cache
def _compile_angle_dt_fwd_chunked(
    angle_dtype,
    dt_dtype,
    initial_dtype,
    index_dtype,
    chunk_size: int,
    dim: int,
    *,
    has_initial: bool,
    return_state: bool,
    is_varlen: bool,
):
    subgroup_size = 8
    features_per_warp = 32 // subgroup_size
    num_warps = dim // features_per_warp
    num_threads = num_warps * 32
    items_per_thread = chunk_size * dim // num_threads
    scan_items = chunk_size // subgroup_size

    class AngleDtFwdChunked:
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
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            sequences = num_sequences if const_expr(is_varlen) else batch
            self.kernel(
                mAngle,
                mDt,
                mInitial,
                mCuSeqlens,
                mOut,
                mFinal,
                seqlen,
                nheads,
            ).launch(
                grid=[sequences, nheads, 1],
                block=[num_threads, 1, 1],
                smem=chunk_size * dim * 4 + chunk_size * 4,
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
            seqlen: Int32,
            nheads: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            sequence, head, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            feature_group = lane // subgroup_size
            token_lane = lane % subgroup_size
            feature = warp * features_per_warp + feature_group
            begin = Int32(0)
            end = seqlen
            batch_idx = sequence
            if const_expr(is_varlen):
                begin = Int32(mCuSeqlens[sequence])
                end = Int32(mCuSeqlens[sequence + 1])
                batch_idx = Int32(0)
            seq_len = end - begin

            smem = cutlass.utils.SmemAllocator()
            sValues = smem.allocate_tensor(
                Float32,
                cute.make_ordered_layout((chunk_size, dim), order=(1, 0)),
                byte_alignment=16,
            )
            sDt = smem.allocate_tensor(
                Float32, cute.make_layout((chunk_size,)), byte_alignment=16
            )

            carry = Float32(0.0)
            if const_expr(has_initial):
                carry = Float32(
                    mInitial[(Int64(sequence) * nheads + head) * dim + feature]
                )

            nchunks = cute.ceil_div(seq_len, chunk_size)
            token = Int32(0)
            angle_idx = Int64(0)
            dt_idx = Int64(0)
            out_idx = Int64(0)
            value = Float32(0.0)
            for chunk in cutlass.range(0, nchunks, 1, unroll=1):
                for item in cutlass.range_constexpr(items_per_thread):
                    linear = tidx + item * num_threads
                    pos = linear // dim
                    load_feature = linear - pos * dim
                    local_token = chunk * chunk_size + pos
                    if const_expr(is_varlen):
                        value = Float32(0.0)
                        if local_token < seq_len:
                            token = begin + local_token
                            angle_idx = (
                                (Int64(batch_idx) * seqlen + token) * nheads + head
                            ) * dim + load_feature
                            value = tanh_approx(Float32(mAngle[angle_idx])) * Float32(
                                _PI
                            )
                        sValues[pos, load_feature] = value
                    else:
                        token = local_token
                        angle_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * dim + load_feature
                        sValues[pos, load_feature] = tanh_approx(
                            Float32(mAngle[angle_idx])
                        ) * Float32(_PI)
                if tidx < chunk_size:
                    local_token = chunk * chunk_size + tidx
                    if const_expr(is_varlen):
                        value = Float32(0.0)
                        if local_token < seq_len:
                            token = begin + local_token
                            dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                            value = Float32(mDt[dt_idx])
                        sDt[tidx] = value
                    else:
                        token = local_token
                        dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                        sDt[tidx] = Float32(mDt[dt_idx])
                cute.arch.sync_threads()

                for item in cutlass.range_constexpr(scan_items):
                    pos = item * subgroup_size + token_lane
                    value = sValues[pos, feature] * sDt[pos]
                    for offset in (1, 2, 4):
                        addend = cute.arch.shuffle_sync_up(
                            value, offset, mask_and_clamp=0
                        )
                        if token_lane >= offset:
                            value += addend
                    value = _wrap_angle(value + carry)
                    sValues[pos, feature] = value
                    carry = cute.arch.shuffle_sync(
                        value, feature_group * subgroup_size + subgroup_size - 1
                    )
                cute.arch.sync_threads()

                for item in cutlass.range_constexpr(items_per_thread):
                    linear = tidx + item * num_threads
                    pos = linear // dim
                    store_feature = linear - pos * dim
                    local_token = chunk * chunk_size + pos
                    if const_expr(is_varlen):
                        if local_token < seq_len:
                            token = begin + local_token
                            out_idx = (
                                (Int64(batch_idx) * seqlen + token) * nheads + head
                            ) * dim + store_feature
                            mOut[out_idx] = angle_dtype(sValues[pos, store_feature])
                    else:
                        token = local_token
                        out_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * dim + store_feature
                        mOut[out_idx] = angle_dtype(sValues[pos, store_feature])
                cute.arch.sync_threads()

            if const_expr(return_state):
                if token_lane == 0:
                    final_idx = (Int64(sequence) * nheads + head) * dim + feature
                    mFinal[final_idx] = angle_dtype(carry)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        AngleDtFwdChunked(),
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
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


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
    if chunk_size == 64 and dim in (16, 32) and (is_varlen or seqlen % chunk_size == 0):
        compiled = _compile_angle_dt_fwd_chunked(
            torch2cute_dtype_map[angle.dtype],
            torch2cute_dtype_map[dt.dtype],
            torch2cute_dtype_map[init_state.dtype] if init_state is not None else None,
            torch2cute_dtype_map[cu_seqlens.dtype] if is_varlen else None,
            chunk_size,
            dim,
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
            num_sequences,
        )
        return (out, final) if return_output_state else out
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


@cache
def _compile_angle_dt_bwd_chunked(
    angle_dtype,
    dt_dtype,
    grad_dtype,
    final_dtype,
    index_dtype,
    chunk_size: int,
    dim: int,
    *,
    has_initial: bool,
    has_grad_final: bool,
    is_varlen: bool,
):
    subgroup_size = 8
    features_per_warp = 32 // subgroup_size
    num_warps = dim // features_per_warp
    num_threads = num_warps * 32
    items_per_thread = chunk_size * dim // num_threads
    scan_items = chunk_size // subgroup_size
    tokens_per_warp = chunk_size // num_warps

    class AngleDtBwdChunked:
        @cute.jit
        def __call__(
            self,
            mGradOut: cute.Tensor,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mGradFinal: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mGradAngle: cute.Tensor,
            mGradDt: cute.Tensor,
            mGradInitial: cute.Tensor | None,
            batch: Int32,
            seqlen: Int32,
            nheads: Int32,
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            sequences = num_sequences if const_expr(is_varlen) else batch
            self.kernel(
                mGradOut,
                mAngle,
                mDt,
                mGradFinal,
                mCuSeqlens,
                mGradAngle,
                mGradDt,
                mGradInitial,
                seqlen,
                nheads,
            ).launch(
                grid=[sequences, nheads, 1],
                block=[num_threads, 1, 1],
                smem=chunk_size * dim * 4 + chunk_size * 4,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mGradOut: cute.Tensor,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mGradFinal: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mGradAngle: cute.Tensor,
            mGradDt: cute.Tensor,
            mGradInitial: cute.Tensor | None,
            seqlen: Int32,
            nheads: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            sequence, head, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            feature_group = lane // subgroup_size
            token_lane = lane % subgroup_size
            feature = warp * features_per_warp + feature_group
            begin = Int32(0)
            end = seqlen
            batch_idx = sequence
            if const_expr(is_varlen):
                begin = Int32(mCuSeqlens[sequence])
                end = Int32(mCuSeqlens[sequence + 1])
                batch_idx = Int32(0)
            seq_len = end - begin

            smem = cutlass.utils.SmemAllocator()
            sValues = smem.allocate_tensor(
                Float32,
                cute.make_ordered_layout((chunk_size, dim), order=(1, 0)),
                byte_alignment=16,
            )
            sDt = smem.allocate_tensor(
                Float32, cute.make_layout((chunk_size,)), byte_alignment=16
            )

            carry = Float32(0.0)
            if const_expr(has_grad_final):
                carry = Float32(
                    mGradFinal[(Int64(sequence) * nheads + head) * dim + feature]
                )

            nchunks = cute.ceil_div(seq_len, chunk_size)
            token = Int32(0)
            grad_idx = Int64(0)
            dt_idx = Int64(0)
            angle_idx = Int64(0)
            value = Float32(0.0)
            product = Float32(0.0)
            angle = Float32(0.0)
            tanh_angle = Float32(0.0)
            grad_value = Float32(0.0)
            for reverse_chunk in cutlass.range(0, nchunks, 1, unroll=1):
                chunk = nchunks - 1 - reverse_chunk
                for item in cutlass.range_constexpr(items_per_thread):
                    linear = tidx + item * num_threads
                    pos = linear // dim
                    load_feature = linear - pos * dim
                    local_token = chunk * chunk_size + pos
                    if const_expr(is_varlen):
                        value = Float32(0.0)
                        if local_token < seq_len:
                            token = begin + local_token
                            grad_idx = (
                                (Int64(batch_idx) * seqlen + token) * nheads + head
                            ) * dim + load_feature
                            value = Float32(mGradOut[grad_idx])
                        sValues[pos, load_feature] = value
                    else:
                        token = local_token
                        grad_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * dim + load_feature
                        sValues[pos, load_feature] = Float32(mGradOut[grad_idx])
                if tidx < chunk_size:
                    local_token = chunk * chunk_size + tidx
                    if const_expr(is_varlen):
                        value = Float32(0.0)
                        if local_token < seq_len:
                            token = begin + local_token
                            dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                            value = Float32(mDt[dt_idx])
                        sDt[tidx] = value
                    else:
                        token = local_token
                        dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                        sDt[tidx] = Float32(mDt[dt_idx])
                cute.arch.sync_threads()

                for reverse_item in cutlass.range_constexpr(scan_items):
                    item = scan_items - 1 - reverse_item
                    pos = item * subgroup_size + token_lane
                    value = sValues[pos, feature]
                    for offset in (1, 2, 4):
                        addend = cute.arch.shuffle_sync_down(value, offset)
                        if token_lane + offset < subgroup_size:
                            value += addend
                    value += carry
                    sValues[pos, feature] = value
                    carry = cute.arch.shuffle_sync(value, feature_group * subgroup_size)
                cute.arch.sync_threads()

                for item in cutlass.range_constexpr(items_per_thread):
                    linear = tidx + item * num_threads
                    pos = linear // dim
                    item_feature = linear - pos * dim
                    local_token = chunk * chunk_size + pos
                    if const_expr(is_varlen):
                        product = Float32(0.0)
                        if local_token < seq_len:
                            token = begin + local_token
                            angle_idx = (
                                (Int64(batch_idx) * seqlen + token) * nheads + head
                            ) * dim + item_feature
                            angle = Float32(mAngle[angle_idx])
                            tanh_angle = tanh_approx(angle)
                            grad_value = sValues[pos, item_feature]
                            mGradAngle[angle_idx] = angle_dtype(
                                grad_value
                                * sDt[pos]
                                * Float32(_PI)
                                * (Float32(1.0) - tanh_angle * tanh_angle)
                            )
                            product = grad_value * tanh_angle * Float32(_PI)
                        sValues[pos, item_feature] = product
                    else:
                        token = local_token
                        angle_idx = (
                            (Int64(batch_idx) * seqlen + token) * nheads + head
                        ) * dim + item_feature
                        angle = Float32(mAngle[angle_idx])
                        tanh_angle = tanh_approx(angle)
                        grad_value = sValues[pos, item_feature]
                        mGradAngle[angle_idx] = angle_dtype(
                            grad_value
                            * sDt[pos]
                            * Float32(_PI)
                            * (Float32(1.0) - tanh_angle * tanh_angle)
                        )
                        sValues[pos, item_feature] = (
                            grad_value * tanh_angle * Float32(_PI)
                        )
                cute.arch.sync_threads()

                for item in cutlass.range_constexpr(tokens_per_warp):
                    pos = warp + item * num_warps
                    value = Float32(0.0)
                    if lane < dim:
                        value = sValues[pos, lane]
                    for offset in (16, 8, 4, 2, 1):
                        value += cute.arch.shuffle_sync_down(value, offset)
                    if lane == 0:
                        local_token = chunk * chunk_size + pos
                        if const_expr(is_varlen):
                            if local_token < seq_len:
                                token = begin + local_token
                                dt_idx = (
                                    Int64(batch_idx) * nheads + head
                                ) * seqlen + token
                                mGradDt[dt_idx] = dt_dtype(value)
                        else:
                            token = local_token
                            dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                            mGradDt[dt_idx] = dt_dtype(value)
                cute.arch.sync_threads()

            if const_expr(has_initial):
                if token_lane == 0:
                    initial_idx = (Int64(sequence) * nheads + head) * dim + feature
                    mGradInitial[initial_idx] = carry

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        AngleDtBwdChunked(),
        fake(grad_dtype),
        fake(angle_dtype),
        fake(dt_dtype),
        fake(final_dtype) if has_grad_final else None,
        fake(index_dtype) if is_varlen else None,
        fake(angle_dtype),
        fake(dt_dtype),
        fake(Float32) if has_initial else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_angle_dt_bwd(
    angle_dtype,
    dt_dtype,
    grad_dtype,
    final_dtype,
    index_dtype,
    *,
    has_initial: bool,
    has_grad_final: bool,
    is_varlen: bool,
):
    class AngleDtBwd:
        @cute.jit
        def __call__(
            self,
            mGradOut: cute.Tensor,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mGradFinal: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mGradAngle: cute.Tensor,
            mGradDtPartial: cute.Tensor,
            mGradDt: cute.Tensor,
            mGradInitial: cute.Tensor | None,
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
            self.scan_kernel(
                mGradOut,
                mAngle,
                mDt,
                mGradFinal,
                mCuSeqlens,
                mGradAngle,
                mGradDtPartial,
                mGradInitial,
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
            dt_count = Int64(batch) * nheads * seqlen
            self.reduce_dt_kernel(
                mGradDtPartial, mGradDt, dt_count, feature_blocks
            ).launch(
                grid=[cute.ceil_div(dt_count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def scan_kernel(
            self,
            mGradOut: cute.Tensor,
            mAngle: cute.Tensor,
            mDt: cute.Tensor,
            mGradFinal: cute.Tensor | None,
            mCuSeqlens: cute.Tensor | None,
            mGradAngle: cute.Tensor,
            mGradDtPartial: cute.Tensor,
            mGradInitial: cute.Tensor | None,
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

                grad_state = [Float32(0.0) for _ in range(_VEC_SIZE)]
                state_base = (Int64(sequence) * nheads + head) * dim + feature
                if const_expr(has_grad_final):
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            grad_state[item] = Float32(mGradFinal[state_base + item])

                for token_offset in cutlass.range(0, end - begin, 1, unroll=1):
                    token = end - Int32(1) - token_offset
                    dt_idx = (Int64(batch_idx) * nheads + head) * seqlen + token
                    dt_value = Float32(mDt[dt_idx])
                    angle_base = (
                        (Int64(batch_idx) * seqlen + token) * nheads + head
                    ) * dim + feature
                    grad_dt = Float32(0.0)
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            grad_state[item] += Float32(mGradOut[angle_base + item])
                            angle = Float32(mAngle[angle_base + item])
                            tanh_angle = tanh_approx(angle)
                            grad_dt += grad_state[item] * tanh_angle * Float32(_PI)
                            mGradAngle[angle_base + item] = angle_dtype(
                                grad_state[item]
                                * dt_value
                                * Float32(_PI)
                                * (Float32(1.0) - tanh_angle * tanh_angle)
                            )
                    partial_idx = dt_idx * feature_blocks + feature_block
                    mGradDtPartial[partial_idx] = grad_dt

                if const_expr(has_initial):
                    for item in cutlass.range_constexpr(_VEC_SIZE):
                        if feature + item < dim:
                            mGradInitial[state_base + item] = grad_state[item]

        @cute.kernel
        def reduce_dt_kernel(
            self,
            mGradDtPartial: cute.Tensor,
            mGradDt: cute.Tensor,
            count: Int64,
            feature_blocks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            idx = Int64(block) * _NUM_THREADS + tidx
            if idx < count:
                value = Float32(0.0)
                for feature_block in cutlass.range(0, feature_blocks, 1, unroll=1):
                    value += Float32(
                        mGradDtPartial[idx * feature_blocks + feature_block]
                    )
                mGradDt[idx] = dt_dtype(value)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        AngleDtBwd(),
        fake(grad_dtype),
        fake(angle_dtype),
        fake(dt_dtype),
        fake(final_dtype) if has_grad_final else None,
        fake(index_dtype) if is_varlen else None,
        fake(angle_dtype),
        fake(Float32),
        fake(dt_dtype),
        fake(Float32) if has_initial else None,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def angle_dt_bwd_cute(
    grad_out,
    angle,
    dt,
    has_init_state=False,
    chunk_size=64,
    grad_output_state=None,
    cu_seqlens=None,
):
    batch, seqlen, nheads, dim = angle.shape
    if grad_out.shape != angle.shape or dt.shape != (batch, nheads, seqlen):
        raise ValueError("CuTe angle_dt backward input shapes do not match")
    is_varlen = cu_seqlens is not None
    num_sequences = cu_seqlens.shape[0] - 1 if is_varlen else batch
    if is_varlen and batch != 1:
        raise ValueError("Varlen mode requires batch=1")
    if grad_output_state is not None and grad_output_state.shape != (
        num_sequences,
        nheads,
        dim,
    ):
        raise ValueError("grad_output_state has an invalid shape")
    tensors = (grad_out, angle, dt, grad_output_state, cu_seqlens)
    if any(t is not None and (not t.is_cuda or not t.is_contiguous()) for t in tensors):
        raise ValueError("CuTe angle_dt backward requires contiguous CUDA tensors")

    grad_angle = torch.empty_like(angle)
    grad_dt = torch.empty_like(dt)
    grad_initial = (
        torch.empty(
            num_sequences, nheads, dim, device=angle.device, dtype=torch.float32
        )
        if has_init_state
        else None
    )
    if chunk_size == 64 and dim in (16, 32) and (is_varlen or seqlen % chunk_size == 0):
        compiled = _compile_angle_dt_bwd_chunked(
            torch2cute_dtype_map[angle.dtype],
            torch2cute_dtype_map[dt.dtype],
            torch2cute_dtype_map[grad_out.dtype],
            torch2cute_dtype_map[grad_output_state.dtype]
            if grad_output_state is not None
            else None,
            torch2cute_dtype_map[cu_seqlens.dtype] if is_varlen else None,
            chunk_size,
            dim,
            has_initial=has_init_state,
            has_grad_final=grad_output_state is not None,
            is_varlen=is_varlen,
        )
        compiled(
            grad_out.view(-1),
            angle.view(-1),
            dt.view(-1),
            grad_output_state.view(-1) if grad_output_state is not None else None,
            cu_seqlens.view(-1) if is_varlen else None,
            grad_angle.view(-1),
            grad_dt.view(-1),
            grad_initial.view(-1) if grad_initial is not None else None,
            batch,
            seqlen,
            nheads,
            num_sequences,
        )
        return grad_angle, grad_dt, grad_initial

    feature_blocks = (dim + _VEC_SIZE - 1) // _VEC_SIZE
    grad_dt_partial = torch.empty(
        batch, nheads, seqlen, feature_blocks, device=dt.device, dtype=torch.float32
    )
    compiled = _compile_angle_dt_bwd(
        torch2cute_dtype_map[angle.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[grad_out.dtype],
        torch2cute_dtype_map[grad_output_state.dtype]
        if grad_output_state is not None
        else None,
        torch2cute_dtype_map[cu_seqlens.dtype] if is_varlen else None,
        has_initial=has_init_state,
        has_grad_final=grad_output_state is not None,
        is_varlen=is_varlen,
    )
    compiled(
        grad_out.view(-1),
        angle.view(-1),
        dt.view(-1),
        grad_output_state.view(-1) if grad_output_state is not None else None,
        cu_seqlens.view(-1) if is_varlen else None,
        grad_angle.view(-1),
        grad_dt_partial.view(-1),
        grad_dt.view(-1),
        grad_initial.view(-1) if grad_initial is not None else None,
        batch,
        seqlen,
        nheads,
        dim,
        num_sequences,
    )
    return grad_angle, grad_dt, grad_initial


__all__ = ["angle_dt_bwd_cute", "angle_dt_fwd_cute"]
