from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map


_NUM_THREADS = 256
_NUM_WARPS = _NUM_THREADS // 32


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for offset in (16, 8, 4, 2, 1):
        value += cute.arch.shuffle_sync_down(value, offset)
    return value


@cache
def _compile_reduce_grouped_qk_grads(input_dtype, group_size: int, vec_size: int):
    class ReduceGroupedQKGrads:
        @cute.jit
        def __call__(
            self,
            mInput: cute.Tensor,
            mOutput: cute.Tensor,
            total_rows: Int32,
            rank: Int32,
            nheads: Int32,
            ngroups: Int32,
            dim: Int32,
            stream: cuda.CUstream,
        ):
            work = Int64(total_rows) * rank * ngroups * (dim // vec_size)
            self.kernel(mInput, mOutput, work, rank, nheads, ngroups, dim).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mInput: cute.Tensor,
            mOutput: cute.Tensor,
            work: Int64,
            rank: Int32,
            nheads: Int32,
            ngroups: Int32,
            dim: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < work:
                feature_block = linear % (dim // vec_size)
                outer = linear // (dim // vec_size)
                group = outer % ngroups
                outer = outer // ngroups
                rank_idx = outer % rank
                row = outer // rank
                feature = feature_block * vec_size
                input_idx = (
                    (row * rank + rank_idx) * nheads + group * group_size
                ) * dim + feature
                values = [Float32(0.0) for _ in range(vec_size)]
                for head_offset in cutlass.range_constexpr(group_size):
                    for item in cutlass.range_constexpr(vec_size):
                        values[item] += Float32(
                            mInput[input_idx + head_offset * dim + item]
                        )
                output_idx = ((row * rank + rank_idx) * ngroups + group) * dim + feature
                for item in cutlass.range_constexpr(vec_size):
                    mOutput[output_idx + item] = input_dtype(values[item])

    def fake():
        return cute.runtime.make_fake_tensor(
            input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        ReduceGroupedQKGrads(),
        fake(),
        fake(),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def reduce_grouped_qk_grads_cute(
    qk_grads: torch.Tensor, num_qk_groups: int
) -> torch.Tensor:
    if qk_grads.ndim != 5:
        raise ValueError(
            f"Expected qk_grads to have shape [B, S, R, H, N], got {tuple(qk_grads.shape)}."
        )
    num_heads = qk_grads.shape[3]
    if num_heads % num_qk_groups != 0:
        raise ValueError(
            f"Expected H ({num_heads}) to be divisible by G ({num_qk_groups})."
        )
    if not qk_grads.is_cuda or not qk_grads.is_contiguous():
        raise ValueError(
            "CuTe grouped Q/K reduction requires a contiguous CUDA tensor."
        )

    batch, seqlen, rank, _, dim = qk_grads.shape
    output = torch.empty(
        batch,
        seqlen,
        rank,
        num_qk_groups,
        dim,
        device=qk_grads.device,
        dtype=qk_grads.dtype,
    )
    vec_size = 4 if dim % 4 == 0 else 1
    compiled = _compile_reduce_grouped_qk_grads(
        torch2cute_dtype_map[qk_grads.dtype], num_heads // num_qk_groups, vec_size
    )
    compiled(
        qk_grads.view(-1),
        output.view(-1),
        batch * seqlen,
        rank,
        num_heads,
        num_qk_groups,
        dim,
    )
    return output


@cache
def _compile_reduce_grouped_qk_grads_pair(input_dtype, group_size: int, vec_size: int):
    class ReduceGroupedQKGradsPair:
        @cute.jit
        def __call__(
            self,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDQOut: cute.Tensor,
            mDKOut: cute.Tensor,
            total_rows: Int32,
            rank: Int32,
            nheads: Int32,
            ngroups: Int32,
            dim: Int32,
            stream: cuda.CUstream,
        ):
            work = Int64(total_rows) * rank * ngroups * (dim // vec_size)
            self.kernel(
                mDQ, mDK, mDQOut, mDKOut, work, rank, nheads, ngroups, dim
            ).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDQOut: cute.Tensor,
            mDKOut: cute.Tensor,
            work: Int64,
            rank: Int32,
            nheads: Int32,
            ngroups: Int32,
            dim: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            linear = Int64(block) * _NUM_THREADS + tidx
            if linear < work:
                feature_block = linear % (dim // vec_size)
                outer = linear // (dim // vec_size)
                group = outer % ngroups
                outer = outer // ngroups
                rank_idx = outer % rank
                row = outer // rank
                feature = feature_block * vec_size
                input_idx = (
                    (row * rank + rank_idx) * nheads + group * group_size
                ) * dim + feature
                dq_values = [Float32(0.0) for _ in range(vec_size)]
                dk_values = [Float32(0.0) for _ in range(vec_size)]
                for head_offset in cutlass.range_constexpr(group_size):
                    for item in cutlass.range_constexpr(vec_size):
                        offset = input_idx + head_offset * dim + item
                        dq_values[item] += Float32(mDQ[offset])
                        dk_values[item] += Float32(mDK[offset])
                output_idx = ((row * rank + rank_idx) * ngroups + group) * dim + feature
                for item in cutlass.range_constexpr(vec_size):
                    mDQOut[output_idx + item] = input_dtype(dq_values[item])
                    mDKOut[output_idx + item] = input_dtype(dk_values[item])

    def fake():
        return cute.runtime.make_fake_tensor(
            input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        ReduceGroupedQKGradsPair(),
        fake(),
        fake(),
        fake(),
        fake(),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _reduce_grouped_qk_grads_pair_cute(
    dq_raw: torch.Tensor, dk_raw: torch.Tensor, num_qk_groups: int
):
    batch, seqlen, rank, nheads, dim = dq_raw.shape
    dq_out = torch.empty(
        batch,
        seqlen,
        rank,
        num_qk_groups,
        dim,
        device=dq_raw.device,
        dtype=dq_raw.dtype,
    )
    dk_out = torch.empty_like(dq_out)
    vec_size = 4 if dim % 4 == 0 else 1
    compiled = _compile_reduce_grouped_qk_grads_pair(
        torch2cute_dtype_map[dq_raw.dtype], nheads // num_qk_groups, vec_size
    )
    compiled(
        dq_raw.view(-1),
        dk_raw.view(-1),
        dq_out.view(-1),
        dk_out.view(-1),
        batch * seqlen,
        rank,
        nheads,
        num_qk_groups,
        dim,
    )
    return dq_out, dk_out


@cache
def _compile_qk_bias_reduction(input_dtype, vec_size: int):
    class QKBiasReduction:
        @cute.jit
        def __call__(
            self,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDQBias: cute.Tensor,
            mDKBias: cute.Tensor,
            total_rows: Int32,
            rank: Int32,
            nheads: Int32,
            dim: Int32,
            stream: cuda.CUstream,
        ):
            feature_blocks = cute.ceil_div(dim, vec_size)
            work = Int64(nheads) * rank * feature_blocks
            self.kernel(
                mDQ,
                mDK,
                mDQBias,
                mDKBias,
                total_rows,
                rank,
                nheads,
                dim,
                feature_blocks,
            ).launch(
                grid=[work, 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDQ: cute.Tensor,
            mDK: cute.Tensor,
            mDQBias: cute.Tensor,
            mDKBias: cute.Tensor,
            total_rows: Int32,
            rank: Int32,
            nheads: Int32,
            dim: Int32,
            feature_blocks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            feature_block = block % feature_blocks
            outer = block // feature_blocks
            rank_idx = outer % rank
            head = outer // rank
            feature = feature_block * vec_size

            dq_sum = [Float32(0.0) for _ in range(vec_size)]
            dk_sum = [Float32(0.0) for _ in range(vec_size)]
            for row in cutlass.range(tidx, total_rows, _NUM_THREADS, unroll=1):
                base = ((Int64(row) * rank + rank_idx) * nheads + head) * dim + feature
                for item in cutlass.range_constexpr(vec_size):
                    if feature + item < dim:
                        dq_sum[item] += Float32(mDQ[base + item])
                        dk_sum[item] += Float32(mDK[base + item])

            smem = cutlass.utils.SmemAllocator()
            partial = smem.allocate_tensor(
                Float32,
                cute.make_layout(2 * vec_size * _NUM_WARPS),
                byte_alignment=16,
            )
            for item in cutlass.range_constexpr(vec_size):
                dq_sum[item] = _warp_sum(dq_sum[item])
                dk_sum[item] = _warp_sum(dk_sum[item])
                if lane == 0:
                    partial[item * _NUM_WARPS + warp] = dq_sum[item]
                    partial[(vec_size + item) * _NUM_WARPS + warp] = dk_sum[item]
            cute.arch.sync_threads()

            if warp == 0:
                for item in cutlass.range_constexpr(vec_size):
                    dq_value = Float32(0.0)
                    dk_value = Float32(0.0)
                    if lane < _NUM_WARPS:
                        dq_value = partial[item * _NUM_WARPS + lane]
                        dk_value = partial[(vec_size + item) * _NUM_WARPS + lane]
                    dq_value = _warp_sum(dq_value)
                    dk_value = _warp_sum(dk_value)
                    if lane == 0 and feature + item < dim:
                        output_idx = (head * rank + rank_idx) * dim + feature + item
                        mDQBias[output_idx] = dq_value
                        mDKBias[output_idx] = dk_value

    input_fake = cute.runtime.make_fake_tensor(
        input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    output_fake = cute.runtime.make_fake_tensor(
        Float32, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    return cute.compile(
        QKBiasReduction(),
        input_fake,
        input_fake,
        output_fake,
        output_fake,
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def reduce_grouped_qk_grads_and_bias_cute(
    dq_raw: torch.Tensor,
    dk_raw: torch.Tensor,
    num_qk_groups: int,
    block_m: int = 64,
    block_n: int | None = None,
):
    del block_m, block_n
    if dq_raw.shape != dk_raw.shape:
        raise ValueError(
            f"Expected dq_raw and dk_raw to have the same shape, got "
            f"{tuple(dq_raw.shape)} and {tuple(dk_raw.shape)}."
        )
    if dq_raw.ndim != 5:
        raise ValueError(
            f"Expected raw Q/K grads to have shape [B, S, R, H, N], got "
            f"{tuple(dq_raw.shape)}."
        )
    if not dq_raw.is_cuda or not dq_raw.is_contiguous() or not dk_raw.is_contiguous():
        raise ValueError("CuTe raw Q/K grads must be contiguous CUDA tensors.")

    batch, seqlen, rank, nheads, dim = dq_raw.shape
    if nheads % num_qk_groups != 0:
        raise ValueError(
            f"Expected H ({nheads}) to be divisible by G ({num_qk_groups})."
        )
    if nheads == num_qk_groups:
        dq_out, dk_out = dq_raw, dk_raw
    elif dq_raw.dtype == torch.bfloat16:
        dq_out, dk_out = _reduce_grouped_qk_grads_pair_cute(
            dq_raw, dk_raw, num_qk_groups
        )
    else:
        dq_out = reduce_grouped_qk_grads_cute(dq_raw, num_qk_groups)
        dk_out = reduce_grouped_qk_grads_cute(dk_raw, num_qk_groups)
    dq_bias = torch.empty(nheads, rank, dim, device=dq_raw.device, dtype=torch.float32)
    dk_bias = torch.empty_like(dq_bias)
    vec_size = 8 if dim % 8 == 0 else 4 if dim % 4 == 0 else 1
    compiled = _compile_qk_bias_reduction(torch2cute_dtype_map[dq_raw.dtype], vec_size)
    compiled(
        dq_raw.view(-1),
        dk_raw.view(-1),
        dq_bias.view(-1),
        dk_bias.view(-1),
        batch * seqlen,
        rank,
        nheads,
        dim,
    )
    return dq_out, dk_out, dq_bias, dk_bias


__all__ = ["reduce_grouped_qk_grads_and_bias_cute", "reduce_grouped_qk_grads_cute"]
