from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map


_NUM_THREADS = 256


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
                input_idx = ((row * rank + rank_idx) * nheads + group * group_size) * dim + feature
                values = [Float32(0.0) for _ in range(vec_size)]
                for head_offset in cutlass.range_constexpr(group_size):
                    for item in cutlass.range_constexpr(vec_size):
                        values[item] += Float32(mInput[input_idx + head_offset * dim + item])
                output_idx = ((row * rank + rank_idx) * ngroups + group) * dim + feature
                for item in cutlass.range_constexpr(vec_size):
                    mOutput[output_idx + item] = input_dtype(values[item])

    fake = lambda: cute.runtime.make_fake_tensor(
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


def reduce_grouped_qk_grads_cute(qk_grads: torch.Tensor, num_qk_groups: int) -> torch.Tensor:
    if qk_grads.ndim != 5:
        raise ValueError(
            f"Expected qk_grads to have shape [B, S, R, H, N], got {tuple(qk_grads.shape)}."
        )
    num_heads = qk_grads.shape[3]
    if num_heads % num_qk_groups != 0:
        raise ValueError(f"Expected H ({num_heads}) to be divisible by G ({num_qk_groups}).")
    if not qk_grads.is_cuda or not qk_grads.is_contiguous():
        raise ValueError("CuTe grouped Q/K reduction requires a contiguous CUDA tensor.")

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


__all__ = ["reduce_grouped_qk_grads_cute"]
