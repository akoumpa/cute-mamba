from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map


@cache
def _compile_dacs_segsum(input_dtype, chunk_size: int):
    num_threads = 64 if chunk_size <= 32 else 128 if chunk_size <= 64 else 256

    class DacsSegsum:
        @cute.jit
        def __call__(
            self,
            mDa: cute.Tensor,
            mDaCs: cute.Tensor,
            mDaCsRev: cute.Tensor,
            mSegsum: cute.Tensor,
            batch: Int32,
            nheads: Int32,
            seqlen: Int32,
            nchunks: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDa,
                mDaCs,
                mDaCsRev,
                mSegsum,
                nheads,
                seqlen,
                nchunks,
            ).launch(
                grid=[batch * nheads * nchunks, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDa: cute.Tensor,
            mDaCs: cute.Tensor,
            mDaCsRev: cute.Tensor,
            mSegsum: cute.Tensor,
            nheads: Int32,
            seqlen: Int32,
            nchunks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            chunk = block % nchunks
            outer = block // nchunks
            head = outer % nheads
            batch_idx = outer // nheads
            chunk_start = chunk * chunk_size
            input_base = (Int64(batch_idx) * nheads + head) * seqlen + chunk_start

            smem = cutlass.utils.SmemAllocator()
            prefix = smem.allocate_tensor(
                Float32,
                cute.make_layout(chunk_size),
                byte_alignment=16,
            )
            if tidx == 0:
                value = Float32(0.0)
                for token in cutlass.range_constexpr(chunk_size):
                    if chunk_start + token < seqlen:
                        value += Float32(mDa[input_base + token])
                    prefix[token] = value
            cute.arch.sync_threads()

            if tidx < chunk_size and chunk_start + tidx < seqlen:
                value = prefix[tidx]
                reverse = prefix[chunk_size - 1] - value
                if value > 0.0:
                    value = Float32(0.0)
                if reverse > 0.0:
                    reverse = Float32(0.0)
                mDaCs[input_base + tidx] = input_dtype(value)
                mDaCsRev[input_base + tidx] = input_dtype(reverse)

            matrix_base = (
                ((Int64(batch_idx) * nheads + head) * nchunks + chunk)
                * chunk_size
                * chunk_size
            )
            for linear in cutlass.range(
                tidx, chunk_size * chunk_size, num_threads, unroll=1
            ):
                row = linear // chunk_size
                col = linear - row * chunk_size
                value = Float32(0.0)
                if row > col:
                    value = prefix[row] - prefix[col]
                    if value > 0.0:
                        value = Float32(0.0)
                mSegsum[matrix_base + linear] = input_dtype(value)

    def fake():
        return cute.runtime.make_fake_tensor(
            input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        DacsSegsum(),
        fake(),
        fake(),
        fake(),
        fake(),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compute_dacs_segsum_fixed_cute(da, chunk_size):
    if da.ndim != 3:
        raise AssertionError("da must have shape (batch, nheads, seqlen)")
    if not da.is_cuda:
        raise AssertionError("da must be a CUDA tensor")
    if chunk_size <= 0:
        raise AssertionError("chunk_size must be positive")
    da = da.contiguous()
    batch, nheads, seqlen = da.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.empty(
        batch,
        nheads,
        nchunks,
        chunk_size,
        chunk_size,
        dtype=da.dtype,
        device=da.device,
    )
    compiled = _compile_dacs_segsum(torch2cute_dtype_map[da.dtype], chunk_size)
    compiled(
        da.view(-1),
        da_cs.view(-1),
        da_cs_rev.view(-1),
        segsum.view(-1),
        batch,
        nheads,
        seqlen,
        nchunks,
    )
    return da_cs, da_cs_rev, segsum


@cache
def _compile_dacs_segsum_varlen(input_dtype, index_dtype, chunk_size: int):
    num_threads = 64 if chunk_size <= 32 else 128 if chunk_size <= 64 else 256

    class DacsSegsumVarlen:
        @cute.jit
        def __call__(
            self,
            mDa: cute.Tensor,
            mDaCs: cute.Tensor,
            mDaCsRev: cute.Tensor,
            mSegsum: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            batch: Int32,
            nheads: Int32,
            seqlen: Int32,
            nchunks: Int32,
            num_sequences: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDa,
                mDaCs,
                mDaCsRev,
                mSegsum,
                mCuSeqlens,
                nheads,
                seqlen,
                nchunks,
                num_sequences,
            ).launch(
                grid=[batch * nheads * nchunks, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDa: cute.Tensor,
            mDaCs: cute.Tensor,
            mDaCsRev: cute.Tensor,
            mSegsum: cute.Tensor,
            mCuSeqlens: cute.Tensor,
            nheads: Int32,
            seqlen: Int32,
            nchunks: Int32,
            num_sequences: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            chunk = block % nchunks
            outer = block // nchunks
            head = outer % nheads
            batch_idx = outer // nheads

            lower = Int32(0)
            upper = num_sequences
            while lower < upper:
                middle = (lower + upper) // 2
                sequence_start = Int32(mCuSeqlens[middle])
                global_start = sequence_start // chunk_size + middle
                if global_start <= chunk:
                    lower = middle + 1
                else:
                    upper = middle
            sequence = lower - 1
            sequence_start = Int32(mCuSeqlens[sequence])
            sequence_end = Int32(mCuSeqlens[sequence + 1])
            global_start = sequence_start // chunk_size + sequence
            local_chunk = chunk - global_start
            token_start = sequence_start + local_chunk * chunk_size
            input_base = (Int64(batch_idx) * nheads + head) * seqlen

            smem = cutlass.utils.SmemAllocator()
            prefix = smem.allocate_tensor(
                Float32,
                cute.make_layout(chunk_size),
                byte_alignment=16,
            )
            if tidx == 0:
                value = Float32(0.0)
                for token in cutlass.range_constexpr(chunk_size):
                    position = token_start + token
                    if position < sequence_end and position < seqlen:
                        value += Float32(mDa[input_base + position])
                    prefix[token] = value
            cute.arch.sync_threads()

            if tidx < chunk_size:
                position = token_start + tidx
                if position < sequence_end and position < seqlen:
                    value = prefix[tidx]
                    reverse = prefix[chunk_size - 1] - value
                    if value > 0.0:
                        value = Float32(0.0)
                    if reverse > 0.0:
                        reverse = Float32(0.0)
                    mDaCs[input_base + position] = input_dtype(value)
                    mDaCsRev[input_base + position] = input_dtype(reverse)

            matrix_base = (
                ((Int64(batch_idx) * nheads + head) * nchunks + chunk)
                * chunk_size
                * chunk_size
            )
            for linear in cutlass.range(
                tidx, chunk_size * chunk_size, num_threads, unroll=1
            ):
                row = linear // chunk_size
                col = linear - row * chunk_size
                value = Float32(0.0)
                if row > col:
                    value = prefix[row] - prefix[col]
                    if value > 0.0:
                        value = Float32(0.0)
                mSegsum[matrix_base + linear] = input_dtype(value)

    def fake(dtype, align=16):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=align
        )

    return cute.compile(
        DacsSegsumVarlen(),
        fake(input_dtype),
        fake(input_dtype),
        fake(input_dtype),
        fake(input_dtype),
        fake(index_dtype, 4),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compute_dacs_segsum_varlen_cute(da, chunk_size, cu_seqlens):
    if da.ndim != 3 or cu_seqlens.ndim != 1:
        raise AssertionError("da must be rank 3 and cu_seqlens must be rank 1")
    if not da.is_cuda or not cu_seqlens.is_cuda:
        raise AssertionError("varlen segmented prefix tensors must be CUDA tensors")
    da = da.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    batch, nheads, seqlen = da.shape
    num_sequences = cu_seqlens.numel() - 1
    nchunks = seqlen // chunk_size + num_sequences
    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.empty(
        batch,
        nheads,
        nchunks,
        chunk_size,
        chunk_size,
        dtype=da.dtype,
        device=da.device,
    )
    compiled = _compile_dacs_segsum_varlen(
        torch2cute_dtype_map[da.dtype],
        torch2cute_dtype_map[cu_seqlens.dtype],
        chunk_size,
    )
    compiled(
        da.view(-1),
        da_cs.view(-1),
        da_cs_rev.view(-1),
        segsum.view(-1),
        cu_seqlens.view(-1),
        batch,
        nheads,
        seqlen,
        nchunks,
        num_sequences,
    )
    return da_cs, da_cs_rev, segsum


__all__ = ["compute_dacs_segsum_fixed_cute", "compute_dacs_segsum_varlen_cute"]
