from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map


@cache
def _compile_dadt_backward(
    dss_dtype,
    ss_dtype,
    ddacs_dtype,
    ddacs_rev_dtype,
    dacs_dtype,
    dacs_rev_dtype,
    chunk_size: int,
):
    num_threads = max(32, chunk_size)

    class DadtBackward:
        @cute.jit
        def __call__(
            self,
            mDss: cute.Tensor,
            mSs: cute.Tensor,
            mDdacs: cute.Tensor,
            mDdacsRev: cute.Tensor,
            mDacs: cute.Tensor,
            mDacsRev: cute.Tensor,
            mOut: cute.Tensor,
            batch: Int32,
            nheads: Int32,
            nchunks: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDss,
                mSs,
                mDdacs,
                mDdacsRev,
                mDacs,
                mDacsRev,
                mOut,
                nheads,
                nchunks,
            ).launch(
                grid=[batch * nheads * nchunks, 1, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDss: cute.Tensor,
            mSs: cute.Tensor,
            mDdacs: cute.Tensor,
            mDdacsRev: cute.Tensor,
            mDacs: cute.Tensor,
            mDacsRev: cute.Tensor,
            mOut: cute.Tensor,
            nheads: Int32,
            nchunks: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            chunk = block % nchunks
            outer = block // nchunks
            head = outer % nheads
            batch_idx = outer // nheads
            chunk_base = (Int64(batch_idx) * nheads + head) * nchunks + chunk
            matrix_base = chunk_base * chunk_size * chunk_size
            vector_base = chunk_base * chunk_size

            smem = cutlass.utils.SmemAllocator()
            reverse_columns = smem.allocate_tensor(
                Float32,
                cute.make_layout(chunk_size * chunk_size),
                byte_alignment=16,
            )
            if tidx < chunk_size:
                column = tidx
                value = Float32(0.0)
                for step in cutlass.range_constexpr(chunk_size):
                    row = chunk_size - 1 - step
                    dss = Float32(mDss[matrix_base + column * chunk_size + row])
                    ss = Float32(mSs[matrix_base + row * chunk_size + column])
                    value += dss * cute.math.exp(ss, fastmath=True)
                    reverse_columns[row * chunk_size + column] = value
            cute.arch.sync_threads()

            if tidx < chunk_size:
                row = tidx
                cs = Float32(0.0)
                rev = Float32(0.0)
                seg = Float32(0.0)
                for token in cutlass.range_constexpr(chunk_size):
                    if token >= row:
                        cs += Float32(mDdacs[vector_base + token]) * cute.math.exp(
                            Float32(mDacs[vector_base + token]), fastmath=True
                        )
                    if token < row:
                        rev += Float32(mDdacsRev[vector_base + token]) * cute.math.exp(
                            Float32(mDacsRev[vector_base + token]), fastmath=True
                        )
                        seg += reverse_columns[row * chunk_size + token]
                mOut[vector_base + row] = cs + rev + seg

    def fake(dtype):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        DadtBackward(),
        fake(dss_dtype),
        fake(ss_dtype),
        fake(ddacs_dtype),
        fake(ddacs_rev_dtype),
        fake(dacs_dtype),
        fake(dacs_rev_dtype),
        fake(Float32),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def bwd_dadt_fixed_cute(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev):
    batch, nheads, seqlen = ddA_cs.shape
    chunk_size = dSSdA.shape[-1]
    nchunks = seqlen // chunk_size
    expected_matrix_shape = (batch, nheads, nchunks, chunk_size, chunk_size)
    if dSSdA.shape != expected_matrix_shape or SSdA.shape != expected_matrix_shape:
        raise AssertionError("dSSdA and SSdA must match the chunked matrix shape")
    if any(tensor.shape != ddA_cs.shape for tensor in (ddA_cs_rev, dA_cs, dA_cs_rev)):
        raise AssertionError("scalar backward tensors must have identical shapes")
    tensors = (dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev)
    if any(not tensor.is_cuda for tensor in tensors):
        raise AssertionError("Mamba3 dA backward tensors must be CUDA tensors")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev = tensors
    out = torch.empty(ddA_cs.shape, dtype=torch.float32, device=ddA_cs.device)
    compiled = _compile_dadt_backward(
        *(torch2cute_dtype_map[tensor.dtype] for tensor in tensors),
        chunk_size,
    )
    compiled(
        dSSdA.view(-1),
        SSdA.view(-1),
        ddA_cs.view(-1),
        ddA_cs_rev.view(-1),
        dA_cs.view(-1),
        dA_cs_rev.view(-1),
        out.view(-1),
        batch,
        nheads,
        nchunks,
    )
    return out


__all__ = ["bwd_dadt_fixed_cute"]
