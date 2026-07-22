from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.utils import sigmoid_approx


_NUM_THREADS = 256
_VEC_SIZE = 1


@cache
def _compile_compute_dzdo(do_dtype, z_dtype, o_dtype):
    class ComputeDzDo:
        @cute.jit
        def __call__(
            self,
            mDo: cute.Tensor,
            mZ: cute.Tensor,
            mO: cute.Tensor,
            mDz: cute.Tensor,
            mDoScaled: cute.Tensor,
            count: Int64,
            stream: cuda.CUstream,
        ):
            work = cute.ceil_div(count, _VEC_SIZE)
            self.kernel(mDo, mZ, mO, mDz, mDoScaled, count).launch(
                grid=[cute.ceil_div(work, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDo: cute.Tensor,
            mZ: cute.Tensor,
            mO: cute.Tensor,
            mDz: cute.Tensor,
            mDoScaled: cute.Tensor,
            count: Int64,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            base = (Int64(block) * _NUM_THREADS + tidx) * _VEC_SIZE
            for item in cutlass.range_constexpr(_VEC_SIZE):
                index = base + item
                if index < count:
                    grad = Float32(mDo[index])
                    z = Float32(mZ[index])
                    output = Float32(mO[index])
                    sigmoid = sigmoid_approx(z)
                    silu = z * sigmoid
                    mDz[index] = do_dtype(
                        grad * output * (sigmoid + silu * (Float32(1.0) - sigmoid))
                    )
                    mDoScaled[index] = do_dtype(grad * silu)

    def fake(dtype):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        ComputeDzDo(),
        fake(do_dtype),
        fake(z_dtype),
        fake(o_dtype),
        fake(do_dtype),
        fake(do_dtype),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compute_dzdo_cute(do, z, o, chunk_size=64):
    del chunk_size
    if z.shape != do.shape or o.shape != do.shape:
        raise AssertionError("Z, O, and dO must have identical shapes")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in (do, z, o)):
        raise ValueError("CuTe gate gradients require contiguous CUDA tensors")
    dz, do_scaled = torch.empty_like(do), torch.empty_like(do)
    compiled = _compile_compute_dzdo(
        torch2cute_dtype_map[do.dtype],
        torch2cute_dtype_map[z.dtype],
        torch2cute_dtype_map[o.dtype],
    )
    compiled(
        do.view(-1),
        z.view(-1),
        o.view(-1),
        dz.view(-1),
        do_scaled.view(-1),
        do.numel(),
    )
    return dz, do_scaled


__all__ = ["compute_dzdo_cute"]
