from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, const_expr
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.packing import bf16x2_to_i32, i32_to_bf16x2
from mamba_ssm.ops.cute.mamba3.utils import sigmoid_approx


_NUM_THREADS = 256
_VEC_SIZE = 1
_PACKED_BF16_MIN_ELEMENTS = 1 << 22


@cache
def _compile_compute_dzdo(do_dtype, z_dtype, o_dtype, *, packed_bf16: bool):
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
            work = count // 2 if const_expr(packed_bf16) else count
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
            if const_expr(packed_bf16):
                packed_index = Int64(block) * _NUM_THREADS + tidx
                if packed_index < count // 2:
                    do_packed = cute.recast_tensor(mDo, Int32)
                    z_packed = cute.recast_tensor(mZ, Int32)
                    o_packed = cute.recast_tensor(mO, Int32)
                    dz_packed = cute.recast_tensor(mDz, Int32)
                    do_scaled_packed = cute.recast_tensor(mDoScaled, Int32)
                    grad0, grad1 = i32_to_bf16x2(do_packed[packed_index])
                    z0, z1 = i32_to_bf16x2(z_packed[packed_index])
                    output0, output1 = i32_to_bf16x2(o_packed[packed_index])
                    sigmoid0 = sigmoid_approx(Float32(z0))
                    sigmoid1 = sigmoid_approx(Float32(z1))
                    silu0 = Float32(z0) * sigmoid0
                    silu1 = Float32(z1) * sigmoid1
                    dz0 = do_dtype(
                        Float32(grad0)
                        * Float32(output0)
                        * (sigmoid0 + silu0 * (Float32(1.0) - sigmoid0))
                    )
                    dz1 = do_dtype(
                        Float32(grad1)
                        * Float32(output1)
                        * (sigmoid1 + silu1 * (Float32(1.0) - sigmoid1))
                    )
                    do_scaled0 = do_dtype(Float32(grad0) * silu0)
                    do_scaled1 = do_dtype(Float32(grad1) * silu1)
                    dz_packed[packed_index] = bf16x2_to_i32(dz0, dz1)
                    do_scaled_packed[packed_index] = bf16x2_to_i32(
                        do_scaled0, do_scaled1
                    )
            else:
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
    supported = (
        z.shape == do.shape
        and o.shape == do.shape
        and do.dtype == z.dtype == o.dtype == torch.bfloat16
        and do.numel() >= _PACKED_BF16_MIN_ELEMENTS
        and do.numel() % 2 == 0
        and all(tensor.is_cuda and tensor.is_contiguous() for tensor in (do, z, o))
        and torch.cuda.get_device_capability(do.device)[0] >= 9
    )
    if not supported:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo

        return compute_dzdo(do, z, o, chunk_size)

    dz, do_scaled = torch.empty_like(do), torch.empty_like(do)
    compiled = _compile_compute_dzdo(
        torch2cute_dtype_map[do.dtype],
        torch2cute_dtype_map[z.dtype],
        torch2cute_dtype_map[o.dtype],
        packed_bf16=True,
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
