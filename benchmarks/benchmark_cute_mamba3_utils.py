from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32, Int32, Int64
from triton.testing import do_bench

from mamba_ssm.ops.cute.mamba3.utils import (
    cos_approx as cos_approx_cute,
    sech2_approx as sech2_approx_cute,
    sigmoid_approx as sigmoid_approx_cute,
    silu as silu_cute,
    sin_approx as sin_approx_cute,
    tanh_approx as tanh_approx_cute,
)
from mamba_ssm.ops.cute.softplus import softplus as softplus_cute
from mamba_ssm.ops.triton.mamba3.utils import (
    cos_approx,
    sech2_approx,
    sigmoid_approx,
    silu,
    sin_approx,
    tanh_approx,
)
from mamba_ssm.ops.triton.softplus import softplus


OPERATIONS = ("cos", "sin", "tanh", "sech2", "sigmoid", "silu", "softplus")


@cache
def _compile_cute_utils():
    class UtilsKernel:
        @cute.jit
        def __call__(self, mX, mOut, count: Int32, stream: cuda.CUstream):
            self.kernel(mX, mOut, count).launch(
                grid=[cute.ceil_div(count, 256), 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mOut: cute.Tensor, count: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            idx = Int64(block) * 256 + tidx
            if idx < count:
                x = Float32(mX[idx])
                base = idx * 7
                mOut[base + 0] = cos_approx_cute(x)
                mOut[base + 1] = sin_approx_cute(x)
                mOut[base + 2] = tanh_approx_cute(x)
                mOut[base + 3] = sech2_approx_cute(x)
                mOut[base + 4] = sigmoid_approx_cute(x)
                mOut[base + 5] = silu_cute(x)
                mOut[base + 6] = softplus_cute(x)

    x_fake = cute.runtime.make_fake_tensor(
        Float32, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    out_fake = cute.runtime.make_fake_tensor(
        Float32, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    return cute.compile(
        UtilsKernel(),
        x_fake,
        out_fake,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@triton.jit
def _triton_utils_kernel(x_ptr, out_ptr, count: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < count
    x = tl.load(x_ptr + idx, mask=mask)
    base = idx * 7
    tl.store(out_ptr + base + 0, cos_approx(x), mask=mask)
    tl.store(out_ptr + base + 1, sin_approx(x), mask=mask)
    tl.store(out_ptr + base + 2, tanh_approx(x), mask=mask)
    tl.store(out_ptr + base + 3, sech2_approx(x), mask=mask)
    tl.store(out_ptr + base + 4, sigmoid_approx(x), mask=mask)
    tl.store(out_ptr + base + 5, silu(x), mask=mask)
    tl.store(out_ptr + base + 6, softplus(x), mask=mask)


@cache
def _compile_cute_util(operation: str):
    class UtilKernel:
        @cute.jit
        def __call__(self, mX, mOut, count: Int32, stream: cuda.CUstream):
            self.kernel(mX, mOut, count).launch(
                grid=[cute.ceil_div(count, 256), 1, 1],
                block=[256, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mOut: cute.Tensor, count: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            idx = Int64(block) * 256 + tidx
            if idx < count:
                x = Float32(mX[idx])
                value = Float32(0.0)
                if operation == "cos":
                    value = cos_approx_cute(x)
                elif operation == "sin":
                    value = sin_approx_cute(x)
                elif operation == "tanh":
                    value = tanh_approx_cute(x)
                elif operation == "sech2":
                    value = sech2_approx_cute(x)
                elif operation == "sigmoid":
                    value = sigmoid_approx_cute(x)
                elif operation == "silu":
                    value = silu_cute(x)
                else:
                    value = softplus_cute(x)
                mOut[idx] = value

    fake = cute.runtime.make_fake_tensor(
        Float32, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    return cute.compile(
        UtilKernel(),
        fake,
        fake,
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@triton.jit
def _triton_util_kernel(
    x_ptr,
    out_ptr,
    count: tl.constexpr,
    block: tl.constexpr,
    operation: tl.constexpr,
):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    mask = idx < count
    x = tl.load(x_ptr + idx, mask=mask)
    if operation == 0:
        value = cos_approx(x)
    elif operation == 1:
        value = sin_approx(x)
    elif operation == 2:
        value = tanh_approx(x)
    elif operation == 3:
        value = sech2_approx(x)
    elif operation == 4:
        value = sigmoid_approx(x)
    elif operation == 5:
        value = silu(x)
    else:
        value = softplus(x)
    tl.store(out_ptr + idx, value, mask=mask)


def benchmark_case(count):
    x = torch.randn(count, device="cuda")
    cute_out = torch.empty(count, 7, device="cuda")
    triton_out = torch.empty_like(cute_out)
    compiled = _compile_cute_utils()

    def cute_fn():
        compiled(x, cute_out.view(-1), count)

    def triton_fn():
        _triton_utils_kernel[(triton.cdiv(count, 256),)](
            x, triton_out, count, 256, num_warps=8
        )

    cute_fn()
    triton_fn()
    torch.testing.assert_close(cute_out, triton_out, rtol=3e-4, atol=3e-4)
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"op=fused count={count} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


def benchmark_operation(operation, count):
    x = torch.randn(count, device="cuda")
    cute_out = torch.empty_like(x)
    triton_out = torch.empty_like(x)
    compiled = _compile_cute_util(operation)
    operation_id = OPERATIONS.index(operation)

    def cute_fn():
        compiled(x, cute_out, count)

    def triton_fn():
        _triton_util_kernel[(triton.cdiv(count, 256),)](
            x, triton_out, count, 256, operation_id, num_warps=8
        )

    cute_fn()
    triton_fn()
    torch.testing.assert_close(cute_out, triton_out, rtol=3e-4, atol=3e-4)
    cute_ms = do_bench(cute_fn, warmup=50, rep=300)
    triton_ms = do_bench(triton_fn, warmup=50, rep=300)
    print(
        f"op={operation} count={count} speedup={triton_ms / cute_ms:.3f} "
        f"cute_ms={cute_ms:.4f} triton_ms={triton_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    for count in (4096, 1 << 20, 1 << 24):
        benchmark_case(count)
        for operation in OPERATIONS:
            benchmark_operation(operation, count)
