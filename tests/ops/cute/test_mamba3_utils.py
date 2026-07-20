from functools import cache

import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@cache
def _compile_utils_kernel():
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64

    from mamba_ssm.ops.cute.mamba3.utils import (
        cos_approx,
        sech2_approx,
        sigmoid_approx,
        silu,
        sin_approx,
        tanh_approx,
    )
    from mamba_ssm.ops.cute.softplus import softplus

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
                mOut[base + 0] = cos_approx(x)
                mOut[base + 1] = sin_approx(x)
                mOut[base + 2] = tanh_approx(x)
                mOut[base + 3] = sech2_approx(x)
                mOut[base + 4] = sigmoid_approx(x)
                mOut[base + 5] = silu(x)
                mOut[base + 6] = softplus(x)

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


def test_mamba3_cute_utils_match_reference():
    _require_cute()
    x = torch.linspace(-5.0, 5.0, 4096, device="cuda")
    out = torch.empty(x.numel(), 7, device="cuda")
    _compile_utils_kernel()(x, out.view(-1), x.numel())
    expected = torch.stack(
        (
            torch.cos(x),
            torch.sin(x),
            torch.tanh(x),
            1.0 - torch.tanh(x).square(),
            torch.sigmoid(x),
            torch.nn.functional.silu(x),
            torch.nn.functional.softplus(x),
        ),
        dim=-1,
    )
    torch.testing.assert_close(out, expected, rtol=3e-4, atol=3e-4)
