from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.utils import sigmoid_approx


_NUM_THREADS = 256


@cache
def _compile_ddt_dtrap(
    dscale_dtype, dgamma_dtype, dt_dtype, trap_dtype, ddt_dtype, dtrap_dtype
):
    class ComputeDdtDtrap:
        @cute.jit
        def __call__(
            self,
            mDscale: cute.Tensor,
            mDgamma: cute.Tensor,
            mDt: cute.Tensor,
            mTrap: cute.Tensor,
            mDdt: cute.Tensor,
            mDtrap: cute.Tensor,
            count: Int64,
            seqlen: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mDscale, mDgamma, mDt, mTrap, mDdt, mDtrap, count, seqlen
            ).launch(
                grid=[cute.ceil_div(count, _NUM_THREADS), 1, 1],
                block=[_NUM_THREADS, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mDscale: cute.Tensor,
            mDgamma: cute.Tensor,
            mDt: cute.Tensor,
            mTrap: cute.Tensor,
            mDdt: cute.Tensor,
            mDtrap: cute.Tensor,
            count: Int64,
            seqlen: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            index = Int64(block) * _NUM_THREADS + tidx
            if index < count:
                dscale = Float32(mDscale[index])
                dgamma = Float32(mDgamma[index])
                dt = Float32(mDt[index])
                trap = sigmoid_approx(Float32(mTrap[index]))
                dscale_prev = Float32(0.0)
                if index % seqlen != 0:
                    dscale_prev = Float32(mDscale[index - 1])
                combined = dgamma + dscale
                mDdt[index] = ddt_dtype(
                    combined * trap + dscale_prev * (Float32(1.0) - trap)
                )
                mDtrap[index] = dtrap_dtype(
                    (combined - dscale_prev) * dt * trap * (Float32(1.0) - trap)
                )

    def fake(dtype):
        return cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )

    return cute.compile(
        ComputeDdtDtrap(),
        fake(dscale_dtype),
        fake(dgamma_dtype),
        fake(dt_dtype),
        fake(trap_dtype),
        fake(ddt_dtype),
        fake(dtrap_dtype),
        Int64(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compute_shifted_scalar_grads_cute(
    dscale, dgamma, dt, trap, *, output_float32=False
):
    if dgamma.shape != dscale.shape or dt.shape != dscale.shape:
        raise AssertionError("dscale, dgamma, and dt must have identical shapes")
    if trap.shape != dscale.shape or dscale.ndim != 3:
        raise AssertionError("trap and scalar gradients must have rank-3 shapes")
    tensors = tuple(tensor.contiguous() for tensor in (dscale, dgamma, dt, trap))
    if any(not tensor.is_cuda for tensor in tensors):
        raise AssertionError("Mamba3 scalar backward tensors must be CUDA tensors")
    dscale, dgamma, dt, trap = tensors
    ddt_dtype = torch.float32 if output_float32 else dt.dtype
    dtrap_dtype = torch.float32 if output_float32 else trap.dtype
    ddt = torch.empty(dscale.shape, dtype=ddt_dtype, device=dscale.device)
    dtrap = torch.empty(dscale.shape, dtype=dtrap_dtype, device=dscale.device)
    compiled = _compile_ddt_dtrap(
        torch2cute_dtype_map[dscale.dtype],
        torch2cute_dtype_map[dgamma.dtype],
        torch2cute_dtype_map[dt.dtype],
        torch2cute_dtype_map[trap.dtype],
        torch2cute_dtype_map[ddt_dtype],
        torch2cute_dtype_map[dtrap_dtype],
    )
    compiled(
        dscale.view(-1),
        dgamma.view(-1),
        dt.view(-1),
        trap.view(-1),
        ddt.view(-1),
        dtrap.view(-1),
        dscale.numel(),
        dscale.shape[-1],
    )
    return ddt, dtrap


def compute_ddt_dtrap_cute(dscale, dgamma, dt, trap):
    return compute_shifted_scalar_grads_cute(
        dscale, dgamma, dt, trap, output_float32=True
    )


__all__ = ["compute_ddt_dtrap_cute", "compute_shifted_scalar_grads_cute"]
