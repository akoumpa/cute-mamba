# Copyright (c) 2024, Tri Dao, Albert Gu.

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from quack.activation import dswiglu, swiglu
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.packing import bf16x2_to_i32, i32_to_bf16x2


_BLOCK_SIZE = 2048
_VALUES_PER_ITEM = 2


@cache
def _compile_swiglu_fwd(input_dtype, num_threads: int, pack_bf16: bool):
    values_per_item = 4 if pack_bf16 else _VALUES_PER_ITEM
    items_per_thread = _BLOCK_SIZE // (num_threads * values_per_item)

    class SwiGLUForward:
        @cute.jit
        def __call__(
            self,
            mXY: cute.Tensor,
            mOut: cute.Tensor,
            nrows: Int32,
            ncols: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mXY, mOut, nrows, ncols).launch(
                grid=[nrows, cute.ceil_div(ncols, _BLOCK_SIZE), 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mXY: cute.Tensor,
            mOut: cute.Tensor,
            nrows: Int32,
            ncols: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, block_col, _ = cute.arch.block_idx()
            block_base = Int64(block_col) * _BLOCK_SIZE
            row_base = Int64(row) * (Int64(2) * ncols)
            out_base = Int64(row) * ncols

            if cutlass.const_expr(pack_bf16):
                mXYPacked = cute.recast_tensor(mXY, Int32)
                mOutPacked = cute.recast_tensor(mOut, Int32)
            for item in cutlass.range_constexpr(items_per_thread):
                col = block_base + (tidx + item * num_threads) * values_per_item
                if col < ncols:
                    xy_offset = row_base + col
                    if cutlass.const_expr(pack_bf16):
                        x_raw, x_next_raw = i32_to_bf16x2(mXYPacked[xy_offset // 2])
                        x2_raw, x3_raw = i32_to_bf16x2(mXYPacked[xy_offset // 2 + 1])
                        y_raw, y_next_raw = i32_to_bf16x2(
                            mXYPacked[(xy_offset + ncols) // 2]
                        )
                        y2_raw, y3_raw = i32_to_bf16x2(
                            mXYPacked[(xy_offset + ncols) // 2 + 1]
                        )
                        packed_x0 = Float32(x_raw)
                        packed_x1 = Float32(x_next_raw)
                        packed_x2 = Float32(x2_raw)
                        packed_x3 = Float32(x3_raw)
                        packed_y0 = Float32(y_raw)
                        packed_y1 = Float32(y_next_raw)
                        packed_y2 = Float32(y2_raw)
                        packed_y3 = Float32(y3_raw)
                        mOutPacked[(out_base + col) // 2] = bf16x2_to_i32(
                            input_dtype(swiglu(packed_x0, packed_y0)),
                            input_dtype(swiglu(packed_x1, packed_y1)),
                        )
                        mOutPacked[(out_base + col) // 2 + 1] = bf16x2_to_i32(
                            input_dtype(swiglu(packed_x2, packed_y2)),
                            input_dtype(swiglu(packed_x3, packed_y3)),
                        )
                    else:
                        x = Float32(mXY[xy_offset])
                        y = Float32(mXY[xy_offset + ncols])
                        if col + 1 < ncols:
                            x_next = Float32(mXY[xy_offset + 1])
                            y_next = Float32(mXY[xy_offset + ncols + 1])
                            mOut[out_base + col] = input_dtype(swiglu(x, y))
                            mOut[out_base + col + 1] = input_dtype(
                                swiglu(x_next, y_next)
                            )
                        else:
                            mOut[out_base + col] = input_dtype(swiglu(x, y))

    xy_fake = cute.runtime.make_fake_tensor(
        input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    out_fake = cute.runtime.make_fake_tensor(
        input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    return cute.compile(
        SwiGLUForward(),
        xy_fake,
        out_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@cache
def _compile_swiglu_bwd(
    input_dtype,
    grad_dtype,
    recompute_output: bool,
    num_threads: int,
    pack_bf16: bool,
):
    items_per_thread = _BLOCK_SIZE // (num_threads * _VALUES_PER_ITEM)

    class SwiGLUBackward:
        @cute.jit
        def __call__(
            self,
            mXY: cute.Tensor,
            mDOut: cute.Tensor,
            mDXY: cute.Tensor,
            mOut: cute.Tensor | None,
            nrows: Int32,
            ncols: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(mXY, mDOut, mDXY, mOut, nrows, ncols).launch(
                grid=[nrows, cute.ceil_div(ncols, _BLOCK_SIZE), 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mXY: cute.Tensor,
            mDOut: cute.Tensor,
            mDXY: cute.Tensor,
            mOut: cute.Tensor | None,
            nrows: Int32,
            ncols: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            row, block_col, _ = cute.arch.block_idx()
            block_base = Int64(block_col) * _BLOCK_SIZE
            row_base = Int64(row) * (Int64(2) * ncols)
            out_base = Int64(row) * ncols

            if cutlass.const_expr(pack_bf16):
                mXYPacked = cute.recast_tensor(mXY, Int32)
                mDOutPacked = cute.recast_tensor(mDOut, Int32)
                mDXYPacked = cute.recast_tensor(mDXY, Int32)
                if cutlass.const_expr(recompute_output):
                    mOutPacked = cute.recast_tensor(mOut, Int32)
            for item in cutlass.range_constexpr(items_per_thread):
                col = block_base + (tidx + item * num_threads) * _VALUES_PER_ITEM
                if col < ncols:
                    xy_offset = row_base + col
                    out_offset = out_base + col
                    if cutlass.const_expr(pack_bf16):
                        x_raw, x_next_raw = i32_to_bf16x2(mXYPacked[xy_offset // 2])
                        y_raw, y_next_raw = i32_to_bf16x2(
                            mXYPacked[(xy_offset + ncols) // 2]
                        )
                        dout_raw, dout_next_raw = i32_to_bf16x2(
                            mDOutPacked[out_offset // 2]
                        )
                        packed_x0 = Float32(x_raw)
                        packed_x1 = Float32(x_next_raw)
                        packed_y0 = Float32(y_raw)
                        packed_y1 = Float32(y_next_raw)
                        packed_dout0 = Float32(dout_raw)
                        packed_dout1 = Float32(dout_next_raw)
                        packed_dx0, packed_dy0, packed_out0 = dswiglu(
                            packed_x0, packed_y0, packed_dout0
                        )
                        packed_dx1, packed_dy1, packed_out1 = dswiglu(
                            packed_x1, packed_y1, packed_dout1
                        )
                        mDXYPacked[xy_offset // 2] = bf16x2_to_i32(
                            input_dtype(packed_dx0), input_dtype(packed_dx1)
                        )
                        mDXYPacked[(xy_offset + ncols) // 2] = bf16x2_to_i32(
                            input_dtype(packed_dy0), input_dtype(packed_dy1)
                        )
                        if cutlass.const_expr(recompute_output):
                            mOutPacked[out_offset // 2] = bf16x2_to_i32(
                                input_dtype(packed_out0), input_dtype(packed_out1)
                            )
                    else:
                        x = Float32(mXY[xy_offset])
                        y = Float32(mXY[xy_offset + ncols])
                        dout = Float32(mDOut[out_offset])
                        if col + 1 < ncols:
                            x_next = Float32(mXY[xy_offset + 1])
                            y_next = Float32(mXY[xy_offset + ncols + 1])
                            dout_next = Float32(mDOut[out_offset + 1])
                            dx, dy, out = dswiglu(x, y, dout)
                            dx_next, dy_next, out_next = dswiglu(
                                x_next, y_next, dout_next
                            )
                            mDXY[xy_offset] = input_dtype(dx)
                            mDXY[xy_offset + 1] = input_dtype(dx_next)
                            mDXY[xy_offset + ncols] = input_dtype(dy)
                            mDXY[xy_offset + ncols + 1] = input_dtype(dy_next)
                            if cutlass.const_expr(recompute_output):
                                mOut[out_offset] = input_dtype(out)
                                mOut[out_offset + 1] = input_dtype(out_next)
                        else:
                            dx, dy, out = dswiglu(x, y, dout)
                            mDXY[xy_offset] = input_dtype(dx)
                            mDXY[xy_offset + ncols] = input_dtype(dy)
                            if cutlass.const_expr(recompute_output):
                                mOut[out_offset] = input_dtype(out)

    xy_fake = cute.runtime.make_fake_tensor(
        input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    dout_fake = cute.runtime.make_fake_tensor(
        grad_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    dxy_fake = cute.runtime.make_fake_tensor(
        input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
    )
    out_fake = (
        cute.runtime.make_fake_tensor(
            input_dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
        )
        if recompute_output
        else None
    )
    return cute.compile(
        SwiGLUBackward(),
        xy_fake,
        dout_fake,
        dxy_fake,
        out_fake,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _swiglu_fwd(xy: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    if xy.shape[-1] % 2 != 0:
        raise ValueError("SwiGLU input's last dimension must be even")
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    nrows, twice_ncols = xy.shape
    ncols = twice_ncols // 2
    if out is None:
        out = torch.empty((*batch_shape, ncols), dtype=xy.dtype, device=xy.device)
    out_2d = out.reshape(-1, out.shape[-1])
    if out_2d.shape != (nrows, ncols) or out_2d.stride(-1) != 1:
        raise ValueError(
            "SwiGLU output must have the input batch shape and a contiguous last dimension"
        )
    if not (xy.dtype == torch.bfloat16 and nrows == 8192 and ncols == 4096):
        from mamba_ssm.ops.triton.k_activations import _swiglu_fwd as triton_fwd

        return triton_fwd(xy, out_2d).reshape(*batch_shape, ncols)
    num_threads = 512 if 512 <= ncols <= 1024 else 256
    compiled = _compile_swiglu_fwd(
        torch2cute_dtype_map[xy.dtype],
        num_threads,
        xy.dtype == torch.bfloat16 and ncols % 4 == 0,
    )
    compiled(xy.view(-1), out_2d.view(-1), nrows, ncols)
    return out_2d.reshape(*batch_shape, ncols)


def _swiglu_bwd(
    xy: torch.Tensor,
    dout: torch.Tensor,
    dxy: torch.Tensor | None = None,
    recompute_output: bool = False,
    out: torch.Tensor | None = None,
):
    if xy.shape[-1] % 2 != 0:
        raise ValueError("SwiGLU input's last dimension must be even")
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    nrows, twice_ncols = xy.shape
    ncols = twice_ncols // 2
    dout = dout.reshape(-1, dout.shape[-1])
    if dout.shape != (nrows, ncols):
        raise ValueError("SwiGLU output gradient shape does not match the input")
    if dxy is None:
        dxy = torch.empty_like(xy)
    dxy_2d = dxy.reshape(-1, dxy.shape[-1])
    if dxy_2d.shape != xy.shape or dxy_2d.stride(-1) != 1:
        raise ValueError(
            "SwiGLU input gradient must match the input with a contiguous last dimension"
        )
    if recompute_output:
        if out is None:
            out = torch.empty((*batch_shape, ncols), dtype=xy.dtype, device=xy.device)
        out_2d = out.reshape(-1, out.shape[-1])
        if out_2d.shape != (nrows, ncols) or out_2d.stride(-1) != 1:
            raise ValueError("SwiGLU recomputed output has an invalid shape or stride")
    else:
        out_2d = None
    compiled = _compile_swiglu_bwd(
        torch2cute_dtype_map[xy.dtype],
        torch2cute_dtype_map[dout.dtype],
        recompute_output,
        512 if 512 <= ncols <= 1024 else 256,
        xy.dtype == dout.dtype == torch.bfloat16 and ncols % 2 == 0,
    )
    compiled(
        xy.view(-1),
        dout.view(-1),
        dxy_2d.view(-1),
        out_2d.view(-1) if out_2d is not None else None,
        nrows,
        ncols,
    )
    dxy = dxy_2d.reshape(*batch_shape, twice_ncols)
    if not recompute_output:
        return dxy
    return dxy, out_2d.reshape(*batch_shape, ncols)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy):
        ctx.save_for_backward(xy)
        return _swiglu_fwd(xy)

    @staticmethod
    def backward(ctx, dout):
        (xy,) = ctx.saved_tensors
        return _swiglu_bwd(xy, dout)


swiglu_fn = SwiGLU.apply


__all__ = ["SwiGLU", "_swiglu_bwd", "_swiglu_fwd", "swiglu_fn"]
