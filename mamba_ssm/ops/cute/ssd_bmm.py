# Copyright (c) 2024, Tri Dao, Albert Gu.

from __future__ import annotations

import torch


def _can_use_cute_bmm(a, chunk_size, *, seq_idx=None):
    return (
        a.is_cuda
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.stride(-1) == 1
        and a.shape[1] % chunk_size == 0
        and seq_idx is None
        and (a.dim() == 3 or (a.dim() == 4 and a.shape[2] == 1))
    )


def _chunk_view(x, chunk_size):
    batch, seqlen = x.shape[:2]
    x = x if x.dim() == 3 else x.squeeze(2)
    return x.view(batch * (seqlen // chunk_size), chunk_size, x.shape[-1])


def _bmm_chunk_fwd_cute(
    a,
    b,
    chunk_size,
    seq_idx=None,
    causal=False,
    output_dtype=None,
):
    if not _can_use_cute_bmm(a, chunk_size, seq_idx=seq_idx) or not _can_use_cute_bmm(
        b, chunk_size, seq_idx=seq_idx
    ):
        from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd

        return _bmm_chunk_fwd(a, b, chunk_size, seq_idx, causal, output_dtype)

    from quack.gemm_interface import gemm

    batch, seqlen = a.shape[:2]
    nchunks = seqlen // chunk_size
    a_chunks = _chunk_view(a, chunk_size)
    b_chunks_t = _chunk_view(b, chunk_size).transpose(-1, -2)
    out = gemm(
        a_chunks,
        b_chunks_t,
        out_dtype=a.dtype if output_dtype is None else output_dtype,
        tuned=False,
    )
    out = out.view(batch, nchunks, chunk_size, chunk_size)
    return out if a.dim() == 3 else out.unsqueeze(2)


def _bmm_chunk_bwd_cute(a, dout, residual=None, out=None):
    chunk_size = dout.shape[-1]
    if not _can_use_cute_bmm(a, chunk_size):
        from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_bwd

        return _bmm_chunk_bwd(a, dout, residual, out)

    from quack.gemm_interface import gemm

    batch, seqlen = a.shape[:2]
    nchunks = seqlen // chunk_size
    a_chunks = _chunk_view(a, chunk_size)
    dout_chunks = dout if dout.dim() == 4 else dout.squeeze(2)
    dout_chunks = dout_chunks.reshape(batch * nchunks, chunk_size, chunk_size)
    out_view = None
    if out is not None:
        out_view = _chunk_view(out, chunk_size)
    result = gemm(
        dout_chunks.transpose(-1, -2),
        a_chunks,
        out=out_view,
        tuned=False,
    )
    result = result.view(batch, seqlen, a.shape[-1])
    if a.dim() == 4:
        result = result.unsqueeze(2)
    if residual is not None:
        result.add_(residual)
    return result


__all__ = ["_bmm_chunk_bwd_cute", "_bmm_chunk_fwd_cute"]
