from __future__ import annotations

import torch
import torch.nn.functional as F


def _pad_to_chunk(x, chunk_size):
    padding = (-x.shape[-1]) % chunk_size
    return F.pad(x, (0, padding)) if padding else x


def compute_dacs_segsum_cute(da, chunk_size):
    batch, nheads, seqlen = da.shape
    padded = _pad_to_chunk(da, chunk_size)
    nchunks = padded.shape[-1] // chunk_size
    chunks = padded.view(batch, nheads, nchunks, chunk_size)
    da_cs_chunks_raw = chunks.cumsum(dim=-1)
    da_cs_chunks = da_cs_chunks_raw.clamp_max(0)
    da_cs_rev_chunks = (
        chunks.sum(dim=-1, keepdim=True) - chunks.cumsum(dim=-1)
    ).clamp_max(0)
    row = da_cs_chunks_raw.unsqueeze(-1)
    col = da_cs_chunks_raw.unsqueeze(-2)
    mask = torch.ones(chunk_size, chunk_size, device=da.device, dtype=torch.bool).tril(
        -1
    )
    segsum = (row - col).clamp_max(0).masked_fill(~mask, 0)
    da_cs = da_cs_chunks.view(batch, nheads, -1)[..., :seqlen]
    da_cs_rev = da_cs_rev_chunks.view(batch, nheads, -1)[..., :seqlen]
    return da_cs, da_cs_rev, segsum


def bwd_dadt_fused_cute(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size):
    batch, nheads, seqlen = ddA_cs.shape
    if seqlen % chunk_size != 0:
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
            bwd_dadt_fused_triton,
        )

        return bwd_dadt_fused_triton(
            dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
        )
    nchunks = seqlen // chunk_size
    shape = (batch, nheads, nchunks, chunk_size)
    term_cs = (dA_cs.exp() * ddA_cs).view(shape).flip(-1).cumsum(-1).flip(-1)
    scaled_rev = (dA_cs_rev.exp() * ddA_cs_rev).view(shape)
    term_rev = F.pad(scaled_rev.cumsum(-1)[..., :-1], (1, 0))
    weighted = dSSdA.transpose(-1, -2).float() * SSdA.float().exp()
    mask = torch.ones(
        chunk_size, chunk_size, device=dSSdA.device, dtype=torch.bool
    ).tril(-1)
    term_seg = weighted.flip(-2).cumsum(-2).flip(-2).masked_fill(~mask, 0).sum(-1)
    return (term_cs.float() + term_rev.float() + term_seg).view(batch, nheads, seqlen)


def bwd_dtrap_ddt_cute(trap, dt, dfactor, dgamma_diag, chunk_size):
    del chunk_size
    from mamba_ssm.ops.cute.mamba3.mamba3_siso_scalar_bwd import (
        compute_shifted_scalar_grads_cute,
    )

    return compute_shifted_scalar_grads_cute(dfactor, dgamma_diag, dt, trap)


def _sequence_ranges(cu_seqlens):
    offsets = cu_seqlens.detach().cpu().tolist()
    return list(zip(offsets[:-1], offsets[1:]))


def compute_dacs_segsum_cute_varlen(da, chunk_size, cu_seqlens=None):
    if cu_seqlens is None or cu_seqlens.numel() <= 1:
        return compute_dacs_segsum_cute(da, chunk_size)
    batch, nheads, seqlen = da.shape
    ranges = _sequence_ranges(cu_seqlens)
    nchunks_global = seqlen // chunk_size + len(ranges)
    da_cs = torch.empty_like(da)
    da_cs_rev = torch.empty_like(da)
    segsum = torch.zeros(
        batch,
        nheads,
        nchunks_global,
        chunk_size,
        chunk_size,
        device=da.device,
        dtype=da.dtype,
    )
    for sequence, (start, end) in enumerate(ranges):
        length = end - start
        chunks = length // chunk_size + 1
        padded = F.pad(da[..., start:end], (0, chunks * chunk_size - length))
        seq_cs, seq_rev, seq_seg = compute_dacs_segsum_cute(padded, chunk_size)
        if length:
            da_cs[..., start:end] = seq_cs[..., :length]
            da_cs_rev[..., start:end] = seq_rev[..., :length]
        global_start = start // chunk_size + sequence
        segsum[..., global_start : global_start + chunks, :, :] = seq_seg
    return da_cs, da_cs_rev, segsum


def bwd_dadt_fused_cute_varlen(
    dSSdA,
    SSdA,
    ddA_cs,
    ddA_cs_rev,
    dA_cs,
    dA_cs_rev,
    chunk_size,
    cu_seqlens,
):
    batch, nheads, seqlen = ddA_cs.shape
    out = torch.zeros(batch, nheads, seqlen, device=ddA_cs.device, dtype=torch.float32)
    for sequence, (start, end) in enumerate(_sequence_ranges(cu_seqlens)):
        length = end - start
        if not length:
            continue
        chunks = length // chunk_size + 1
        padded_length = chunks * chunk_size

        def pad(x):
            return F.pad(x[..., start:end], (0, padded_length - length))

        global_start = start // chunk_size + sequence
        seq_out = bwd_dadt_fused_cute(
            dSSdA[..., global_start : global_start + chunks, :, :],
            SSdA[..., global_start : global_start + chunks, :, :],
            pad(ddA_cs),
            pad(ddA_cs_rev),
            pad(dA_cs),
            pad(dA_cs_rev),
            chunk_size,
        )
        out[..., start:end] = seq_out[..., :length]
    return out


def bwd_dtrap_ddt_cute_varlen(trap, dt, dfactor, dgamma_diag, chunk_size, cu_seqlens):
    ddt = torch.zeros_like(dt)
    dtrap = torch.zeros_like(trap)
    for start, end in _sequence_ranges(cu_seqlens):
        if start == end:
            continue
        seq_ddt, seq_dtrap = bwd_dtrap_ddt_cute(
            trap[..., start:end],
            dt[..., start:end],
            dfactor[..., start:end],
            dgamma_diag[..., start:end],
            chunk_size,
        )
        ddt[..., start:end] = seq_ddt
        dtrap[..., start:end] = seq_dtrap
    return ddt, dtrap


__all__ = [
    "bwd_dadt_fused_cute",
    "bwd_dadt_fused_cute_varlen",
    "bwd_dtrap_ddt_cute",
    "bwd_dtrap_ddt_cute_varlen",
    "compute_dacs_segsum_cute",
    "compute_dacs_segsum_cute_varlen",
]
