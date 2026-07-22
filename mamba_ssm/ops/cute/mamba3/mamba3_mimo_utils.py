from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_dacs_segsum_cute(da, chunk_size):
    from mamba_ssm.ops.cute.mamba3.dacs_segsum import (
        compute_dacs_segsum_fixed_cute,
    )

    batch, nheads, seqlen = da.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    work = batch * nheads * nchunks
    use_cute = seqlen % chunk_size != 0
    if da.dtype == torch.bfloat16:
        threshold = 512 if chunk_size <= 16 else 1024 if chunk_size <= 32 else 128
        use_cute |= chunk_size <= 64 and work >= threshold
    elif da.dtype == torch.float32:
        use_cute |= chunk_size == 64 and work >= 1024
    if use_cute:
        return compute_dacs_segsum_fixed_cute(da, chunk_size)

    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
        compute_dacs_segsum_triton,
    )

    return compute_dacs_segsum_triton(da, chunk_size)


def bwd_dadt_fused_cute(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size):
    seqlen = ddA_cs.shape[-1]
    use_cute = (
        seqlen % chunk_size == 0
        and chunk_size <= 64
        and ddA_cs.dtype in (torch.float16, torch.bfloat16)
    )
    if not use_cute:
        from mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils import (
            bwd_dadt_fused_triton,
        )

        return bwd_dadt_fused_triton(
            dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev, chunk_size
        )
    from mamba_ssm.ops.cute.mamba3.dadt_backward import bwd_dadt_fixed_cute

    return bwd_dadt_fixed_cute(dSSdA, SSdA, ddA_cs, ddA_cs_rev, dA_cs, dA_cs_rev)


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
    from mamba_ssm.ops.cute.mamba3.dacs_segsum import (
        compute_dacs_segsum_varlen_cute,
    )

    return compute_dacs_segsum_varlen_cute(da, chunk_size, cu_seqlens)


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
    del chunk_size
    from mamba_ssm.ops.cute.mamba3.mamba3_siso_scalar_bwd import (
        compute_shifted_scalar_grads_cute,
    )

    return compute_shifted_scalar_grads_cute(
        dfactor, dgamma_diag, dt, trap, cu_seqlens=cu_seqlens
    )


__all__ = [
    "bwd_dadt_fused_cute",
    "bwd_dadt_fused_cute_varlen",
    "bwd_dtrap_ddt_cute",
    "bwd_dtrap_ddt_cute_varlen",
    "compute_dacs_segsum_cute",
    "compute_dacs_segsum_cute_varlen",
]
