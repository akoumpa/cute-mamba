# Copyright (c) 2026, NVIDIA CORPORATION.

from __future__ import annotations

import torch
import torch.nn.functional as F

from einops import rearrange

from mamba_ssm.ops.cute.ssd_chunk_scan import chunk_scan_bwd_dx_cute, chunk_scan_cute
from mamba_ssm.ops.cute.ssd_chunk_state import chunk_state_bwd_dx_cute, chunk_state_cute
from mamba_ssm.ops.cute.ssd_state_passing import state_passing_cute


def _needs_backward(*tensors):
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def chunk_scan_chunk_state_bwd_dx_cute(
    x,
    dt,
    dA_cumsum,
    B,
    CB,
    dout,
    dstates,
    D=None,
    seq_idx=None,
    dx=None,
):
    """Compute the combined scan and chunk-state input gradients."""
    tensors = (x, dt, dA_cumsum, B, CB, dout, dstates, D, dx)
    if seq_idx is not None or any(
        tensor is not None and (not tensor.is_cuda or not tensor.is_contiguous())
        for tensor in tensors
    ):
        from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx

        return _chunk_scan_chunk_state_bwd_dx(
            x, dt, dA_cumsum, B, CB, dout, dstates, D, seq_idx, dx
        )

    dx_scan, ddt_scan = chunk_scan_bwd_dx_cute(CB, x, dt, dA_cumsum, dout, D)
    dx_state, ddt_state, _ = chunk_state_bwd_dx_cute(B, x, dt, dA_cumsum, dstates)
    dx_result = dx_scan + dx_state
    if dx is not None:
        if dx.shape != x.shape:
            raise AssertionError("dx must have the same shape as x")
        dx.copy_(dx_result)
        dx_result = dx

    ddt = ddt_scan + ddt_state
    if D is None:
        dD = None
    elif D.ndim == 2:
        dD = (dout.float() * x.float()).sum(dim=(0, 1)).to(D.dtype)
    else:
        dD = (dout.float() * x.float()).sum(dim=(0, 1, 3)).to(D.dtype)
    return dx_result, ddt, dD


def _mamba_chunk_scan_fwd_cute(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    seqlen = x.shape[1]
    dstate = B.shape[-1]
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> 1 1 h")
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dA_cumsum = torch.cumsum(dt * rearrange(A, "h -> h 1 1"), dim=-1)
    states = chunk_state_cute(B, x, dt, dA_cumsum, states_in_fp32=True)
    states, final_states = state_passing_cute(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[..., -1].contiguous(),
        rearrange(initial_states, "... p n -> ... (p n)").contiguous()
        if initial_states is not None
        else None,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
    out = chunk_scan_cute(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out, final_states


def mamba_chunk_scan_cute(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
):
    if not _needs_backward(x, dt, A, B, C, D, z, dt_bias):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan

        return mamba_chunk_scan(x, dt, A, B, C, chunk_size, D, z, dt_bias, dt_softplus)
    return _mamba_chunk_scan_fwd_cute(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        dt_softplus=dt_softplus,
    )[0]


def mamba_chunk_scan_combined_cute(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False,
    state_dtype=None,
):
    if (
        seq_idx is not None
        or cu_seqlens is not None
        or return_varlen_states
        or state_dtype is not None
        or not _needs_backward(x, dt, A, B, C, D, z, dt_bias, initial_states)
    ):
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        return mamba_chunk_scan_combined(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            initial_states,
            seq_idx,
            cu_seqlens,
            dt_softplus,
            dt_limit,
            return_final_states,
            return_varlen_states,
            state_dtype,
        )
    out, final_states = _mamba_chunk_scan_fwd_cute(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        dt_softplus,
        dt_limit,
    )
    return (out, final_states) if return_final_states else out


def selective_scan_bwd_cute(dout, x, dt, A, B, C, D=None, z=None):
    if dt.ndim != 4 or A.ndim != 1:
        from mamba_ssm.ops.triton.ssd_combined import selective_scan_bwd

        return selective_scan_bwd(dout, x, dt, A, B, C, D, z)

    chunk_size = dt.shape[-1]
    seqlen = x.shape[1]
    with torch.enable_grad():
        dt_var = dt.detach().requires_grad_(True)
        A_var = A.detach().requires_grad_(True)
        dt_tokens = rearrange(dt_var, "b h c l -> b (c l) h")[:, :seqlen]
        out = mamba_chunk_scan_cute(
            x.detach(),
            dt_tokens,
            A_var,
            B.detach(),
            C.detach(),
            chunk_size,
            D.detach() if D is not None else None,
            z.detach() if z is not None else None,
        )
        ddt, dA = torch.autograd.grad(out, (dt_var, A_var), dout)
    return ddt, dA


def mamba_split_conv1d_scan_combined_cute(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
    state_dtype=None,
):
    if (
        seq_idx is not None
        or rmsnorm_weight is not None
        or outproj_weight is not None
        or outproj_bias is not None
        or state_dtype is not None
    ):
        from mamba_ssm.ops.triton.ssd_combined import (
            mamba_split_conv1d_scan_combined,
        )

        return mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D,
            chunk_size,
            initial_states,
            seq_idx,
            dt_limit,
            return_final_states,
            activation,
            rmsnorm_weight,
            rmsnorm_eps,
            outproj_weight,
            outproj_bias,
            headdim,
            ngroups,
            norm_before_gate,
            state_dtype,
        )

    from causal_conv1d import causal_conv1d_fn

    if activation not in (None, "silu", "swish"):
        raise AssertionError("activation must be None, silu, or swish")
    nheads = D.shape[0]
    if D.ndim == 1:
        if headdim is None:
            raise AssertionError("headdim is required when D is one-dimensional")
    else:
        headdim = D.shape[1]
    dim = nheads * headdim
    dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
    d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
    if d_nonssm < 0 or nheads % ngroups != 0:
        raise AssertionError("incompatible fused conv/scan dimensions")
    zx0, z, xBC, dt = torch.split(
        zxbcdt,
        [2 * d_nonssm, dim, dim + 2 * ngroups * dstate, nheads],
        dim=-1,
    )
    xBC = rearrange(
        causal_conv1d_fn(
            rearrange(xBC.contiguous(), "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            activation=activation,
        ),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    scan = mamba_chunk_scan_combined_cute(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D,
        z,
        dt_bias,
        initial_states,
        dt_softplus=True,
        dt_limit=dt_limit,
        return_final_states=return_final_states,
    )
    if return_final_states:
        out, final_states = scan
    else:
        out = scan
    out = rearrange(out, "b s h p -> b s (h p)")
    if d_nonssm > 0:
        left, right = zx0.chunk(2, dim=-1)
        out = torch.cat([F.silu(left) * right, out], dim=-1)
    return (out, final_states) if return_final_states else out


__all__ = [
    "chunk_scan_chunk_state_bwd_dx_cute",
    "mamba_chunk_scan_combined_cute",
    "mamba_chunk_scan_cute",
    "mamba_split_conv1d_scan_combined_cute",
    "selective_scan_bwd_cute",
]
