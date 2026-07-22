from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm.ops.cute.mamba3.mamba3_siso_gate import compute_dzdo_cute


def _rotate_pairwise(x, angles):
    pairs = x.float().view(*x.shape[:-1], x.shape[-1] // 2, 2)
    rotary_pairs = angles.shape[-1]
    if rotary_pairs < pairs.shape[-2]:
        angles = F.pad(angles, (0, pairs.shape[-2] - rotary_pairs))
    cosine, sine = torch.cos(angles), torch.sin(angles)
    even, odd = pairs.unbind(dim=-1)
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def mamba3_siso_step_cute(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    batch, nheads_qk, headdim_qk = Q.shape
    _, nheads, _ = V.shape
    if K.shape != Q.shape or nheads % nheads_qk:
        raise AssertionError("invalid Q/K shape or GQA ratio")
    if headdim_qk % 2:
        raise AssertionError("Q/K head dimension must be even")
    if Input_States is None:
        raise TypeError("Input_States must contain angle, SSM, K, and V states")
    angle_state, ssm_state, k_state, v_state = Input_States
    group_size = nheads // nheads_qk
    q = Q.repeat_interleave(group_size, dim=1).float() + Q_bias.float()
    k = K.repeat_interleave(group_size, dim=1).float() + K_bias.float()
    angle_out = torch.remainder(
        angle_state.float()
        + torch.tanh(Angles.float()) * DT.float().unsqueeze(-1) * math.pi,
        2 * math.pi,
    )
    q_rot = _rotate_pairwise(q, angle_out).to(Q.dtype)
    k_rot = _rotate_pairwise(k, angle_out).to(K.dtype)

    alpha = torch.exp(ADT.float())
    trap = torch.sigmoid(Trap.float())
    beta = alpha * DT.float() * (1 - trap)
    gamma = trap * DT.float()
    state_out = ssm_state.float() * alpha[..., None, None]
    state_out = state_out + torch.einsum(
        "bhv,bhq->bhvq", beta[..., None] * v_state.float(), k_state.float()
    )
    state_out = state_out + torch.einsum(
        "bhv,bhq->bhvq", gamma[..., None] * V.float(), k_rot.float()
    )
    out = (
        torch.matmul(
            state_out.to(torch.bfloat16), q_rot.to(torch.bfloat16).unsqueeze(-1)
        )
        .squeeze(-1)
        .float()
    )
    if D is not None:
        out = out + D.float()[None, :, None] * V.float()
    if Z is not None:
        out = out * F.silu(Z.float())
    return out.to(V.dtype), [angle_out, state_out, k_rot.float(), V]


def mamba3_siso_fwd_cute(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    store_states_adt_outv: bool = False,
    return_final_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd

    return mamba3_siso_fwd(
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        Angles,
        D,
        Z,
        Initial_States,
        chunk_size,
        store_states_adt_outv,
        return_final_states,
        cu_seqlens,
    )


def compute_dqkv_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_sum: torch.Tensor,
    qk_dot: torch.Tensor,
    SSM_States: torch.Tensor,
    do: torch.Tensor,
    d_ossm_state: Optional[torch.Tensor] = None,
    d_ov_state: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    has_input_state: bool = False,
    Cu_Seqlens: Optional[torch.Tensor] = None,
):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dqkv

    return compute_dqkv(
        q,
        k,
        v,
        da_cs,
        da_cs_sum,
        qk_dot,
        SSM_States,
        do,
        d_ossm_state,
        d_ov_state,
        D,
        chunk_size,
        has_input_state,
        Cu_Seqlens,
    )


def compute_dqktheta_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: torch.Tensor,
    gamma: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    angles: torch.Tensor,
    dq_in: torch.Tensor,
    dk_in: torch.Tensor,
    dqk: torch.Tensor,
    d_ok_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    Cu_Seqlens: Optional[torch.Tensor] = None,
):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dqktheta

    return compute_dqktheta(
        q,
        k,
        scale,
        gamma,
        q_bias,
        k_bias,
        angles,
        dq_in,
        dk_in,
        dqk,
        d_ok_state,
        chunk_size,
        Cu_Seqlens,
    )


def apply_dk_state_post_cute(
    d_ok_state: torch.Tensor,
    angles: torch.Tensor,
    k: torch.Tensor,
    k_bias: torch.Tensor,
    dk: torch.Tensor,
    dk_bias: torch.Tensor,
    dangles: torch.Tensor,
    Cu_Seqlens: Optional[torch.Tensor] = None,
):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import apply_dk_state_post

    return apply_dk_state_post(
        d_ok_state, angles, k, k_bias, dk, dk_bias, dangles, Cu_Seqlens
    )


def compute_ddt_dtrap_dinput_states_cute(
    dscale: torch.Tensor,
    dgamma: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    d_issm_state: Optional[torch.Tensor] = None,
    input_k_state: Optional[torch.Tensor] = None,
    input_v_state: Optional[torch.Tensor] = None,
    Cu_Seqlens: Optional[torch.Tensor] = None,
):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import (
        compute_ddt_dtrap_dinput_states,
    )

    return compute_ddt_dtrap_dinput_states(
        dscale,
        dgamma,
        dt,
        trap,
        d_issm_state,
        input_k_state,
        input_v_state,
        Cu_Seqlens,
    )


def mamba3_siso_combined_cute(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    chunk_size: int = 64,
    return_final_states: bool = False,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        mamba3_siso_combined,
    )

    return mamba3_siso_combined(
        Q,
        K,
        V,
        ADT,
        DT,
        Trap,
        Q_bias,
        K_bias,
        Angles,
        D,
        Z,
        Input_States,
        chunk_size,
        return_final_states,
        cu_seqlens,
    )


__all__ = [
    "apply_dk_state_post_cute",
    "compute_ddt_dtrap_dinput_states_cute",
    "compute_dqktheta_cute",
    "compute_dqkv_cute",
    "compute_dzdo_cute",
    "mamba3_siso_combined_cute",
    "mamba3_siso_fwd_cute",
    "mamba3_siso_step_cute",
]
