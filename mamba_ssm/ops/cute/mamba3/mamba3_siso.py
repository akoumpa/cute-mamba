from __future__ import annotations

import math
from functools import cache
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from torch import Tensor
from quack.cute_dsl_utils import torch2cute_dtype_map

from mamba_ssm.ops.cute.mamba3.mamba3_siso_gate import compute_dzdo_cute
from mamba_ssm.ops.cute.mamba3.utils import (
    cos_approx,
    sigmoid_approx,
    silu,
    sin_approx,
    tanh_approx,
)


_LOG2E = 1.4426950408889634
_PI = 3.141592653589793
_TWO_PI = 2.0 * _PI


@cache
def _compile_mamba3_siso_step(
    q_dtype,
    k_dtype,
    v_dtype,
    adt_dtype,
    dt_dtype,
    trap_dtype,
    bias_q_dtype,
    bias_k_dtype,
    angle_proj_dtype,
    d_dtype,
    z_dtype,
    angle_state_dtype,
    ssm_state_dtype,
    k_state_dtype,
    v_state_dtype,
    *,
    qk_dim: int,
    v_dim: int,
    angle_dim: int,
    num_threads: int,
    has_d: bool,
    has_z: bool,
):
    num_warps = num_threads // 32
    qk_items = qk_dim // 32
    v_items = math.ceil(v_dim / num_warps)

    class Mamba3SisoStep:
        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mAdt: cute.Tensor,
            mDt: cute.Tensor,
            mTrap: cute.Tensor,
            mQBias: cute.Tensor,
            mKBias: cute.Tensor,
            mAngles: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mAngleState: cute.Tensor,
            mSsmState: cute.Tensor,
            mKState: cute.Tensor,
            mVState: cute.Tensor,
            mOut: cute.Tensor,
            mAngleOut: cute.Tensor,
            mSsmOut: cute.Tensor,
            mKOut: cute.Tensor,
            batch: Int32,
            nheads: Int32,
            nheads_qk: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                mQ,
                mK,
                mV,
                mAdt,
                mDt,
                mTrap,
                mQBias,
                mKBias,
                mAngles,
                mD,
                mZ,
                mAngleState,
                mSsmState,
                mKState,
                mVState,
                mOut,
                mAngleOut,
                mSsmOut,
                mKOut,
                nheads,
                nheads_qk,
            ).launch(
                grid=[nheads, batch, 1],
                block=[num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mAdt: cute.Tensor,
            mDt: cute.Tensor,
            mTrap: cute.Tensor,
            mQBias: cute.Tensor,
            mKBias: cute.Tensor,
            mAngles: cute.Tensor,
            mD: cute.Tensor | None,
            mZ: cute.Tensor | None,
            mAngleState: cute.Tensor,
            mSsmState: cute.Tensor,
            mKState: cute.Tensor,
            mVState: cute.Tensor,
            mOut: cute.Tensor,
            mAngleOut: cute.Tensor,
            mSsmOut: cute.Tensor,
            mKOut: cute.Tensor,
            nheads: Int32,
            nheads_qk: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            head, batch_idx, _ = cute.arch.block_idx()
            lane = tidx % 32
            warp = tidx // 32
            group_size = nheads // nheads_qk
            qk_head = head // group_size
            token = Int64(batch_idx) * nheads + head
            qk_token = Int64(batch_idx) * nheads_qk + qk_head

            smem = cutlass.utils.SmemAllocator()
            sQ = smem.allocate_tensor(
                q_dtype, cute.make_layout((qk_dim,)), byte_alignment=16
            )
            sK = smem.allocate_tensor(
                k_dtype, cute.make_layout((qk_dim,)), byte_alignment=16
            )

            pair = tidx
            if pair < qk_dim // 2:
                first = pair * 2
                second = first + 1
                q_base = qk_token * qk_dim
                bias_base = Int64(head) * qk_dim
                q0 = Float32(mQ[q_base + first]) + Float32(mQBias[bias_base + first])
                q1 = Float32(mQ[q_base + second]) + Float32(mQBias[bias_base + second])
                k0 = Float32(mK[q_base + first]) + Float32(mKBias[bias_base + first])
                k1 = Float32(mK[q_base + second]) + Float32(mKBias[bias_base + second])
                angle = Float32(0.0)
                if pair < angle_dim:
                    angle_idx = token * angle_dim + pair
                    angle = Float32(mAngleState[angle_idx]) + tanh_approx(
                        Float32(mAngles[angle_idx])
                    ) * Float32(mDt[token]) * Float32(_PI)
                    angle -= Float32(_TWO_PI) * Float32(Int32(angle / Float32(_TWO_PI)))
                    if angle < Float32(0.0):
                        angle += Float32(_TWO_PI)
                    mAngleOut[angle_idx] = Float32(angle)
                cosine = cos_approx(angle)
                sine = sin_approx(angle)
                q0_rot = q0 * cosine - q1 * sine
                q1_rot = q0 * sine + q1 * cosine
                k0_rot = k0 * cosine - k1 * sine
                k1_rot = k0 * sine + k1 * cosine
                sQ[first] = q_dtype(q0_rot)
                sQ[second] = q_dtype(q1_rot)
                sK[first] = k_dtype(k0_rot)
                sK[second] = k_dtype(k1_rot)
                mKOut[token * qk_dim + first] = Float32(sK[first])
                mKOut[token * qk_dim + second] = Float32(sK[second])

            cute.arch.sync_threads()
            alpha = cute.math.exp2(
                Float32(mAdt[token]) * Float32(_LOG2E), fastmath=True
            )
            dt = Float32(mDt[token])
            trap = sigmoid_approx(Float32(mTrap[token]))
            beta = alpha * dt * (Float32(1.0) - trap)
            gamma = trap * dt
            state_base = token * (v_dim * qk_dim)
            vector_base = token * v_dim
            k_state_base = token * qk_dim

            for v_item in cutlass.range_constexpr(v_items):
                value = warp + v_item * num_warps
                if value < v_dim:
                    v = Float32(mV[vector_base + value])
                    v_prev = Float32(mVState[vector_base + value])
                    output = Float32(0.0)
                    for qk_item in cutlass.range_constexpr(qk_items):
                        feature = lane + qk_item * 32
                        state_idx = state_base + Int64(value) * qk_dim + feature
                        state = Float32(mSsmState[state_idx]) * alpha
                        state += (
                            beta * v_prev * Float32(mKState[k_state_base + feature])
                        )
                        state += gamma * v * Float32(sK[feature])
                        mSsmOut[state_idx] = Float32(state)
                        output += Float32(BFloat16(state)) * Float32(
                            BFloat16(Float32(sQ[feature]))
                        )
                    for offset in (16, 8, 4, 2, 1):
                        output += cute.arch.shuffle_sync_down(output, offset)
                    if lane == 0:
                        if const_expr(has_d):
                            output += Float32(mD[head]) * v
                        if const_expr(has_z):
                            output *= silu(Float32(mZ[vector_base + value]))
                        mOut[vector_base + value] = v_dtype(output)

    def fake(dtype):
        return (
            cute.runtime.make_fake_tensor(
                dtype, (cute.sym_int(),), stride=(1,), assumed_align=16
            )
            if dtype is not None
            else None
        )

    return cute.compile(
        Mamba3SisoStep(),
        fake(q_dtype),
        fake(k_dtype),
        fake(v_dtype),
        fake(adt_dtype),
        fake(dt_dtype),
        fake(trap_dtype),
        fake(bias_q_dtype),
        fake(bias_k_dtype),
        fake(angle_proj_dtype),
        fake(d_dtype) if has_d else None,
        fake(z_dtype) if has_z else None,
        fake(angle_state_dtype),
        fake(ssm_state_dtype),
        fake(k_state_dtype),
        fake(v_state_dtype),
        fake(v_dtype),
        fake(Float32),
        fake(Float32),
        fake(Float32),
        Int32(0),
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


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
    _, nheads, headdim_v = V.shape
    if K.shape != Q.shape or nheads % nheads_qk:
        raise AssertionError("invalid Q/K shape or GQA ratio")
    if headdim_qk % 2:
        raise AssertionError("Q/K head dimension must be even")
    if Input_States is None:
        raise TypeError("Input_States must contain angle, SSM, K, and V states")
    angle_state, ssm_state, k_state, v_state = Input_States
    angle_dim = Angles.shape[-1]
    if V.shape[:2] != (batch, nheads):
        raise AssertionError("invalid V shape")
    if ADT.shape != (batch, nheads) or DT.shape != (batch, nheads):
        raise AssertionError("invalid ADT or DT shape")
    if Trap.shape != (batch, nheads):
        raise AssertionError("invalid Trap shape")
    if Q_bias.shape != (nheads, headdim_qk) or K_bias.shape != Q_bias.shape:
        raise AssertionError("invalid Q/K bias shape")
    if angle_dim % 2 or Angles.shape != (batch, nheads, angle_dim):
        raise AssertionError("invalid angle shape")
    if angle_state.shape != Angles.shape:
        raise AssertionError("invalid angle state shape")
    if ssm_state.shape != (batch, nheads, headdim_v, headdim_qk):
        raise AssertionError("invalid SSM state shape")
    if k_state.shape != (batch, nheads, headdim_qk):
        raise AssertionError("invalid K state shape")
    if v_state.shape != V.shape:
        raise AssertionError("invalid V state shape")
    if D is not None and D.shape != (nheads,):
        raise AssertionError("invalid D shape")
    if Z is not None and Z.shape != V.shape:
        raise AssertionError("invalid Z shape")
    supported = (
        Q.is_cuda
        and headdim_qk % 32 == 0
        and headdim_qk <= 256
        and angle_dim <= headdim_qk // 2
    )
    if supported:
        tensors = (
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
            angle_state,
            ssm_state,
            k_state,
            v_state,
        )
        if any(t is not None and not t.is_cuda for t in tensors):
            raise AssertionError("all Mamba3 SISO step tensors must be CUDA tensors")
        (
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
            angle_state,
            ssm_state,
            k_state,
            v_state,
        ) = tuple(t.contiguous() if t is not None else None for t in tensors)
        out = torch.empty_like(V)
        angle_out = torch.empty(angle_state.shape, device=Q.device, dtype=torch.float32)
        ssm_out = torch.empty(ssm_state.shape, device=Q.device, dtype=torch.float32)
        k_out = torch.empty(k_state.shape, device=Q.device, dtype=torch.float32)
        useful_warps = min(32, max(4, 1 << (headdim_v - 1).bit_length()))
        if batch * nheads > 128:
            useful_warps = min(useful_warps, 16)
        num_threads = useful_warps * 32
        compiled = _compile_mamba3_siso_step(
            torch2cute_dtype_map[Q.dtype],
            torch2cute_dtype_map[K.dtype],
            torch2cute_dtype_map[V.dtype],
            torch2cute_dtype_map[ADT.dtype],
            torch2cute_dtype_map[DT.dtype],
            torch2cute_dtype_map[Trap.dtype],
            torch2cute_dtype_map[Q_bias.dtype],
            torch2cute_dtype_map[K_bias.dtype],
            torch2cute_dtype_map[Angles.dtype],
            torch2cute_dtype_map[D.dtype] if D is not None else None,
            torch2cute_dtype_map[Z.dtype] if Z is not None else None,
            torch2cute_dtype_map[angle_state.dtype],
            torch2cute_dtype_map[ssm_state.dtype],
            torch2cute_dtype_map[k_state.dtype],
            torch2cute_dtype_map[v_state.dtype],
            qk_dim=headdim_qk,
            v_dim=headdim_v,
            angle_dim=angle_dim,
            num_threads=num_threads,
            has_d=D is not None,
            has_z=Z is not None,
        )
        compiled(
            Q.view(-1),
            K.view(-1),
            V.view(-1),
            ADT.view(-1),
            DT.view(-1),
            Trap.view(-1),
            Q_bias.view(-1),
            K_bias.view(-1),
            Angles.view(-1),
            D.view(-1) if D is not None else None,
            Z.view(-1) if Z is not None else None,
            angle_state.view(-1),
            ssm_state.view(-1),
            k_state.view(-1),
            v_state.view(-1),
            out.view(-1),
            angle_out.view(-1),
            ssm_out.view(-1),
            k_out.view(-1),
            batch,
            nheads,
            nheads_qk,
        )
        return out, [angle_out, ssm_out, k_out, V]

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
