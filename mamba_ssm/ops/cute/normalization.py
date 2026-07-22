from __future__ import annotations

import torch
import torch.nn.functional as F


def layer_norm_fn_cute(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
    return_dropout_mask=False,
):
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn

    return layer_norm_fn(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        return_dropout_mask,
    )


def layer_norm_linear_fn_cute(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    from mamba_ssm.ops.triton.layer_norm import layer_norm_linear_fn

    return layer_norm_linear_fn(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )


def _gated_norm(x, weight, bias, z, eps, group_size, norm_before_gate, *, is_rms_norm):
    hidden_size = x.shape[-1]
    group_size = hidden_size if group_size is None else group_size
    if hidden_size % group_size:
        raise AssertionError("hidden size must be divisible by group size")
    gated_x = x * F.silu(z) if z is not None and not norm_before_gate else x
    groups = gated_x.reshape(*gated_x.shape[:-1], hidden_size // group_size, group_size)
    if is_rms_norm:
        inverse_rms = torch.rsqrt(
            groups.float().square().mean(dim=-1, keepdim=True) + eps
        )
        normalized = groups.float() * inverse_rms
    else:
        normalized = F.layer_norm(groups.float(), (group_size,), eps=eps)
    output = normalized.reshape_as(gated_x).to(x.dtype) * weight
    if bias is not None:
        output = output + bias
    if z is not None and norm_before_gate:
        output = output * F.silu(z)
    return output.to(x.dtype)


def layernorm_fn_cute(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    return _gated_norm(
        x,
        weight,
        bias,
        z,
        eps,
        group_size,
        norm_before_gate,
        is_rms_norm=is_rms_norm,
    )


def rmsnorm_fn_cute(
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
):
    if z is None and group_size is None:
        from mamba_ssm.ops.triton.layer_norm import rms_norm_fn

        return rms_norm_fn(x, weight, bias, eps=eps)
    return _gated_norm(
        x,
        weight,
        bias,
        z,
        eps,
        group_size,
        norm_before_gate,
        is_rms_norm=True,
    )


__all__ = [
    "layer_norm_fn_cute",
    "layer_norm_linear_fn_cute",
    "layernorm_fn_cute",
    "rmsnorm_fn_cute",
]
