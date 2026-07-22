from __future__ import annotations

import torch
import torch.nn.functional as F


def _can_use_quack_rmsnorm(x, weight, bias, *, gated_api):
    if (
        not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.shape[-1] not in (256, 2048, 4096)
        or weight.shape != (x.shape[-1],)
        or (bias is not None and bias.shape != weight.shape)
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
        or (bias is not None and bias.stride(-1) != 1)
    ):
        return False
    rows = x.numel() // x.shape[-1]
    if gated_api:
        return x.dtype == torch.bfloat16 and x.shape[-1] == 256 and rows >= 512
    return x.dtype == torch.bfloat16 and (
        (x.shape[-1] == 256 and rows >= 128)
        or (x.shape[-1] == 2048 and 512 <= rows <= 1024)
    )


def _can_use_quack_layernorm(x, weight, bias):
    rows = x.numel() // x.shape[-1]
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-1] == 256
        and rows >= 128
        and weight.shape == (256,)
        and (bias is None or bias.shape == weight.shape)
        and x.stride(-1) == 1
        and weight.stride(-1) == 1
        and (bias is None or bias.stride(-1) == 1)
        and (
            not torch.is_grad_enabled()
            or not any(
                tensor is not None and tensor.requires_grad
                for tensor in (x, weight, bias)
            )
        )
    )


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
    if (
        is_rms_norm
        and residual is None
        and x1 is None
        and weight1 is None
        and bias1 is None
        and dropout_p == 0.0
        and rowscale is None
        and not return_dropout_mask
        and _can_use_quack_rmsnorm(x, weight, bias, gated_api=False)
    ):
        from quack.rmsnorm import rmsnorm

        return rmsnorm(
            x,
            weight,
            bias,
            eps=eps,
            residual_dtype=torch.float32 if residual_in_fp32 else None,
            prenorm=prenorm,
        )

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
    if (
        residual is None
        and not prenorm
        and not residual_in_fp32
        and not is_rms_norm
        and not torch.is_autocast_enabled()
        and _can_use_quack_layernorm(x, norm_weight, norm_bias)
        and linear_weight.is_cuda
        and linear_weight.dtype == torch.bfloat16
        and linear_weight.shape[-1] == x.shape[-1]
        and linear_weight.stride(-1) == 1
        and (
            linear_bias is None
            or (
                linear_bias.is_cuda
                and linear_bias.dtype == linear_weight.dtype
                and linear_bias.shape == (linear_weight.shape[0],)
                and linear_bias.stride(-1) == 1
            )
        )
        and (
            not torch.is_grad_enabled()
            or not any(
                tensor is not None and tensor.requires_grad
                for tensor in (linear_weight, linear_bias)
            )
        )
    ):
        from quack.rmsnorm import layernorm_fwd

        original_shape = x.shape
        normalized = layernorm_fwd(
            x.reshape(-1, original_shape[-1]), norm_weight, norm_bias, eps
        ).reshape(original_shape)
        return F.linear(normalized, linear_weight, linear_bias)

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
    if (
        not is_rms_norm
        and z is None
        and group_size is None
        and _can_use_quack_layernorm(x, weight, bias)
    ):
        from quack.rmsnorm import layernorm_fwd

        original_shape = x.shape
        return layernorm_fwd(
            x.reshape(-1, original_shape[-1]), weight, bias, eps
        ).reshape(original_shape)
    from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn

    return layernorm_fn(
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm
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
        if _can_use_quack_rmsnorm(x, weight, bias, gated_api=True):
            from quack.rmsnorm import rmsnorm

            return rmsnorm(x, weight, bias, eps=eps)
        from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

        return rmsnorm_fn(x, weight, bias, eps=eps)
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

    return rmsnorm_fn(x, weight, bias, z, eps, group_size, norm_before_gate)


__all__ = [
    "layer_norm_fn_cute",
    "layer_norm_linear_fn_cute",
    "layernorm_fn_cute",
    "rmsnorm_fn_cute",
]
