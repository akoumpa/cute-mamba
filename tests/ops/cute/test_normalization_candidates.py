import inspect

import pytest
import torch


def _contract(function):
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_normalization_candidate_signatures():
    from mamba_ssm.ops.cute import normalization as candidate
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, layer_norm_linear_fn
    from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn, rmsnorm_fn

    pairs = [
        (candidate.layer_norm_fn_cute, layer_norm_fn),
        (candidate.layer_norm_linear_fn_cute, layer_norm_linear_fn),
        (candidate.layernorm_fn_cute, layernorm_fn),
        (candidate.rmsnorm_fn_cute, rmsnorm_fn),
    ]
    for actual, expected in pairs:
        assert _contract(actual) == _contract(expected)


@pytest.mark.parametrize("is_rms_norm", [False, True])
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("group_size", [None, 16])
def test_gated_normalization_candidate(
    is_rms_norm, with_gate, norm_before_gate, group_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from mamba_ssm.ops.cute.normalization import layernorm_fn_cute, rmsnorm_fn_cute
    from mamba_ssm.ops.triton.layernorm_gated import layernorm_fn, rmsnorm_fn

    torch.manual_seed(20)
    x = torch.randn(4, 3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, device="cuda", requires_grad=True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    z = torch.randn_like(x, requires_grad=True) if with_gate else None
    function = rmsnorm_fn_cute if is_rms_norm else layernorm_fn_cute
    reference = rmsnorm_fn if is_rms_norm else layernorm_fn
    args = (x, weight, bias, z, 1e-5, group_size, norm_before_gate)
    actual = function(*args)
    expected = reference(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    grad = torch.randn_like(actual)
    inputs = tuple(tensor for tensor in (x, weight, bias, z) if tensor is not None)
    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("api", ["generic", "gated"])
def test_quack_rmsnorm_promoted_route(dtype, api):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from mamba_ssm.ops.cute.normalization import (
        layer_norm_fn_cute,
        rmsnorm_fn_cute,
    )
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

    torch.manual_seed(21)
    x = torch.randn(512, 256, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(256, device="cuda", requires_grad=True)
    bias = torch.randn(256, device="cuda", requires_grad=True)
    if api == "generic":
        actual = layer_norm_fn_cute(x, weight, bias, is_rms_norm=True)
        expected = layer_norm_fn(x, weight, bias, is_rms_norm=True)
    else:
        actual = rmsnorm_fn_cute(x, weight, bias)
        expected = rmsnorm_fn(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(actual)
    inputs = (x, weight, bias)
    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)
