import inspect

import pytest
import torch


def _contract(function):
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_selective_scan_candidate_signatures():
    from mamba_ssm.ops.cute.selective_scan import mamba_inner_cute, selective_scan_cute
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

    assert _contract(selective_scan_cute) == _contract(selective_scan_fn)
    assert _contract(mamba_inner_cute) == _contract(mamba_inner_fn)


@pytest.mark.parametrize("variable_bc", [False, True])
def test_selective_scan_candidate_forward_backward(variable_bc):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from mamba_ssm.ops.cute.selective_scan import selective_scan_cute
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    torch.manual_seed(30)
    batch, dim, seqlen, dstate = 2, 4, 19, 3
    u = torch.randn(batch, dim, seqlen, device="cuda", requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    a = -torch.rand(dim, dstate, device="cuda", requires_grad=True)
    if variable_bc:
        b = torch.randn(batch, dstate, seqlen, device="cuda", requires_grad=True)
        c = torch.randn_like(b, requires_grad=True)
    else:
        b = torch.randn(dim, dstate, device="cuda", requires_grad=True)
        c = torch.randn_like(b, requires_grad=True)
    d = torch.randn(dim, device="cuda", requires_grad=True)
    z = torch.randn_like(u, requires_grad=True)
    delta_bias = torch.randn(dim, device="cuda", requires_grad=True)
    args = (u, delta, a, b, c, d, z, delta_bias, True, True)
    actual, actual_last = selective_scan_cute(*args)
    expected, expected_last = selective_scan_ref(*args)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_last, expected_last)
    assert not actual_last.requires_grad

    inputs = (u, delta, a, b, c, d, z, delta_bias)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("bc_layout", ["constant", "batch", "grouped"])
def test_selective_scan_candidate_bounded_inference(bc_layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("cutlass.cute")
    from mamba_ssm.ops.cute.selective_scan import selective_scan_cute
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    torch.manual_seed(31)
    batch, dim, seqlen, dstate = 2, 8, 4, 32
    u = torch.randn(batch, dim, seqlen, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(u)
    a = -torch.rand(dim, dstate, device="cuda")
    if bc_layout == "constant":
        b = torch.randn(dim, dstate, device="cuda", dtype=torch.bfloat16)
        c = torch.randn_like(b)
    elif bc_layout == "batch":
        b = torch.randn(batch, dstate, seqlen, device="cuda", dtype=torch.bfloat16)
        c = torch.randn_like(b)
    else:
        b = torch.randn(batch, 2, dstate, seqlen, device="cuda", dtype=torch.bfloat16)
        c = torch.randn_like(b)
    d = torch.randn(dim, device="cuda")
    z = torch.randn_like(u)
    delta_bias = torch.randn(dim, device="cuda")
    args = (u, delta, a, b, c, d, z, delta_bias, True, True)
    with torch.no_grad():
        actual, actual_last = selective_scan_cute(*args)
        expected, expected_last = selective_scan_ref(*args)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_last, expected_last, rtol=2e-2, atol=2e-2)
    assert not actual_last.requires_grad
