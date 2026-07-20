import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("prenorm", [False, True])
@pytest.mark.parametrize("residual_in_fp32", [False, True])
def test_cute_rms_norm_matches_triton(
    monkeypatch, dtype, has_bias, has_residual, prenorm, residual_in_fp32
):
    _require_cute()
    from mamba_ssm.ops.triton.layer_norm import rms_norm_fn

    torch.manual_seed(0)
    x = torch.randn(8, 2048, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(2048, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = (
        torch.randn(2048, device="cuda", dtype=torch.float32, requires_grad=True)
        if has_bias
        else None
    )
    residual = torch.randn_like(x, requires_grad=True) if has_residual else None

    monkeypatch.setenv("MAMBA_CUTE_DSL", "0")
    out_ref = rms_norm_fn(
        x,
        weight,
        bias,
        residual=residual,
        eps=1e-5,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )
    monkeypatch.setenv("MAMBA_CUTE_DSL", "1")
    out = rms_norm_fn(
        x,
        weight,
        bias,
        residual=residual,
        eps=1e-5,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )

    out_tensor = out[0] if prenorm else out
    out_ref_tensor = out_ref[0] if prenorm else out_ref
    torch.testing.assert_close(out_tensor, out_ref_tensor, rtol=1e-2, atol=1e-2)
    if prenorm:
        torch.testing.assert_close(out[1], out_ref[1], rtol=0, atol=0)

    grad = torch.randn_like(out_tensor)
    inputs = tuple(t for t in (x, weight, bias, residual) if t is not None)
    grad_outputs = (grad, torch.randn_like(out[1])) if prenorm else grad
    outputs = out if prenorm else out_tensor
    outputs_ref = out_ref if prenorm else out_ref_tensor
    grads = torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph=True)
    grads_ref = torch.autograd.grad(outputs_ref, inputs, grad_outputs)
    for actual, expected in zip(grads, grads_ref):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prenorm", [False, True])
def test_hybrid_rms_norm_matches_triton(monkeypatch, dtype, prenorm):
    _require_cute()
    from mamba_ssm.ops.triton.layer_norm import rms_norm_fn

    torch.manual_seed(1)
    x = torch.randn(256, 2048, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(2048, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(2048, device="cuda", dtype=torch.float32, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True) if prenorm else None
    inputs = tuple(t for t in (x, weight, bias, residual) if t is not None)

    monkeypatch.setenv("MAMBA_CUTE_DSL", "0")
    out_ref = rms_norm_fn(x, weight, bias, residual=residual, prenorm=prenorm, eps=1e-5)
    monkeypatch.setenv("MAMBA_CUTE_DSL", "1")
    out = rms_norm_fn(x, weight, bias, residual=residual, prenorm=prenorm, eps=1e-5)

    actual = out[0] if prenorm else out
    expected = out_ref[0] if prenorm else out_ref
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    grad = torch.randn_like(actual)
    grad_outputs = (grad, torch.randn_like(out[1])) if prenorm else grad
    grads = torch.autograd.grad(
        out if prenorm else actual, inputs, grad_outputs, retain_graph=True
    )
    grads_ref = torch.autograd.grad(
        out_ref if prenorm else expected, inputs, grad_outputs
    )
    for actual_grad, expected_grad in zip(grads, grads_ref):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)
