import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_heads,num_groups", [(8, 8), (16, 4), (64, 4)])
def test_reduce_grouped_qk_grads_cute(dtype, num_heads, num_groups):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.grouped_head_reduction import (
        reduce_grouped_qk_grads_cute,
    )
    from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import (
        reduce_grouped_qk_grads,
    )

    torch.manual_seed(0)
    x = torch.randn(2, 17, 4, num_heads, 64, device="cuda", dtype=dtype)
    actual = reduce_grouped_qk_grads_cute(x, num_groups)
    expected = reduce_grouped_qk_grads(x, num_groups)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_heads,num_groups", [(8, 8), (16, 4), (64, 4)])
def test_reduce_grouped_qk_grads_and_bias_cute(dtype, num_heads, num_groups):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.grouped_head_reduction import (
        reduce_grouped_qk_grads_and_bias_cute,
    )
    from mamba_ssm.ops.triton.mamba3.grouped_head_reduction import (
        _reduce_grouped_qk_grads_and_bias_triton,
    )

    torch.manual_seed(1)
    dq = torch.randn(2, 129, 2, num_heads, 32, device="cuda", dtype=dtype)
    dk = torch.randn_like(dq)
    actual = reduce_grouped_qk_grads_and_bias_cute(dq, dk, num_groups)
    expected = _reduce_grouped_qk_grads_and_bias_triton(dq, dk, num_groups)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-3, atol=4e-3)


def test_reduce_grouped_qk_grads_dispatches_proven_fp32_shape(monkeypatch):
    _require_cute()
    import mamba_ssm.ops.cute.mamba3.grouped_head_reduction as cute_reduction
    import mamba_ssm.ops.triton.mamba3.grouped_head_reduction as triton_reduction

    sentinel = object()
    monkeypatch.setattr(triton_reduction, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(
        cute_reduction, "reduce_grouped_qk_grads_cute", lambda *args: sentinel
    )

    grads = torch.empty(8, 1024, 1, 32, 64, device="cuda", dtype=torch.float32)
    assert triton_reduction.reduce_grouped_qk_grads(grads, 8) is sentinel


def test_grouped_qk_grads_and_bias_dispatches_bf16_identity_groups(monkeypatch):
    _require_cute()
    import mamba_ssm.ops.cute.mamba3.grouped_head_reduction as cute_reduction
    import mamba_ssm.ops.triton.mamba3.grouped_head_reduction as triton_reduction

    sentinel = (object(), object(), object(), object())
    monkeypatch.setattr(triton_reduction, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(
        cute_reduction,
        "reduce_grouped_qk_grads_and_bias_cute",
        lambda *args: sentinel,
    )
    grads = torch.empty(1, 32, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    assert (
        triton_reduction.reduce_grouped_qk_grads_and_bias_triton(grads, grads, 8)
        is sentinel
    )


def test_grouped_qk_grads_and_bias_nonidentity_uses_triton(monkeypatch):
    _require_cute()
    import mamba_ssm.ops.triton.mamba3.grouped_head_reduction as triton_reduction

    sentinel = (object(), object(), object(), object())
    monkeypatch.setattr(
        triton_reduction,
        "_reduce_grouped_qk_grads_and_bias_triton",
        lambda *args: sentinel,
    )
    grads = torch.empty(1, 32, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
    assert (
        triton_reduction.reduce_grouped_qk_grads_and_bias_triton(grads, grads, 4)
        is sentinel
    )
