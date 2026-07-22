import pytest
import torch


def _require_cute():
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("quack")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("rotate_pairwise", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_apply_rotary_qk_inference_fwd_cute(
    dtype, with_bias, conjugate, rotate_pairwise, inplace
):
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd_cute,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_reference,
    )

    torch.manual_seed(0)
    batch, mimo_dim, nheads, headdim, rotary_dim = 2, 3, 4, 32, 16
    q = torch.randn(batch, mimo_dim, nheads, headdim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    angle_state = torch.randn(batch, nheads, rotary_dim // 2, device="cuda")
    angle_proj = torch.randn_like(angle_state)
    dt = torch.randn(batch, nheads, device="cuda")
    bias_q = (
        torch.randn(mimo_dim, nheads, headdim, device="cuda", dtype=dtype)
        if with_bias
        else None
    )
    bias_k = torch.randn_like(bias_q) if with_bias else None
    expected = apply_rotary_qk_inference_reference(
        q, k, angle_state, angle_proj, dt, bias_q, bias_k, conjugate, rotate_pairwise
    )
    expected = (expected[0].to(dtype), expected[1].to(dtype), expected[2])
    q_input, k_input, angle_input = q.clone(), k.clone(), angle_state.clone()
    actual = apply_rotary_qk_inference_fwd_cute(
        q_input,
        k_input,
        angle_input,
        angle_proj,
        dt,
        bias_q,
        bias_k,
        inplace,
        conjugate,
        rotate_pairwise,
    )
    if inplace:
        assert actual[0].data_ptr() == q_input.data_ptr()
        assert actual[1].data_ptr() == k_input.data_ptr()
        assert actual[2].data_ptr() == angle_input.data_ptr()
    qk_rtol, qk_atol = (1e-2, 4e-2) if dtype == torch.bfloat16 else (5e-3, 7e-3)
    torch.testing.assert_close(actual[0], expected[0], rtol=qk_rtol, atol=qk_atol)
    torch.testing.assert_close(actual[1], expected[1], rtol=qk_rtol, atol=qk_atol)
    torch.testing.assert_close(actual[2], expected[2], rtol=5e-4, atol=7e-4)


def test_apply_rotary_qk_inference_fwd_cute_expanded_heads():
    _require_cute()
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd_cute,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_reference,
    )

    batch, mimo_dim, nheads, headdim = 1, 2, 4, 16
    q = torch.randn(batch, mimo_dim, 1, headdim, device="cuda").expand(
        batch, mimo_dim, nheads, headdim
    )
    k = torch.randn(batch, mimo_dim, 1, headdim, device="cuda").expand_as(q)
    angle = torch.randn(batch, nheads, headdim // 2, device="cuda")
    proj = torch.randn_like(angle)
    dt = torch.randn(batch, nheads, device="cuda")
    actual = apply_rotary_qk_inference_fwd_cute(q, k, angle, proj, dt)
    expected = apply_rotary_qk_inference_reference(q, k, angle, proj, dt)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=5e-4, atol=7e-4)


def test_mamba3_step_dispatches_rotary_to_cute():
    _require_cute()
    from mamba_ssm.modules import mamba3
    from mamba_ssm.ops.cute.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd_cute,
    )

    assert mamba3.apply_rotary_qk_inference_fwd is apply_rotary_qk_inference_fwd_cute
