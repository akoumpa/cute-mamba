import inspect

import pytest
import torch


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def test_mamba3_siso_candidate_signatures():
    from mamba_ssm.ops.cute.mamba3 import mamba3_siso as candidate
    from mamba_ssm.ops.triton.mamba3 import mamba3_siso_bwd as bwd
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        mamba3_siso_combined,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step

    pairs = [
        (candidate.mamba3_siso_step_cute, mamba3_siso_step),
        (candidate.mamba3_siso_fwd_cute, mamba3_siso_fwd),
        (candidate.compute_dzdo_cute, bwd.compute_dzdo),
        (candidate.compute_dqkv_cute, bwd.compute_dqkv),
        (candidate.compute_dqktheta_cute, bwd.compute_dqktheta),
        (candidate.apply_dk_state_post_cute, bwd.apply_dk_state_post),
        (
            candidate.compute_ddt_dtrap_dinput_states_cute,
            bwd.compute_ddt_dtrap_dinput_states,
        ),
        (candidate.mamba3_siso_combined_cute, mamba3_siso_combined),
    ]
    for actual, expected in pairs:
        actual_parameters = inspect.signature(actual).parameters.values()
        expected_parameters = inspect.signature(expected).parameters.values()
        assert [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in actual_parameters
        ] == [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in expected_parameters
        ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compute_dzdo_cute(dtype):
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.mamba3_siso import compute_dzdo_cute
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo

    torch.manual_seed(10)
    tensors = [torch.randn(2, 71, 3, 32, device="cuda", dtype=dtype) for _ in range(3)]
    actual = compute_dzdo_cute(*tensors)
    expected = compute_dzdo(*tensors)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-3, 3e-3)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


@pytest.mark.parametrize("qk_dim,v_dim,angle_dim", [(32, 16, 8), (128, 64, 32)])
@pytest.mark.parametrize("with_optional", [False, True])
def test_mamba3_siso_step_cute(with_optional, qk_dim, v_dim, angle_dim):
    _require_cuda()
    from mamba_ssm.ops.cute.mamba3.mamba3_siso import mamba3_siso_step_cute
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step

    torch.manual_seed(11)
    batch, nheads_qk, nheads = 2, 2, 4
    q = torch.randn(batch, nheads_qk, qk_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, nheads, v_dim, device="cuda", dtype=torch.bfloat16)
    adt = -torch.rand(batch, nheads, device="cuda")
    dt = torch.rand(batch, nheads, device="cuda")
    trap = torch.randn(batch, nheads, device="cuda")
    q_bias = torch.randn(nheads, qk_dim, device="cuda", dtype=torch.bfloat16)
    k_bias = torch.randn_like(q_bias)
    angles = torch.randn(batch, nheads, angle_dim, device="cuda")
    states = (
        torch.rand(batch, nheads, angle_dim, device="cuda") * 2 * torch.pi,
        torch.randn(batch, nheads, v_dim, qk_dim, device="cuda"),
        torch.randn(batch, nheads, qk_dim, device="cuda"),
        torch.randn(batch, nheads, v_dim, device="cuda"),
    )
    d = torch.randn(nheads, device="cuda") if with_optional else None
    z = torch.randn_like(v) if with_optional else None
    args = (q, k, v, adt, dt, trap, q_bias, k_bias, angles, d, z, states)
    actual = mamba3_siso_step_cute(*args)
    expected = mamba3_siso_step(*args)
    output_atol = 1e-1 if qk_dim >= 128 else 5e-2
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=output_atol)
    for actual_tensor, expected_tensor in zip(actual[1], expected[1]):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=3e-2, atol=5e-2)
