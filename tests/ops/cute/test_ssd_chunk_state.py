import pytest
import torch


def _require_cute():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer CUDA GPU is required")
    pytest.importorskip("cutlass.cute")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "chunk_size,has_bias,dt_softplus,dt_limit",
    [
        (32, False, False, (0.0, float("inf"))),
        (64, True, True, (0.001, 0.2)),
        (128, True, False, (0.0, float("inf"))),
    ],
)
def test_chunk_cumsum_fwd_cute(dtype, chunk_size, has_bias, dt_softplus, dt_limit):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_chunk_state import chunk_cumsum_fwd_cute
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd

    torch.manual_seed(0)
    batch, seqlen, nheads = 3, chunk_size + 17, 7
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=dtype)
    A = -torch.rand(nheads, device="cuda")
    dt_bias = torch.randn(nheads, device="cuda") if has_bias else None
    actual_dA, actual_dt = chunk_cumsum_fwd_cute(
        dt, A, chunk_size, dt_bias, dt_softplus, dt_limit
    )
    expected_dA, expected_dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias, dt_softplus, dt_limit
    )

    torch.testing.assert_close(actual_dt, expected_dt, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_dA, expected_dA, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "chunk_size,has_bias,dt_softplus,dt_limit",
    [
        (32, False, False, (0.0, float("inf"))),
        (64, True, True, (0.001, 0.2)),
        (128, True, False, (0.0, float("inf"))),
    ],
)
def test_chunk_cumsum_bwd_cute(dtype, chunk_size, has_bias, dt_softplus, dt_limit):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_chunk_state import chunk_cumsum_bwd_cute
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_bwd

    torch.manual_seed(1)
    batch, seqlen, nheads = 3, chunk_size + 17, 7
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=dtype)
    A = -torch.rand(nheads, device="cuda")
    dt_bias = torch.randn(nheads, device="cuda") if has_bias else None
    ddA = torch.randn(batch, nheads, nchunks, chunk_size, device="cuda")
    ddt_out = torch.randn_like(ddA)
    ddt_buffer = torch.empty_like(dt)
    actual = chunk_cumsum_bwd_cute(
        ddA, ddt_out, dt, A, dt_bias, dt_softplus, dt_limit, ddt_buffer
    )
    expected = _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias, dt_softplus, dt_limit)
    assert actual[0].data_ptr() == ddt_buffer.data_ptr()
    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor is not None:
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=3e-3, atol=4e-3
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_seq_idx", [False, True])
@pytest.mark.parametrize("states_in_fp32", [False, True])
def test_chunk_state_fwd_cute(dtype, has_seq_idx, states_in_fp32):
    _require_cute()
    from mamba_ssm.ops.cute.ssd_chunk_state import chunk_state_fwd_cute
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd

    torch.manual_seed(2)
    batch, seqlen, nheads, ngroups = 2, 83, 4, 2
    headdim, dstate, chunk_size = 17, 19, 32
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    x = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    B = torch.randn(batch, seqlen, ngroups, dstate, device="cuda", dtype=dtype)
    dt = torch.rand(batch, nheads, nchunks, chunk_size, device="cuda")
    dA = -torch.rand_like(dt).cumsum(-1)
    seq_idx = None
    if has_seq_idx:
        seq_idx = torch.zeros(batch, seqlen, device="cuda", dtype=torch.int32)
        seq_idx[:, 47:] = 1
    output_dtype = torch.float32 if states_in_fp32 else dtype
    buffer = torch.empty(
        batch, nchunks, nheads, headdim, dstate, device="cuda", dtype=output_dtype
    )
    actual = chunk_state_fwd_cute(B, x, dt, dA, seq_idx, buffer, states_in_fp32)
    expected = _chunk_state_fwd(B, x, dt, dA, seq_idx, states_in_fp32=states_in_fp32)
    assert actual.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=7e-3)
